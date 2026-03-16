"""
Continuous batching for memopt.

Static batching (bad for GPU utilization):
  - Wait for batch of N requests
  - Run all N together
  - All must finish before next batch starts
  - Short requests waste GPU waiting for long ones

Continuous batching (better):
  - Accept requests as they arrive
  - Fill batch slots immediately when one finishes
  - GPU stays busy processing different-length sequences
  - 2-4x throughput improvement at realistic concurrency

Key implementation detail:
  - ONE model call per scheduling step for ALL pending requests
  - Left-padding aligns last real token to position max_len-1
  - output_ids stored per-request is always UNPADDED
  - Attention mask marks real tokens as 1, padding as 0

This is pure Python asyncio.
No custom CUDA kernels required.

Kernel hooks (Pillar 3):
  This module handles request scheduling and batch assembly — it does not
  perform attention computation directly. The model's forward pass (called
  via _run_model_step) is where RoPE, layer norm, and softmax execute.
  To wire fused kernel hooks into inference, apply them inside the model's
  forward method or wrap the model before passing it to ContinuousBatchingEngine.
  See memopt/serving/kernel_hooks.py for the apply_rope / apply_layer_norm_residual
  / apply_scaled_softmax hook API.
"""
import asyncio
import torch
import torch.nn as nn
import time
import uuid
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

from .paged_attention import PagedKVCache

log = logging.getLogger(__name__)


class RequestStatus(Enum):
    QUEUED    = "queued"
    RUNNING   = "running"
    DONE      = "done"
    FAILED    = "failed"


@dataclass
class Request:
    request_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    input_ids: torch.Tensor = None
    max_new_tokens: int = 100
    temperature: float = 1.0
    status: RequestStatus = RequestStatus.QUEUED
    output_ids: Optional[torch.Tensor] = None
    tokens_generated: int = 0
    queued_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None

    @property
    def latency_ms(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at) * 1000
        return None

    @property
    def wait_ms(self) -> Optional[float]:
        if self.started_at:
            return (self.started_at - self.queued_at) * 1000
        return None


@dataclass
class BatchingConfig:
    max_batch_size: int = 8
    max_queue_size: int = 256
    max_wait_ms: float = 50.0
    max_seq_len: int = 2048
    schedule_interval_ms: float = 5.0


class ContinuousBatchingEngine:
    """
    Continuous batching inference engine for memopt.

    True dynamic batching: ONE model call per step for ALL pending requests.
    Variable-length sequences are left-padded to the same length so that
    the last real token always sits at position max_len-1 — the position
    causal attention uses to predict the next token.

    Two usage modes:

    1. Async (production):
        engine = ContinuousBatchingEngine(model, config)
        await engine.start()
        result = await engine.submit(input_ids)
        await engine.stop()

    2. Sync (testing/benchmarking):
        results = engine.run_sync(list_of_input_ids, max_new_tokens=50)
    """

    def __init__(
        self,
        model: nn.Module,
        config: BatchingConfig = None,
        tokenizer=None,
        pad_token_id: int = 0,
        eos_token_ids: Optional[List[int]] = None,
        use_paged_attention: bool = False,
    ):
        self.model = model
        self.config = config or BatchingConfig()
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id
        # Use tokenizer eos if available, else common EOS ids
        if tokenizer and hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
            self.eos_token_ids = [tokenizer.eos_token_id]
        else:
            self.eos_token_ids = eos_token_ids or [2, 1, 50256, 32000]
        model_cfg = getattr(model, "config", None)
        if use_paged_attention:
            _device = "cuda" if torch.cuda.is_available() else "cpu"
            _dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            _n_heads = getattr(model_cfg, "num_attention_heads", 12)
            self.paged_kv_cache: Optional[PagedKVCache] = PagedKVCache(
                num_blocks=self.config.max_batch_size * 64,
                num_layers=getattr(model_cfg, "num_hidden_layers", 12),
                num_heads=_n_heads,
                head_dim=getattr(model_cfg, "hidden_size", 768) // _n_heads,
                dtype=_dtype,
                device=_device,
            )
        else:
            self.paged_kv_cache: Optional[PagedKVCache] = None

        self._queue: asyncio.Queue = None
        self._active: Dict[str, Request] = {}
        self._completed: Dict[str, Request] = {}
        self._running = False
        self._worker_task = None

        # Metrics
        self.total_requests = 0
        self.total_tokens = 0
        self.start_time = None

    async def start(self):
        """Start the batching engine."""
        self._queue = asyncio.Queue(maxsize=self.config.max_queue_size)
        self._running = True
        self.start_time = time.time()
        self._worker_task = asyncio.create_task(self._scheduling_loop())
        log.info(
            f"ContinuousBatchingEngine started | "
            f"max_batch={self.config.max_batch_size} | "
            f"max_wait={self.config.max_wait_ms}ms"
        )

    async def stop(self):
        """Stop the engine gracefully."""
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        log.info(
            f"Engine stopped | "
            f"total_requests={self.total_requests} | "
            f"total_tokens={self.total_tokens}"
        )

    async def submit(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
    ) -> Request:
        """
        Submit a request and wait for completion.
        Returns completed Request with output_ids.
        """
        req = Request(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        await self._queue.put(req)
        self.total_requests += 1

        # Wait for completion
        while req.status not in (RequestStatus.DONE, RequestStatus.FAILED):
            await asyncio.sleep(0.001)

        return req

    def _run_batch_step(self, pending: List[Request]) -> List[Request]:
        """
        Execute ONE batched forward pass for all pending requests.

        Left-pads sequences so all have the same length.
        After left-padding, last real token is always at position max_len-1.
        output_ids per-request stays UNPADDED between steps.

        Returns list of requests that are still running.
        """
        device = pending[0].output_ids.device
        dtype = pending[0].output_ids.dtype

        # Compute lengths and max
        lengths = [req.output_ids.shape[1] for req in pending]
        max_len = max(lengths)

        padded_ids_list: List[torch.Tensor] = []
        attention_masks_list: List[torch.Tensor] = []

        for req, seq_len in zip(pending, lengths):
            pad_len = max_len - seq_len
            if pad_len > 0:
                # Left-pad: padding goes BEFORE the real tokens
                pad_tensor = torch.full(
                    (1, pad_len), self.pad_token_id,
                    dtype=dtype, device=device,
                )
                padded = torch.cat([pad_tensor, req.output_ids], dim=1)
            else:
                padded = req.output_ids

            # Attention mask: 0 for padding, 1 for real tokens
            mask = torch.zeros(1, max_len, dtype=torch.long, device=device)
            mask[0, pad_len:] = 1

            padded_ids_list.append(padded)
            attention_masks_list.append(mask)

        # Stack into single batch — shape: (N, max_len)
        batch_ids = torch.cat(padded_ids_list, dim=0)
        batch_mask = torch.cat(attention_masks_list, dim=0)

        # ONE model call for all pending requests
        with torch.no_grad():
            out = self.model(input_ids=batch_ids, attention_mask=batch_mask)

        still_running: List[Request] = []

        for i, req in enumerate(pending):
            # After left-padding, last real token sits at max_len-1
            logits_i = out.logits[i, max_len - 1, :]  # (vocab_size,)

            if req.temperature == 0 or req.temperature < 1e-6:
                next_token_id = logits_i.argmax(dim=-1)
                next_token = next_token_id.view(1, 1)
            else:
                probs = torch.softmax(logits_i / req.temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).unsqueeze(0)

            # Append to UNPADDED output_ids
            req.output_ids = torch.cat([req.output_ids, next_token], dim=1)
            req.tokens_generated += 1
            self.total_tokens += 1

            is_eos = next_token.item() in self.eos_token_ids
            is_max = req.tokens_generated >= req.max_new_tokens

            if is_eos or is_max:
                req.status = RequestStatus.DONE
                req.completed_at = time.time()
            else:
                still_running.append(req)

        return still_running

    async def _scheduling_loop(self):
        """
        Main scheduling loop.

        Drains queue into pending list, then calls _run_batch_step()
        which does a SINGLE batched forward pass for all pending requests.
        Completed requests are dropped from pending; remaining continue
        in the next step — this is the "continuous" part.
        """
        pending: List[Request] = []

        while self._running:
            # Fill pending slots from queue up to max_batch_size
            while len(pending) < self.config.max_batch_size:
                try:
                    req = self._queue.get_nowait()
                    req.status = RequestStatus.RUNNING
                    req.started_at = time.time()
                    # Initialize output_ids from input_ids (unpadded)
                    if req.output_ids is None:
                        req.output_ids = req.input_ids.clone()
                    pending.append(req)
                except asyncio.QueueEmpty:
                    # If we already have requests, proceed with what we have
                    if pending:
                        break
                    # Otherwise wait for at least one request
                    await asyncio.sleep(self.config.schedule_interval_ms / 1000)

            if not pending:
                await asyncio.sleep(self.config.schedule_interval_ms / 1000)
                continue

            # ONE batched forward pass for all pending requests
            try:
                pending = self._run_batch_step(pending)
            except Exception as e:
                log.error(f"Batch step failed: {e}", exc_info=True)
                for req in pending:
                    req.status = RequestStatus.FAILED
                    req.error = str(e)
                pending = []

            # Yield to event loop so submit() waiters can check status
            await asyncio.sleep(0)

    def run_sync(
        self,
        input_ids_list: List[torch.Tensor],
        max_new_tokens: int = 50,
        temperature: float = 1.0,
    ) -> List[Request]:
        """
        Synchronous interface for benchmarking.
        Runs all requests through the engine, returns when all done.
        """
        async def _run():
            await self.start()
            tasks = [
                self.submit(ids, max_new_tokens, temperature)
                for ids in input_ids_list
            ]
            results = await asyncio.gather(*tasks)
            await self.stop()
            return list(results)

        return asyncio.run(_run())

    def throughput_rps(self) -> float:
        """Requests per second since engine started."""
        if not self.start_time or self.total_requests == 0:
            return 0.0
        elapsed = time.time() - self.start_time
        return self.total_requests / elapsed

    def tokens_per_second(self) -> float:
        """Tokens generated per second."""
        if not self.start_time or self.total_tokens == 0:
            return 0.0
        elapsed = time.time() - self.start_time
        return self.total_tokens / elapsed

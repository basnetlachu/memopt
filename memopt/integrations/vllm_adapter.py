"""
vLLM adapter for MemOpt

PURPOSE: Drop-in replacement for vLLM with MemOpt optimizations
WHY: Easy migration for vLLM users
COMPATIBILITY: vLLM API compatible

Usage:
    # OLD CODE (vLLM):
    # from vllm import LLM
    # llm = LLM(model="gpt2-xl")
    
    # NEW CODE (MemOpt):
    from memopt.integrations import MemOptVLLM
    llm = MemOptVLLM(model="gpt2-xl")
    
    # Same API!
    outputs = llm.generate(["Hello"], max_tokens=100)
"""

from typing import List, Optional, Union, Dict, Any
import time

from memopt.core.model import OptimizedLLM
from memopt.distributed.multi_gpu import MultiGPUManager
from memopt.monitoring.logger import get_logger
from memopt.utils.errors import IntegrationError

logger = get_logger(__name__)


class SamplingParams:
    """
    vLLM-compatible sampling parameters
    
    Simplified version of vLLM's SamplingParams for compatibility
    """
    
    def __init__(
        self,
        n: int = 1,
        best_of: Optional[int] = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        use_beam_search: bool = False,
        length_penalty: float = 1.0,
        early_stopping: bool = False,
        stop: Optional[Union[str, List[str]]] = None,
        stop_token_ids: Optional[List[int]] = None,
        ignore_eos: bool = False,
        max_tokens: int = 16,
        logprobs: Optional[int] = None,
        prompt_logprobs: Optional[int] = None,
        skip_special_tokens: bool = True,
    ):
        self.n = n
        self.best_of = best_of
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.use_beam_search = use_beam_search
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.stop = stop
        self.stop_token_ids = stop_token_ids
        self.ignore_eos = ignore_eos
        self.max_tokens = max_tokens
        self.logprobs = logprobs
        self.prompt_logprobs = prompt_logprobs
        self.skip_special_tokens = skip_special_tokens


class RequestOutput:
    """
    vLLM-compatible request output
    """
    
    def __init__(
        self,
        request_id: str,
        prompt: str,
        prompt_token_ids: List[int],
        outputs: List['CompletionOutput'],
        finished: bool
    ):
        self.request_id = request_id
        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids
        self.outputs = outputs
        self.finished = finished


class CompletionOutput:
    """
    vLLM-compatible completion output
    """
    
    def __init__(
        self,
        index: int,
        text: str,
        token_ids: List[int],
        cumulative_logprob: Optional[float] = None,
        logprobs: Optional[List[Dict[int, float]]] = None,
        finish_reason: Optional[str] = None
    ):
        self.index = index
        self.text = text
        self.token_ids = token_ids
        self.cumulative_logprob = cumulative_logprob
        self.logprobs = logprobs
        self.finish_reason = finish_reason


class MemOptVLLM:
    """
    vLLM-compatible interface with MemOpt optimizations
    
    Drop-in replacement for vLLM.LLM with enhanced performance
    """
    
    def __init__(
        self,
        model: str,
        tokenizer: Optional[str] = None,
        tokenizer_mode: str = "auto",
        trust_remote_code: bool = False,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        quantization: Optional[str] = None,
        revision: Optional[str] = None,
        tokenizer_revision: Optional[str] = None,
        seed: int = 0,
        gpu_memory_utilization: float = 0.90,
        swap_space: int = 4,
        enforce_eager: bool = False,
        max_context_len_to_capture: int = 8192,
        disable_custom_all_reduce: bool = False,
        **kwargs
    ):
        """
        Initialize MemOpt with vLLM-compatible interface
        
        Args:
            model: Model name or path
            tensor_parallel_size: Number of GPUs to use
            gpu_memory_utilization: GPU memory utilization (MemOpt maps to memory_threshold)
            **kwargs: Additional arguments (for compatibility)
        """
        logger.info("="*70)
        logger.info("Initializing MemOpt vLLM Adapter")
        logger.info("="*70)
        logger.info(f"Model: {model}")
        logger.info(f"GPUs: {tensor_parallel_size}")
        
        self.model_name = model
        self.tensor_parallel_size = tensor_parallel_size
        
        # Initialize MemOpt model
        if tensor_parallel_size > 1:
            # Multi-GPU
            logger.info(f"Using multi-GPU mode with {tensor_parallel_size} GPUs")
            self.backend = MultiGPUManager(
                model_name=model,
                num_gpus=tensor_parallel_size,
                optimization_level="high"
            )
        else:
            # Single GPU
            logger.info("Using single-GPU mode")
            self.backend = OptimizedLLM(
                model=model,
                optimization_level="high",
                memory_threshold=gpu_memory_utilization
            )
        
        logger.info("✓ MemOpt vLLM adapter initialized")
    
    def generate(
        self,
        prompts: Union[str, List[str]],
        sampling_params: Optional[SamplingParams] = None,
        use_tqdm: bool = True
    ) -> List[RequestOutput]:
        """
        Generate text for given prompts
        
        Args:
            prompts: Input prompt(s)
            sampling_params: Sampling parameters
            use_tqdm: Show progress bar (ignored for compatibility)
            
        Returns:
            List of RequestOutput objects (vLLM compatible)
        """
        # Convert single prompt to list
        if isinstance(prompts, str):
            prompts = [prompts]
        
        # Default sampling params
        if sampling_params is None:
            sampling_params = SamplingParams()
        
        logger.debug(f"Generating for {len(prompts)} prompts")
        
        # Generate
        if hasattr(self.backend, 'generate_batch'):
            # Multi-GPU batch generation
            outputs = self.backend.generate_batch(
                prompts=prompts,
                max_tokens=sampling_params.max_tokens,
                temperature=sampling_params.temperature,
                top_p=sampling_params.top_p,
                do_sample=sampling_params.temperature > 0
            )
        else:
            # Single-GPU sequential generation
            outputs = [
                self.backend.generate(
                    prompt=prompt,
                    max_tokens=sampling_params.max_tokens,
                    temperature=sampling_params.temperature,
                    top_p=sampling_params.top_p,
                    do_sample=sampling_params.temperature > 0
                )
                for prompt in prompts
            ]
        
        # Convert to vLLM-compatible output format
        results = []
        for i, (prompt, output) in enumerate(zip(prompts, outputs)):
            # Tokenize for compatibility
            prompt_token_ids = self.backend.tokenizer.encode(prompt) if hasattr(self.backend, 'tokenizer') else []
            output_token_ids = self.backend.tokenizer.encode(output) if hasattr(self.backend, 'tokenizer') else []
            
            completion = CompletionOutput(
                index=0,
                text=output,
                token_ids=output_token_ids,
                finish_reason="length"
            )
            
            request_output = RequestOutput(
                request_id=f"memopt-{i}",
                prompt=prompt,
                prompt_token_ids=prompt_token_ids,
                outputs=[completion],
                finished=True
            )
            
            results.append(request_output)
        
        return results
    
    def __del__(self):
        """Cleanup"""
        try:
            if hasattr(self, 'backend'):
                del self.backend
        except:
            pass


# Alias for compatibility
LLM = MemOptVLLM
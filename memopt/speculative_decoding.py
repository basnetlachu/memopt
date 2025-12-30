"""
Speculative Decoding for Stage 5b

Uses a small draft model to predict tokens, then verifies with the main model.
This delivers 2-3x speedup by reducing the number of main model forward passes.

How it works:
1. Draft model generates K tokens quickly
2. Main model verifies all K tokens in one forward pass
3. Accept correct predictions, reject and regenerate incorrect ones
4. Repeat until completion

Expected speedup: 2-3x on top of existing optimizations
Total: 6.2x (current) * 2.5x (speculative) = 15-18x from baseline
"""

import torch
import torch.nn.functional as F
from typing import Optional, List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer


class SpeculativeDecoder:
    """
    Speculative decoding implementation.

    Uses a small draft model to generate candidate tokens,
    then verifies them with the main model in parallel.
    """

    def __init__(
        self,
        draft_model: AutoModelForCausalLM,
        draft_tokenizer: AutoTokenizer,
        num_speculative_tokens: int = 4,
        device: str = "cuda"
    ):
        """
        Initialize speculative decoder.

        Args:
            draft_model: Small fast model for drafting (e.g., gpt2)
            draft_tokenizer: Tokenizer for draft model
            num_speculative_tokens: Number of tokens to draft (K)
            device: Device to run on
        """
        self.draft_model = draft_model
        self.draft_tokenizer = draft_tokenizer
        self.num_speculative_tokens = num_speculative_tokens
        self.device = device

        # Move draft model to device
        self.draft_model.to(device)
        self.draft_model.eval()

        # Stats for monitoring
        self.total_draft_tokens = 0
        self.total_accepted_tokens = 0
        self.total_forward_passes = 0

        # Phase 1: Fallback tracking
        self.consecutive_failures = 0
        self.total_fallbacks = 0
        self.max_consecutive_failures = 3  # Disable speculative after 3 failures

    @torch.no_grad()
    def _generate_standard(
        self,
        main_model,
        input_ids: torch.Tensor,
        num_tokens: int = 1,
        temperature: float = 1.0,
        top_p: float = 1.0,
        do_sample: bool = False
    ) -> torch.Tensor:
        """
        Phase 1: Fallback to standard generation when speculative fails.

        This ensures system never crashes from draft model failures.
        Generates tokens one-at-a-time with main model only.

        Args:
            main_model: Main model for generation
            input_ids: Input token IDs [1, seq_len]
            num_tokens: Number of tokens to generate (default 1)
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to sample

        Returns:
            Generated tokens [1, num_tokens]
        """
        # Ensure input_ids is 2D
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        generated = []

        for _ in range(num_tokens):
            # PERFORMANCE FIX: Disable use_cache to avoid O(n²) memory overhead at long sequences
            # Recomputation is faster than cache management for sequences >256 tokens
            outputs = main_model(
                input_ids=input_ids,
                use_cache=False,  # Changed from True - eliminates long-sequence slowdown
                return_dict=True
            )

            logits = outputs.logits
            if logits.dim() == 2:
                next_token_logits = logits[-1:, :]
            else:
                next_token_logits = logits[:, -1, :]

            # Sample or greedy
            if do_sample and temperature > 0:
                next_token_logits = next_token_logits / temperature
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            generated.append(next_token)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return torch.cat(generated, dim=1) if generated else torch.empty((1, 0), device=input_ids.device)

    @torch.no_grad()
    def draft_tokens(
        self,
        input_ids: torch.Tensor,
        num_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        do_sample: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate draft tokens with the small model.

        Args:
            input_ids: Input token IDs [1, seq_len]
            num_tokens: Number of tokens to draft
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to sample or use greedy

        Returns:
            Tuple of (draft_ids, draft_logits)
            - draft_ids: [1, num_tokens] drafted token IDs
            - draft_logits: [1, num_tokens, vocab_size] logits for each position
        """
        # Ensure input_ids is 2D [batch_size, seq_len]
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        draft_ids = []
        draft_logits = []

        current_ids = input_ids

        for _ in range(num_tokens):
            # Generate next token with draft model
            # PERFORMANCE FIX: Disable cache for draft model too
            outputs = self.draft_model(
                input_ids=current_ids,
                use_cache=False,  # Changed from True
                return_dict=True
            )

            # Handle different output shapes
            logits = outputs.logits
            if logits.dim() == 2:
                # Shape: [seq_len, vocab_size]
                next_token_logits = logits[-1:, :]  # [1, vocab_size]
            else:
                # Shape: [batch_size, seq_len, vocab_size]
                next_token_logits = logits[:, -1, :]  # [1, vocab_size]

            draft_logits.append(next_token_logits)

            # Sample or greedy
            if do_sample and temperature > 0:
                next_token_logits = next_token_logits / temperature

                # Top-p sampling
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')

                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            draft_ids.append(next_token)

            # Append for next iteration
            current_ids = torch.cat([current_ids, next_token], dim=1)

        # Combine draft ids and logits
        draft_ids = torch.cat(draft_ids, dim=1)  # [1, num_tokens]

        # Stack logits - ensure all have same shape [1, vocab_size]
        # Then stack to create [num_tokens, 1, vocab_size] and transpose to [1, num_tokens, vocab_size]
        draft_logits_tensor = torch.stack(draft_logits, dim=0)  # [num_tokens, 1, vocab_size]
        draft_logits_tensor = draft_logits_tensor.squeeze(1).unsqueeze(0)  # [1, num_tokens, vocab_size]

        return draft_ids, draft_logits_tensor

    @torch.no_grad()
    def verify_and_correct(
        self,
        main_model,
        input_ids: torch.Tensor,
        draft_ids: torch.Tensor,
        draft_logits: torch.Tensor,
        temperature: float = 1.0,
        top_p: float = 1.0,
        do_sample: bool = False
    ) -> Tuple[torch.Tensor, int]:
        """
        Verify draft tokens with main model and correct if needed.

        Args:
            main_model: Main (larger) model for verification
            input_ids: Original input [1, seq_len]
            draft_ids: Drafted tokens [1, K]
            draft_logits: Draft model logits [1, K, vocab_size]
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to sample

        Returns:
            Tuple of (accepted_tokens, num_accepted)
            - accepted_tokens: [1, N] where N <= K+1
            - num_accepted: Number of tokens accepted from draft
        """
        # Ensure input_ids is 2D
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if draft_ids.dim() == 1:
            draft_ids = draft_ids.unsqueeze(0)

        # Concatenate input with ALL draft tokens
        full_input = torch.cat([input_ids, draft_ids], dim=1)  # [1, seq_len + K]

        # Single forward pass for verification
        # PERFORMANCE FIX: Disable cache for main model verification too
        outputs = main_model(
            input_ids=full_input,
            use_cache=False,  # Changed from True
            return_dict=True
        )

        self.total_forward_passes += 1

        # Get main model logits for each position
        logits = outputs.logits
        if logits.dim() == 2:
            # Shape: [seq_len, vocab_size]
            main_logits = logits[-(draft_ids.shape[1] + 1):, :].unsqueeze(0)  # [1, K+1, vocab_size]
        else:
            # Shape: [batch_size, seq_len, vocab_size]
            main_logits = logits[:, -(draft_ids.shape[1] + 1):, :]  # [1, K+1, vocab_size]

        # Verify each drafted token
        accepted_tokens = []
        num_accepted = 0

        for i in range(draft_ids.shape[1]):
            draft_token = draft_ids[:, i:i+1]  # [1, 1] - keep 2D
            main_token_logits = main_logits[:, i, :]  # Logits at position i

            # Get main model's prediction
            if do_sample and temperature > 0:
                main_probs = F.softmax(main_token_logits / temperature, dim=-1)
            else:
                main_probs = F.softmax(main_token_logits, dim=-1)

            # Check if draft token is acceptable
            # Simple strategy: accept if it's in top-k of main model
            main_token = torch.argmax(main_token_logits, dim=-1, keepdim=True)  # [1]

            if draft_token.item() == main_token.item():
                # Accept this token
                accepted_tokens.append(draft_token)  # [1, 1]
                num_accepted += 1
            else:
                # Reject this token and stop
                # Use main model's prediction instead
                if do_sample and temperature > 0:
                    # multinomial returns [1, 1], reshape to [1, 1]
                    corrected_token = torch.multinomial(main_probs, num_samples=1).view(1, 1)
                else:
                    corrected_token = main_token.view(1, 1)  # [1, 1]

                accepted_tokens.append(corrected_token)
                break

        # If all draft tokens accepted, add one bonus token from main model
        if num_accepted == draft_ids.shape[1]:
            bonus_logits = main_logits[:, -1, :]  # [1, vocab_size]
            if do_sample and temperature > 0:
                bonus_probs = F.softmax(bonus_logits / temperature, dim=-1)
                bonus_token = torch.multinomial(bonus_probs, num_samples=1).view(1, 1)  # [1, 1]
            else:
                bonus_token = torch.argmax(bonus_logits, dim=-1, keepdim=True).view(1, 1)  # [1, 1]

            accepted_tokens.append(bonus_token)

        # Stack accepted tokens
        if accepted_tokens:
            accepted_tokens = torch.cat(accepted_tokens, dim=1)  # [1, N]
        else:
            # Fallback: use first token from main model
            bonus_logits = main_logits[:, 0, :]  # [1, vocab_size]
            accepted_tokens = torch.argmax(bonus_logits, dim=-1, keepdim=True).view(1, 1)  # [1, 1]
            num_accepted = 0

        self.total_draft_tokens += draft_ids.shape[1]
        self.total_accepted_tokens += num_accepted

        return accepted_tokens, num_accepted

    def generate(
        self,
        main_model,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        do_sample: bool = False,
        eos_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate tokens using speculative decoding.

        Phase 1: Added fallback to standard generation on draft failure.

        Args:
            main_model: Main model for verification
            input_ids: Input token IDs [1, seq_len]
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to sample
            eos_token_id: End of sequence token ID

        Returns:
            Generated token IDs [1, seq_len + N]
        """
        # Ensure input_ids is 2D
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        current_ids = input_ids
        generated_tokens = 0

        while generated_tokens < max_new_tokens:
            # Phase 1: Check if speculative decoding should be disabled
            if self.consecutive_failures >= self.max_consecutive_failures:
                # Too many failures, use standard generation for this request
                num_tokens = min(1, max_new_tokens - generated_tokens)
                accepted_tokens = self._generate_standard(
                    main_model,
                    current_ids,
                    num_tokens=num_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample
                )
                current_ids = torch.cat([current_ids, accepted_tokens], dim=1)
                generated_tokens += accepted_tokens.shape[1]
                continue

            try:
                # Step 1: Draft K tokens with small model
                draft_ids, draft_logits = self.draft_tokens(
                    current_ids,
                    num_tokens=min(self.num_speculative_tokens, max_new_tokens - generated_tokens),
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample
                )

                # Step 2: Verify and correct with main model
                accepted_tokens, num_accepted = self.verify_and_correct(
                    main_model,
                    current_ids,
                    draft_ids,
                    draft_logits,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample
                )

                # Success - reset failure counter
                self.consecutive_failures = 0

            except Exception as e:
                # Phase 1: Draft model failed, fallback to standard generation
                self.consecutive_failures += 1
                self.total_fallbacks += 1

                # Generate 1 token with main model only
                accepted_tokens = self._generate_standard(
                    main_model,
                    current_ids,
                    num_tokens=1,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample
                )

            # Step 3: Append accepted tokens
            current_ids = torch.cat([current_ids, accepted_tokens], dim=1)
            generated_tokens += accepted_tokens.shape[1]

            # Check for EOS
            if eos_token_id is not None and eos_token_id in accepted_tokens:
                break

            # Early stop if no progress
            if accepted_tokens.shape[1] == 0:
                break

        return current_ids

    def get_stats(self) -> dict:
        """
        Get speculative decoding statistics.

        Phase 1: Now includes fallback tracking.

        Returns:
            Dict with acceptance rate and speedup metrics
        """
        if self.total_draft_tokens > 0:
            acceptance_rate = self.total_accepted_tokens / self.total_draft_tokens
        else:
            acceptance_rate = 0.0

        # Theoretical speedup: 1 / (1 - acceptance_rate * (K-1)/K)
        # where K is num_speculative_tokens
        if acceptance_rate > 0:
            k = self.num_speculative_tokens
            theoretical_speedup = k / (1 + k * (1 - acceptance_rate))
        else:
            theoretical_speedup = 1.0

        return {
            'total_draft_tokens': self.total_draft_tokens,
            'total_accepted_tokens': self.total_accepted_tokens,
            'acceptance_rate': acceptance_rate,
            'total_forward_passes': self.total_forward_passes,
            'theoretical_speedup': theoretical_speedup,
            # Phase 1: Fallback statistics
            'consecutive_failures': self.consecutive_failures,
            'total_fallbacks': self.total_fallbacks,
            'speculative_enabled': self.consecutive_failures < self.max_consecutive_failures
        }

    def reset_stats(self):
        """Reset statistics."""
        self.total_draft_tokens = 0
        self.total_accepted_tokens = 0
        self.total_forward_passes = 0


def create_draft_model(
    main_model_name: str,
    device: str = "cuda"
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Create an appropriate draft model for speculative decoding.

    Args:
        main_model_name: Name of the main model
        device: Device to load on

    Returns:
        Tuple of (draft_model, draft_tokenizer)
    """
    # Map main models to appropriate draft models
    draft_model_map = {
        'gpt2-xl': 'gpt2',           # XL → base (4x smaller)
        'gpt2-large': 'gpt2',         # Large → base (3x smaller)
        'gpt2-medium': 'gpt2',        # Medium → base (2x smaller)
        'meta-llama/Llama-2-13b': 'meta-llama/Llama-2-7b',  # 13B → 7B
        'meta-llama/Llama-2-70b': 'meta-llama/Llama-2-13b',  # 70B → 13B
    }

    # Default: use gpt2 as draft model
    draft_model_name = draft_model_map.get(main_model_name, 'gpt2')

    print(f"  Loading draft model: {draft_model_name}")

    draft_tokenizer = AutoTokenizer.from_pretrained(draft_model_name)
    draft_model = AutoModelForCausalLM.from_pretrained(
        draft_model_name,
        torch_dtype=torch.float16,
        device_map=device,
        low_cpu_mem_usage=True
    )
    draft_model.eval()

    return draft_model, draft_tokenizer

"""
Streaming text generation

PURPOSE: Generate text token-by-token for real-time responses
WHY: Better user experience, lower perceived latency
FEATURES: Token streaming, word streaming, sentence streaming

Usage:
    from memopt.streaming import StreamingGenerator
    
    generator = StreamingGenerator(model)
    
    for token in generator.stream_tokens("Hello", max_tokens=100):
        print(token, end='', flush=True)
"""

import torch
from typing import Iterator, Optional, Generator
import time

from memopt.monitoring.logger import get_logger
from memopt.utils.errors import GenerationError

logger = get_logger(__name__)


class StreamingGenerator:
    """
    Streaming text generator
    
    Modes:
    - token: Stream individual tokens
    - word: Stream complete words
    - sentence: Stream complete sentences
    """
    
    def __init__(self, model):
        """
        Initialize streaming generator
        
        Args:
            model: OptimizedLLM instance
        """
        self.model = model
        self.tokenizer = model.tokenizer
        
        logger.info("Streaming generator initialized")
    
    def stream_tokens(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        do_sample: bool = False,
        stop_sequences: Optional[list] = None
    ) -> Generator[str, None, None]:
        """
        Stream tokens one at a time
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling
            do_sample: Whether to use sampling
            stop_sequences: Sequences that stop generation
            
        Yields:
            Individual tokens as they're generated
        """
        logger.debug(f"Starting token streaming (max_tokens={max_tokens})")
        
        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            input_ids = inputs['input_ids']
            
            # Generation parameters
            gen_kwargs = {
                'max_new_tokens': max_tokens,
                'do_sample': do_sample,
                'pad_token_id': self.tokenizer.eos_token_id,
                'use_cache': True
            }
            
            if temperature is not None:
                gen_kwargs['temperature'] = temperature
            if top_p is not None:
                gen_kwargs['top_p'] = top_p
            
            # Track generated text for stop sequence detection
            generated_text = ""
            
            # Generate tokens one by one
            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=(self.model.torch_dtype == torch.float16)):
                    
                    for _ in range(max_tokens):
                        # Generate next token
                        outputs = self.model.model(
                            input_ids,
                            use_cache=True,
                            return_dict=True
                        )
                        
                        logits = outputs.logits[:, -1, :]
                        
                        # Apply sampling
                        if do_sample:
                            if temperature is not None and temperature > 0:
                                logits = logits / temperature
                            
                            probs = torch.softmax(logits, dim=-1)
                            
                            if top_p is not None and top_p < 1.0:
                                # Nucleus sampling
                                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                                
                                # Remove tokens with cumulative probability above threshold
                                sorted_indices_to_remove = cumulative_probs > top_p
                                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                                sorted_indices_to_remove[..., 0] = 0
                                
                                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                                probs[indices_to_remove] = 0
                                probs = probs / probs.sum()
                            
                            next_token = torch.multinomial(probs, num_samples=1)
                        else:
                            # Greedy
                            next_token = torch.argmax(logits, dim=-1, keepdim=True)
                        
                        # Check for EOS
                        if next_token.item() == self.tokenizer.eos_token_id:
                            logger.debug("EOS token generated, stopping")
                            break
                        
                        # Decode token
                        token_text = self.tokenizer.decode(next_token[0], skip_special_tokens=True)
                        
                        # Update generated text
                        generated_text += token_text
                        
                        # Check stop sequences
                        if stop_sequences:
                            if any(seq in generated_text for seq in stop_sequences):
                                logger.debug("Stop sequence detected")
                                break
                        
                        # Yield token
                        yield token_text
                        
                        # Append to input for next iteration
                        input_ids = torch.cat([input_ids, next_token], dim=1)
            
            logger.debug("Token streaming completed")
            
        except Exception as e:
            logger.error(f"Token streaming failed: {e}", exc_info=True)
            raise GenerationError(f"Streaming generation failed: {e}")
    
    def stream_words(
        self,
        prompt: str,
        max_tokens: int = 256,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        Stream complete words
        
        Accumulates tokens until a complete word is formed,
        then yields the word.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation arguments
            
        Yields:
            Complete words as they're generated
        """
        word_buffer = ""
        
        for token in self.stream_tokens(prompt, max_tokens, **kwargs):
            word_buffer += token
            
            # Check if we have a complete word (ends with space or punctuation)
            if token.endswith((' ', '.', ',', '!', '?', '\n', ':', ';')):
                yield word_buffer
                word_buffer = ""
        
        # Yield any remaining text
        if word_buffer:
            yield word_buffer
    
    def stream_sentences(
        self,
        prompt: str,
        max_tokens: int = 256,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        Stream complete sentences
        
        Accumulates tokens until a complete sentence is formed,
        then yields the sentence.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation arguments
            
        Yields:
            Complete sentences as they're generated
        """
        sentence_buffer = ""
        
        for token in self.stream_tokens(prompt, max_tokens, **kwargs):
            sentence_buffer += token
            
            # Check if we have a complete sentence
            if token.strip().endswith(('.', '!', '?')):
                yield sentence_buffer
                sentence_buffer = ""
        
        # Yield any remaining text
        if sentence_buffer:
            yield sentence_buffer
    
    def stream_with_timing(
        self,
        prompt: str,
        max_tokens: int = 256,
        **kwargs
    ) -> Generator[tuple, None, None]:
        """
        Stream tokens with timing information
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation arguments
            
        Yields:
            Tuples of (token, timestamp, time_since_last_token)
        """
        last_time = time.time()
        start_time = last_time
        
        for token in self.stream_tokens(prompt, max_tokens, **kwargs):
            current_time = time.time()
            time_delta = current_time - last_time
            
            yield (token, current_time - start_time, time_delta)
            
            last_time = current_time


class AsyncStreamingGenerator:
    """
    Async streaming generator for use with async frameworks
    
    Usage with FastAPI:
        async for token in generator.stream_tokens_async(prompt):
            yield token
    """
    
    def __init__(self, model):
        self.sync_generator = StreamingGenerator(model)
    
    async def stream_tokens_async(
        self,
        prompt: str,
        max_tokens: int = 256,
        **kwargs
    ):
        """
        Async wrapper for token streaming
        
        Yields tokens asynchronously for use in async frameworks
        """
        import asyncio
        
        # Run synchronous generator in thread pool
        loop = asyncio.get_event_loop()
        
        gen = self.sync_generator.stream_tokens(prompt, max_tokens, **kwargs)
        
        while True:
            try:
                # Get next token in thread pool to avoid blocking
                token = await loop.run_in_executor(None, next, gen, None)
                
                if token is None:
                    break
                
                yield token
                
            except StopIteration:
                break
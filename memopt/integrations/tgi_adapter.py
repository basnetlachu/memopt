"""
Text Generation Inference (TGI) adapter for MemOpt

PURPOSE: Compatible with HuggingFace TGI
WHY: Easy integration for TGI users
COMPATIBILITY: TGI-like interface

Usage:
    from memopt.integrations import MemOptTGI
    
    tgi = MemOptTGI(model="gpt2-xl")
    response = tgi.generate(
        inputs="Hello",
        max_new_tokens=100
    )
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import time

from memopt.core.model import OptimizedLLM
from memopt.monitoring.logger import get_logger

logger = get_logger(__name__)


@dataclass
class GenerateParameters:
    """TGI-compatible generation parameters"""
    
    do_sample: bool = False
    max_new_tokens: int = 20
    best_of: Optional[int] = None
    repetition_penalty: Optional[float] = None
    return_full_text: bool = False
    seed: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    truncate: Optional[int] = None
    typical_p: Optional[float] = None
    watermark: bool = False
    decoder_input_details: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            k: v for k, v in self.__dict__.items()
            if v is not None
        }


@dataclass
class GenerateResponse:
    """TGI-compatible response"""
    
    generated_text: str
    details: Optional[Dict] = None


@dataclass
class StreamResponse:
    """TGI-compatible stream response"""
    
    token: Dict[str, Any]
    generated_text: Optional[str] = None
    details: Optional[Dict] = None


class MemOptTGI:
    """
    TGI-compatible interface with MemOpt optimizations
    
    Provides HuggingFace Text Generation Inference compatible API
    """
    
    def __init__(
        self,
        model: str,
        revision: Optional[str] = None,
        sharded: bool = False,
        num_shard: Optional[int] = None,
        quantize: Optional[str] = None,
        dtype: Optional[str] = None,
        trust_remote_code: bool = False
    ):
        """
        Initialize MemOpt TGI adapter
        
        Args:
            model: Model name or path
            sharded: Whether to use multi-GPU (maps to MultiGPUManager)
            num_shard: Number of GPUs (if sharded)
            quantize: Quantization method (MemOpt has its own quantization)
        """
        logger.info("="*70)
        logger.info("Initializing MemOpt TGI Adapter")
        logger.info("="*70)
        logger.info(f"Model: {model}")
        
        self.model_name = model
        
        # Initialize MemOpt
        # For now, single GPU (multi-GPU support via MultiGPUManager can be added)
        self.backend = OptimizedLLM(
            model=model,
            optimization_level="high",
            enable_profiling=True
        )
        
        logger.info("✓ MemOpt TGI adapter initialized")
    
    def generate(
        self,
        inputs: str,
        parameters: Optional[GenerateParameters] = None,
        **kwargs
    ) -> GenerateResponse:
        """
        Generate text (TGI-compatible)
        
        Args:
            inputs: Input prompt
            parameters: Generation parameters
            **kwargs: Additional parameters
            
        Returns:
            GenerateResponse with generated text
        """
        # Merge parameters
        if parameters is None:
            parameters = GenerateParameters()
        
        # Override with kwargs
        for key, value in kwargs.items():
            if hasattr(parameters, key):
                setattr(parameters, key, value)
        
        logger.debug(f"TGI generate: {inputs[:50]}...")
        
        start_time = time.time()
        
        # Generate
        output = self.backend.generate(
            prompt=inputs,
            max_tokens=parameters.max_new_tokens,
            temperature=parameters.temperature,
            top_p=parameters.top_p,
            do_sample=parameters.do_sample
        )
        
        generation_time = time.time() - start_time
        
        # Format response
        if parameters.return_full_text:
            generated_text = output
        else:
            # Return only new tokens (remove prompt)
            generated_text = output[len(inputs):]
        
        # Build details if requested
        details = None
        if parameters.decoder_input_details:
            details = {
                'finish_reason': 'length',
                'generated_tokens': len(self.backend.tokenizer.encode(generated_text)),
                'seed': parameters.seed,
                'time': generation_time
            }
        
        return GenerateResponse(
            generated_text=generated_text,
            details=details
        )
    
    def generate_stream(
        self,
        inputs: str,
        parameters: Optional[GenerateParameters] = None,
        **kwargs
    ):
        """
        Generate text with streaming (TGI-compatible)
        
        Args:
            inputs: Input prompt
            parameters: Generation parameters
            **kwargs: Additional parameters
            
        Yields:
            StreamResponse objects
        """
        from memopt.streaming import StreamingGenerator
        
        # Merge parameters
        if parameters is None:
            parameters = GenerateParameters()
        
        for key, value in kwargs.items():
            if hasattr(parameters, key):
                setattr(parameters, key, value)
        
        logger.debug(f"TGI stream: {inputs[:50]}...")
        
        # Stream tokens
        streamer = StreamingGenerator(self.backend)
        
        generated_text = ""
        
        for token in streamer.stream_tokens(
            prompt=inputs,
            max_tokens=parameters.max_new_tokens,
            temperature=parameters.temperature,
            top_p=parameters.top_p,
            do_sample=parameters.do_sample,
            stop_sequences=parameters.stop_sequences
        ):
            generated_text += token
            
            # Format as TGI stream response
            token_info = {
                'id': 0,  # Simplified
                'text': token,
                'logprob': 0.0,  # MemOpt doesn't track logprobs yet
                'special': False
            }
            
            yield StreamResponse(
                token=token_info,
                generated_text=None
            )
        
        # Final response with full text
        final_text = generated_text if parameters.return_full_text else generated_text[len(inputs):]
        
        yield StreamResponse(
            token={},
            generated_text=final_text,
            details={'finish_reason': 'length'}
        )
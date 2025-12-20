"""
Pydantic models for API requests and responses

PURPOSE: Define request/response schemas
WHY: Automatic validation and documentation
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List
from enum import Enum


class OptimizationLevel(str, Enum):
    """Available optimization levels"""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    HIGH = "high"
    AGGRESSIVE = "aggressive"


class GenerateRequest(BaseModel):
    """Request model for text generation"""
    
    prompt: str = Field(
        ...,
        description="Input text prompt",
        min_length=1,
        max_length=100000
    )
    
    max_tokens: int = Field(
        default=256,
        description="Maximum tokens to generate",
        ge=1,
        le=8192
    )
    
    temperature: Optional[float] = Field(
        default=None,
        description="Sampling temperature (0.0-2.0)",
        ge=0.0,
        le=2.0
    )
    
    top_p: Optional[float] = Field(
        default=None,
        description="Nucleus sampling parameter (0.0-1.0)",
        ge=0.0,
        le=1.0
    )
    
    do_sample: bool = Field(
        default=False,
        description="Whether to use sampling"
    )
    
    stream: bool = Field(
        default=False,
        description="Whether to stream the response"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "prompt": "The future of AI is",
                "max_tokens": 100,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True,
                "stream": False
            }
        }


class GenerateResponse(BaseModel):
    """Response model for text generation"""
    
    output: str = Field(..., description="Generated text")
    
    input_tokens: int = Field(..., description="Number of input tokens")
    
    output_tokens: int = Field(..., description="Number of generated tokens")
    
    generation_time: float = Field(..., description="Generation time in seconds")
    
    tokens_per_second: float = Field(..., description="Throughput in tokens/second")
    
    model: str = Field(..., description="Model used for generation")
    
    optimization_level: str = Field(..., description="Optimization level used")
    
    class Config:
        schema_extra = {
            "example": {
                "output": "The future of AI is bright and full of possibilities...",
                "input_tokens": 5,
                "output_tokens": 100,
                "generation_time": 0.5,
                "tokens_per_second": 200.0,
                "model": "gpt2-xl",
                "optimization_level": "high"
            }
        }


class BatchGenerateRequest(BaseModel):
    """Request model for batch generation"""
    
    prompts: List[str] = Field(
        ...,
        description="List of input prompts",
        min_items=1,
        max_items=128
    )
    
    max_tokens: int = Field(default=256, ge=1, le=8192)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    do_sample: bool = Field(default=False)


class BatchGenerateResponse(BaseModel):
    """Response model for batch generation"""
    
    outputs: List[str] = Field(..., description="Generated texts")
    
    total_time: float = Field(..., description="Total batch processing time")
    
    average_tokens_per_second: float = Field(
        ..., 
        description="Average throughput across batch"
    )
    
    batch_size: int = Field(..., description="Number of prompts in batch")


class ModelInfo(BaseModel):
    """Information about loaded model"""
    
    model_name: str
    optimization_level: str
    num_layers: int
    num_heads: int
    hidden_size: int
    device: str
    memory_allocated_gb: float
    kv_cache_enabled: bool
    quantization_enabled: bool
    flash_attention_enabled: bool


class HealthResponse(BaseModel):
    """Health check response"""
    
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    gpu_available: bool = Field(..., description="Whether GPU is available")
    memory_free_gb: float = Field(..., description="Free GPU memory in GB")
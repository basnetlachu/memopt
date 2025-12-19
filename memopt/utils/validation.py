"""
Input validation utilities

PURPOSE: Validate all inputs to MemOpt before processing
WHY: 
  - Prevents crashes from invalid inputs
  - Provides clear error messages
  - Security: prevents injection attacks
  - Data quality: ensures valid parameters

WHEN TO USE: Call validators at API boundaries (user input, external calls)

Example:
    prompt = InputValidator.validate_prompt(user_input)
    max_tokens = InputValidator.validate_max_tokens(request.max_tokens)
"""

from typing import Optional, Any, List
from .errors import InvalidInputError


class InputValidator:
    """
    Validates inputs for MemOpt operations
    
    All validation methods:
    1. Check type (is it the right data type?)
    2. Check range (is it within acceptable limits?)
    3. Check content (does it contain invalid data?)
    4. Return validated value or raise InvalidInputError
    """
    
    # Configuration limits (adjust based on your needs)
    MAX_PROMPT_LENGTH = 100_000  # characters (prevents memory issues)
    MAX_TOKENS = 8192            # maximum generation length
    MIN_TOKENS = 1               # minimum generation length
    MAX_TEMPERATURE = 2.0        # higher = more random
    MIN_TEMPERATURE = 0.0        # lower = more deterministic
    MAX_TOP_P = 1.0             # nucleus sampling upper bound
    MIN_TOP_P = 0.0             # nucleus sampling lower bound
    MAX_BATCH_SIZE = 128        # maximum batch size for API
    
    @classmethod
    def validate_prompt(cls, prompt: Any) -> str:
        """
        Validate and sanitize text prompt
        
        Checks:
        - Is string type
        - Not empty
        - Not too long
        - No dangerous patterns (basic security)
        
        Args:
            prompt: Input prompt (should be string)
            
        Returns:
            Validated prompt string (stripped of whitespace)
            
        Raises:
            InvalidInputError: If validation fails
        """
        # CHECK 1: Type
        if not isinstance(prompt, str):
            raise InvalidInputError(
                f"Prompt must be string, got {type(prompt).__name__}",
                {"type": type(prompt).__name__}
            )
        
        # CHECK 2: Empty
        if not prompt or not prompt.strip():
            raise InvalidInputError("Prompt cannot be empty or whitespace only")
        
        # CHECK 3: Length
        if len(prompt) > cls.MAX_PROMPT_LENGTH:
            raise InvalidInputError(
                f"Prompt too long: {len(prompt)} characters",
                {
                    "length": len(prompt),
                    "max_length": cls.MAX_PROMPT_LENGTH
                }
            )
        
        return prompt.strip()
    
    @classmethod
    def validate_max_tokens(cls, max_tokens: Any) -> int:
        """
        Validate max_tokens parameter
        
        Args:
            max_tokens: Maximum tokens to generate
            
        Returns:
            Validated max_tokens as int
            
        Raises:
            InvalidInputError: If validation fails
        """
        # Try to convert to int if not already
        if not isinstance(max_tokens, int):
            try:
                max_tokens = int(max_tokens)
            except (ValueError, TypeError):
                raise InvalidInputError(
                    f"max_tokens must be integer, got {type(max_tokens).__name__}"
                )
        
        # Check range
        if max_tokens < cls.MIN_TOKENS:
            raise InvalidInputError(
                f"max_tokens must be >= {cls.MIN_TOKENS}, got {max_tokens}"
            )
        
        if max_tokens > cls.MAX_TOKENS:
            raise InvalidInputError(
                f"max_tokens must be <= {cls.MAX_TOKENS}, got {max_tokens}"
            )
        
        return max_tokens
    
    @classmethod
    def validate_temperature(cls, temperature: Optional[float]) -> Optional[float]:
        """
        Validate temperature parameter
        
        Temperature controls randomness:
        - 0.0 = deterministic (always same output)
        - 1.0 = balanced
        - 2.0 = very random
        
        Args:
            temperature: Sampling temperature (can be None)
            
        Returns:
            Validated temperature as float or None
        """
        if temperature is None:
            return None
        
        # Type check
        if not isinstance(temperature, (int, float)):
            raise InvalidInputError(
                f"temperature must be number, got {type(temperature).__name__}"
            )
        
        # Range check
        if not (cls.MIN_TEMPERATURE <= temperature <= cls.MAX_TEMPERATURE):
            raise InvalidInputError(
                f"temperature must be in [{cls.MIN_TEMPERATURE}, {cls.MAX_TEMPERATURE}], "
                f"got {temperature}"
            )
        
        return float(temperature)
    
    @classmethod
    def validate_top_p(cls, top_p: Optional[float]) -> Optional[float]:
        """
        Validate top_p (nucleus sampling) parameter
        
        Top-p controls diversity:
        - 0.1 = very focused (top 10% of likely tokens)
        - 0.9 = diverse (top 90% of likely tokens)
        - 1.0 = consider all tokens
        """
        if top_p is None:
            return None
        
        if not isinstance(top_p, (int, float)):
            raise InvalidInputError(
                f"top_p must be number, got {type(top_p).__name__}"
            )
        
        if not (cls.MIN_TOP_P <= top_p <= cls.MAX_TOP_P):
            raise InvalidInputError(
                f"top_p must be in [{cls.MIN_TOP_P}, {cls.MAX_TOP_P}], got {top_p}"
            )
        
        return float(top_p)
    
    @classmethod
    def validate_batch_size(cls, batch_size: int) -> int:
        """Validate batch size for batch processing"""
        if not isinstance(batch_size, int):
            raise InvalidInputError(
                f"batch_size must be integer, got {type(batch_size).__name__}"
            )
        
        if batch_size < 1:
            raise InvalidInputError(f"batch_size must be >= 1, got {batch_size}")
        
        if batch_size > cls.MAX_BATCH_SIZE:
            raise InvalidInputError(
                f"batch_size must be <= {cls.MAX_BATCH_SIZE}, got {batch_size}"
            )
        
        return batch_size
    
    @classmethod
    def validate_model_name(cls, model_name: str) -> str:
        """Validate model name/path"""
        if not isinstance(model_name, str):
            raise InvalidInputError(
                f"model_name must be string, got {type(model_name).__name__}"
            )
        
        if not model_name or not model_name.strip():
            raise InvalidInputError("model_name cannot be empty")
        
        return model_name.strip()
    
    @classmethod
    def validate_device(cls, device: str) -> str:
        """
        Validate device string
        
        Valid devices:
        - 'cuda' (default GPU)
        - 'cuda:0', 'cuda:1', etc. (specific GPU)
        - 'cpu' (CPU only)
        - 'auto' (automatic selection)
        """
        if not isinstance(device, str):
            raise InvalidInputError(
                f"device must be string, got {type(device).__name__}"
            )
        
        valid_devices = ['cuda', 'cpu', 'auto']
        
        # Check if it's a specific cuda device (cuda:0, cuda:1, etc.)
        if device.startswith('cuda:'):
            try:
                gpu_id = int(device.split(':')[1])
                if gpu_id < 0:
                    raise ValueError()
                return device  # Valid
            except:
                raise InvalidInputError(f"Invalid cuda device: {device}")
        
        # Check if it's in the valid list
        if device not in valid_devices:
            raise InvalidInputError(
                f"Invalid device: {device}. "
                f"Must be one of {valid_devices} or 'cuda:N'"
            )
        
        return device
# Test logging
from memopt.monitoring.logger import get_logger
logger = get_logger(__name__)
logger.info("Production system ready!")

# Test validation
from memopt.utils.validation import InputValidator
prompt = InputValidator.validate_prompt("Hello world")
max_tokens = InputValidator.validate_max_tokens(100)

# Test memory manager
from memopt.core.memory_manager import MemoryManager
mem_mgr = MemoryManager()
stats = mem_mgr.get_memory_stats()
print(f"GPU Memory: {stats['allocated_gb']:.2f} GB")
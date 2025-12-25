from memopt import OptimizedLLM

# Before (Stage 0)
model = OptimizedLLM("gpt2", optimization_level="conservative")
model = OptimizedLLM("gpt2-xl", optimization_level="high")
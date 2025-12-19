from memopt import OptimizedLLM

print("Loading model...")
model = OptimizedLLM("gpt2", optimization_level="high")

print("Generating text...")
response = model.generate("Hello world", max_tokens=50)

print("\nResult:")
print(response)
"""
Quick test to verify v1.1 & v1.2 features are working
"""

print("="*70)
print("Testing MemOpt v1.1 & v1.2 Features")
print("="*70)

# Test 1: API Module
print("\n[1/6] Testing API module...")
try:
    from memopt.api import create_app
    from memopt.api.models import GenerateRequest, GenerateResponse
    print("    ✓ API imports successful")
except Exception as e:
    print(f"    ✗ API import failed: {e}")

# Test 2: Multi-GPU Module
print("\n[2/6] Testing Multi-GPU module...")
try:
    from memopt.distributed import MultiGPUManager, LoadBalancer, BatchProcessor
    print("    ✓ Multi-GPU imports successful")
except Exception as e:
    print(f"    ✗ Multi-GPU import failed: {e}")

# Test 3: Streaming Module
print("\n[3/6] Testing Streaming module...")
try:
    from memopt.streaming import StreamingGenerator
    print("    ✓ Streaming imports successful")
except Exception as e:
    print(f"    ✗ Streaming import failed: {e}")

# Test 4: Integrations Module
print("\n[4/6] Testing Integrations module...")
try:
    from memopt.integrations import MemOptVLLM, MemOptTGI
    print("    ✓ Integrations imports successful")
except Exception as e:
    print(f"    ✗ Integrations import failed: {e}")

# Test 5: Quantization Module
print("\n[5/6] Testing Quantization module...")
try:
    from memopt.quantization import W8A8Quantizer, quantize_model
    print("    ✓ Quantization imports successful")
except Exception as e:
    print(f"    ✗ Quantization import failed: {e}")

# Test 6: Functional Test - API Server
print("\n[6/6] Testing API Server creation...")
try:
    from memopt.api import create_app
    app = create_app(model_name="gpt2", optimization_level="high")
    print("    ✓ API server created successfully")
    print(f"    ✓ Server type: {type(app).__name__}")
except Exception as e:
    print(f"    ✗ API server creation failed: {e}")

print("\n" + "="*70)
print("✓ ALL TESTS COMPLETE!")
print("="*70)
print("\nIf all tests show ✓, your v1.1 & v1.2 features are working!")
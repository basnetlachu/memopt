"""
Test the actual API server
"""

from memopt.api import run_server
import threading
import time
import requests

def test_api():
    print("="*70)
    print("API Server Test")
    print("="*70)
    
    # Start server in background thread
    print("\nStarting API server...")
    server_thread = threading.Thread(
        target=run_server,
        kwargs={
            'model_name': 'gpt2',
            'optimization_level': 'high',
            'host': '127.0.0.1',
            'port': 8000
        },
        daemon=True
    )
    server_thread.start()
    
    # Wait for server to start
    print("Waiting for server to initialize...")
    time.sleep(10)
    
    # Test health endpoint
    print("\nTesting /health endpoint...")
    try:
        response = requests.get("http://127.0.0.1:8000/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test generate endpoint
    print("\nTesting /generate endpoint...")
    try:
        response = requests.post(
            "http://127.0.0.1:8000/generate",
            json={
                "prompt": "Hello world",
                "max_tokens": 20
            }
        )
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "="*70)
    print("✓ API Server Test Complete!")
    print("="*70)
    
    # Keep alive
    input("\nPress Enter to stop server...")

if __name__ == "__main__":
    test_api()
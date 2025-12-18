from setuptools import setup, find_packages

setup(
    name="memopt",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.35.0",
        "accelerate>=0.25.0",
        "numpy>=1.24.0",
        "tqdm>=4.66.0",
    ],
    python_requires=">=3.8",
    author="MemOpt Team",
    description="GPU memory optimization for LLM inference",
)

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="memopt",
    version="0.1.0",
    author="MemOpt Team",
    author_email="info@memopt.ai",
    description="GPU Memory Bandwidth Optimization Engine for LLM Inference",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/memopt/memopt",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.35.0",
        "accelerate>=0.25.0",
        "numpy>=1.24.0",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
        "flash": [
            "flash-attn>=2.3.0",
        ],
        "triton": [
            "triton>=2.1.0",
        ],
    },
)

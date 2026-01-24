from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="memopt",
    version="0.4.0",
    author="MemOpt Team",
    author_email="contact@memopt.dev",
    description="GPU Memory Bandwidth Profiling & Optimization Platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/memopt",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Benchmark",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "visualization": [
            "matplotlib>=3.7.0",
            "plotly>=5.14.0",
        ],
        "full": [
            "transformers>=4.30.0",
            "accelerate>=0.20.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "memopt-profile=memopt.cli:main",
        ],
    },
    keywords=[
        "gpu",
        "memory",
        "bandwidth",
        "profiling",
        "optimization",
        "llm",
        "inference",
        "hbm",
        "performance",
    ],
)

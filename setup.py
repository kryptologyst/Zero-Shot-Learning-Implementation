"""Setup script for zero-shot learning package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="zero-shot-learning",
    version="1.0.0",
    author="kryptologyst",
    author_email="kryptologyst@example.com",
    description="Zero-shot Learning Implementation with CLIP and Advanced Methods",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kryptologyst/zero-shot-learning",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.30.0",
        "clip-by-openai>=1.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.15.0",
        "tqdm>=4.65.0",
        "pillow>=10.0.0",
        "opencv-python>=4.8.0",
        "omegaconf>=2.3.0",
        "hydra-core>=1.3.0",
        "streamlit>=1.25.0",
        "gradio>=3.40.0",
        "wandb>=0.15.0",
        "tensorboard>=2.13.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "pydantic>=2.0.0",
        "httpx>=0.24.0",
    ],
    extras_require={
        "dev": [
            "black>=23.0.0",
            "ruff>=0.0.280",
            "mypy>=1.5.0",
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pre-commit>=3.3.0",
        ],
        "advanced": [
            "open_clip_torch>=2.20.0",
            "sentence-transformers>=2.2.0",
            "faiss-cpu>=1.7.4",
            "hnswlib>=0.7.0",
            "optuna>=3.3.0",
            "ray[tune]>=2.6.0",
        ],
    },
)

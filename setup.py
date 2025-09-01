from setuptools import setup, find_packages

setup(
    name="radiology-report-generator",
    version="1.0.0",
    description="AI-Assisted Radiology Report Generation System",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "monai>=1.3.0",
        "langchain>=0.1.0",
        "openai>=1.0.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.22.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "Pillow>=9.5.0",
        "pyyaml>=6.0",
        "scikit-learn>=1.3.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
"""
PRIMVS-Pipeline: Variable Star Catalogue Construction Pipeline

A clean, portable pipeline for constructing the PRIMVS catalogue from VIRAC data.
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
def read_requirements(filename):
    """Read requirements from file."""
    with open(filename) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="primvs-pipeline",
    version="2.0.0",
    author="Niall Miller",
    author_email="your.email@example.com",
    description="PRIMVS Variable Star Catalogue Construction Pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nialljmiller/PRIMVS-Pipeline",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=22.0",
            "flake8>=5.0",
            "mypy>=0.990",
        ],
    },
    entry_points={
        "console_scripts": [
            "primvs-pipeline=primvs_pipeline.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "primvs_pipeline": ["config/*.yaml"],
    },
)


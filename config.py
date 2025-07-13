from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="modular-transformer",
    version="0.1.0",
    author="Avadhesh",
    author_email="avadheshkumarajay@gmail.com",
    description="A modular implementation of the Transformer architecture with interchangeable positional encodings",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/modular-transformer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "plotly>=5.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "modular-transformer-train=train:main",
            "modular-transformer-eval=evaluate:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/your-username/modular-transformer/issues",
        "Source": "https://github.com/your-username/modular-transformer",
        "Documentation": "https://github.com/your-username/modular-transformer/docs",
    },
)
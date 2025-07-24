# Mathematical Reasoning with Advanced Positional Encoding
# Optimized for HPC deployment with NVIDIA Enroot

FROM nvcr.io/nvidia/pytorch:23.10-py3

LABEL maintainer="Research Team, IIT Delhi"
LABEL description="Mathematical Reasoning with Advanced Positional Encoding"
LABEL version="1.0"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    htop \
    tmux \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install additional HPC-specific packages
RUN pip install --no-cache-dir \
    mpi4py \
    nvidia-ml-py3 \
    psutil

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p /workspace/data_cache /workspace/model_cache /workspace/results /workspace/logs

# Set permissions
RUN chmod +x scripts/*.py scripts/*.sh

# Set environment variables for HuggingFace
ENV HF_HOME=/workspace/model_cache
ENV TRANSFORMERS_CACHE=/workspace/model_cache
ENV HF_DATASETS_CACHE=/workspace/data_cache

# Create non-root user for security
RUN useradd -m -s /bin/bash mathuser && \
    chown -R mathuser:mathuser /workspace

# Switch to non-root user
USER mathuser

# Set default command
CMD ["/bin/bash"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import torch; print('PyTorch available:', torch.cuda.is_available())" || exit 1

# Labels for better organization
LABEL org.opencontainers.image.title="Mathematical Reasoning PE Framework"
LABEL org.opencontainers.image.description="Framework for comparing positional encoding methods in mathematical reasoning"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.authors="Research Team, IIT Delhi"
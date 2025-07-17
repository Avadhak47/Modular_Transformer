# NVIDIA Enroot Container for Mathematical Reasoning Transformer
# Optimized for IITD HPC Multi-Node Deployment

FROM nvcr.io/nvidia/pytorch:24.01-py3

# Container metadata
LABEL maintainer="Mathematical Reasoning Research Team"
LABEL description="Multi-node mathematical reasoning transformer with SOTA models"
LABEL version="1.0"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH="/workspace:$PYTHONPATH"
ENV CUDA_VISIBLE_DEVICES="0"
ENV NCCL_DEBUG=INFO
ENV NCCL_IB_DISABLE=0
ENV NCCL_SOCKET_IFNAME=ib0
ENV OMP_NUM_THREADS=8

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    cmake \
    libssl-dev \
    libffi-dev \
    libblas-dev \
    liblapack-dev \
    gfortran \
    wget \
    curl \
    htop \
    vim \
    screen \
    tmux \
    openssh-client \
    infiniband-diags \
    ibutils \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages for HPC
COPY requirements_hpc.txt /tmp/requirements_hpc.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r /tmp/requirements_hpc.txt

# Install SOTA model dependencies
RUN pip install --no-cache-dir \
    transformers[torch]==4.36.0 \
    accelerate==0.25.0 \
    deepspeed==0.12.0 \
    wandb==0.16.0 \
    flash-attn==2.4.2 \
    peft==0.7.0 \
    bitsandbytes==0.41.3 \
    xformers==0.0.23 \
    datasets==2.14.0 \
    evaluate==0.4.0 \
    scikit-learn==1.3.0 \
    sympy==1.12 \
    streamlit==1.28.0 \
    flask==3.0.0 \
    requests==2.31.0 \
    matplotlib==3.7.2 \
    seaborn==0.12.2 \
    plotly==5.17.0 \
    tensorboard==2.15.0

# Install mathematical reasoning specific packages
RUN pip install --no-cache-dir \
    python-Levenshtein \
    rouge-score \
    bleurt \
    sacrebleu \
    bert-score \
    mauve-text

# Install HPC communication libraries
RUN pip install --no-cache-dir \
    mpi4py \
    horovod[pytorch]

# Copy source code
WORKDIR /workspace
COPY . /workspace/

# Create necessary directories
RUN mkdir -p /workspace/logs \
    /workspace/checkpoints \
    /workspace/results \
    /workspace/models \
    /workspace/data \
    /workspace/configs

# Set permissions
RUN chmod +x /workspace/scripts/*.sh 2>/dev/null || true

# Install the package in development mode
RUN pip install -e .

# Setup Hugging Face cache directory
RUN mkdir -p /root/.cache/huggingface
ENV HF_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface

# Configure git (for model downloads)
RUN git config --global user.email "research@iitd.ac.in"
RUN git config --global user.name "HPC Research"

# Set default working directory
WORKDIR /workspace

# Default entrypoint
ENTRYPOINT ["python"]
CMD ["-m", "training.sota_mathematical_reasoning_trainer"]
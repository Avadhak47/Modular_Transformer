FROM python:3.10-slim

# Install system dependencies including build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    python3-dev \
    git \
    wget \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create directories for HPC deployment
RUN mkdir -p /workspace/checkpoints /workspace/results /workspace/logs /workspace/data

# Set environment variables for HPC
ENV PYTHONPATH=/workspace:$PYTHONPATH
ENV CUDA_VISIBLE_DEVICES=0
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

# Default command
CMD ["python", "main.py"]

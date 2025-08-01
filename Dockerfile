# Use an official PyTorch runtime as a parent image
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Set the working directory
WORKDIR /workspace

# Install necessary system packages
RUN apt-get update && apt-get install -y \
    git \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Clone the mergekit repository
RUN git clone https://github.com/arcee-ai/mergekit.git
WORKDIR /workspace/mergekit

# Install mergekit with evolve and vllm features
RUN pip install -e .[evolve,vllm]

# Fix flash attention if necessary
RUN pip uninstall -y flash-attn && \
    pip cache purge && \
    pip install flash-attn

# Create a directory for storage
RUN mkdir -p /workspace/mergekit-evolve-storage

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

# Install mergekit with evolve and vllm features in non-editable mode
RUN pip install .[evolve,vllm]

# Install packaging before flash-attn
RUN pip install packaging

# Fix flash attention if necessary
RUN pip uninstall -y flash-attn && \
    pip cache purge && \
    pip install flash-attn

# Install additional Python packages for the GUI
RUN pip install gradio huggingface-hub

# Create a directory for storage
RUN mkdir -p /workspace/mergekit-evolve-storage

# Copy the Gradio application script into the container
COPY app.py /workspace/app.py

# Expose the port for the Gradio app
EXPOSE 7860

# Command to run the Gradio app
CMD ["python", "/workspace/app.py"]

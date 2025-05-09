# Dockerfile for FramePack on RunPod

# 1. Base Image
# Using Python 3.11 and CUDA 12.8.1 compatibility
# This image comes with PyTorch 2.8.0, Python 3.11, and CUDA 12.8.1 pre-installed.
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

LABEL maintainer="fei.wong"
LABEL project="FramePack"
LABEL description="Dockerfile for deploying FramePack video generation on RunPod with Python 3.11 and CUDA 12.8.1."

# 2. Environment Variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
# For Gradio, to listen on all interfaces and use a standard port
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT="7860"
# Set Hugging Face cache directories to be under /app or a persistent volume if desired
ENV HF_HOME="/app/huggingface_cache/huggingface"
ENV TRANSFORMERS_CACHE="/app/huggingface_cache/transformers"
ENV DIFFUSERS_CACHE="/app/huggingface_cache/diffusers"
# Add project to PYTHONPATH
ENV PYTHONPATH="${PYTHONPATH}:/app"

# 3. System Dependencies
# The base image should have most essentials.
# Adding git (if you need to clone anything at runtime) and ffmpeg (for 'av' package).
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 4. Set up Working Directory
WORKDIR /app

# 5. Install Python Dependencies from requirements.txt
# PyTorch 2.8.0 is expected to be pre-installed in the base image.
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

# Consider SageAttention/Triton installation here if critical.
# This would require Python 3.11 compatible wheels for Triton and SageAttention for Linux with CUDA 12.8.1.
# Ensure to find wheels compatible with PyTorch 2.8.0 as well.
# Example (highly experimental, check SageAttention/Triton docs):
# RUN pip3 install triton -f &lt;URL_to_Linux_Triton_wheel_for_Py311_and_CUDA_12.8.1&gt;
# RUN pip3 install sageattention -f &lt;URL_to_SageAttention_wheel&gt;

# 6. Copy Application Code
# Ensure .dockerignore is set up to exclude venv, .git, __pycache__, local_models, outputs, logs etc.
COPY . .

# 7. Create directories for models and outputs
# These directories are intended to be mount points for persistent storage on RunPod.
# Models should ideally be downloaded to /workspace/local_models or /runpod-volume/local_models on RunPod.
RUN mkdir -p /app/local_models && \
    mkdir -p /app/outputs && \
    mkdir -p /app/huggingface_cache/huggingface && \
    mkdir -p /app/huggingface_cache/transformers && \
    mkdir -p /app/huggingface_cache/diffusers && \
    chown -R 1000:1000 /app # RunPod often runs containers as user 1000

# 8. Expose Port
# Gradio default port is 7860
EXPOSE 7860

# 9. Entrypoint/Command
# Run the Gradio application using arguments recognized by app.py
CMD ["python3", "app.py", "--server", "0.0.0.0", "--port", "7860"]
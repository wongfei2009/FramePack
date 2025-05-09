# Dockerfile for FramePack on RunPod

# 1. Base Image
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

LABEL maintainer="fei.wong"
LABEL project="FramePack"
LABEL description="Dockerfile for deploying FramePack video generation on RunPod with Python 3.11 and CUDA 12.8.1."

# 2. Environment Variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT="7860"
ENV HF_HOME="/app/huggingface_cache/huggingface"
ENV TRANSFORMERS_CACHE="/app/huggingface_cache/transformers"
ENV DIFFUSERS_CACHE="/app/huggingface_cache/diffusers"
ENV PYTHONPATH="${PYTHONPATH}:/app"

# 3. System Dependencies
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 4. Set up Working Directory
WORKDIR /app

# 5. Install Python Dependencies from requirements.txt and Triton
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

RUN pip3 install --no-cache-dir triton

# Attempt to install SageAttention via pip, ONLY if a pre-built binary wheel is available.
RUN echo "Attempting to install SageAttention via pip (binary wheel only)..." && \
    pip3 install --no-cache-dir --only-binary=:all: sageattention && \
    echo "pip install sageattention --only-binary=:all: command finished successfully." && \
    echo "Installed sageattention version details:" && \
    pip3 show sageattention || echo "SageAttention not found or pip show failed."

# 6. Copy Application Code
COPY . .

# 7. Create directories for models and outputs
RUN mkdir -p /app/local_models && \
    mkdir -p /app/outputs && \
    mkdir -p /app/huggingface_cache/huggingface && \
    mkdir -p /app/huggingface_cache/transformers && \
    mkdir -p /app/huggingface_cache/diffusers && \
    chown -R 1000:1000 /app

# 8. Expose Port
EXPOSE 7860

# 9. Entrypoint/Command
CMD ["python3", "app.py", "--server", "0.0.0.0", "--port", "7860"]
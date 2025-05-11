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
ENV RUNPOD=1
# Update all directories to use a single organized directory on the network drive
ENV HF_HOME="/workspace/framepack/huggingface_cache/huggingface"
ENV TRANSFORMERS_CACHE="/workspace/framepack/huggingface_cache/transformers"
ENV DIFFUSERS_CACHE="/workspace/framepack/huggingface_cache/diffusers"
ENV LOCAL_MODELS_DIR="/workspace/framepack/local_models"
ENV OUTPUTS_DIR="/workspace/framepack/outputs"
ENV PYTHONPATH="${PYTHONPATH}:/app"

# 3. System Dependencies - Added openssh-server
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    openssh-server && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 4. SSH Configuration
# Create SSH directory and set permissions
RUN mkdir -p /root/.ssh && \
    chmod 700 /root/.ssh

# 5. Set up Working Directory
WORKDIR /app

# 6. Install Python Dependencies from requirements.txt and Triton
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

# 7. Copy Application Code
COPY . .

# 8. Make scripts executable
RUN chmod +x /app/setup_workspace.sh && \
    chmod +x /app/setup_ssh.sh && \
    chmod +x /app/startup.sh

# 9. Expose Ports - added SSH port 22
EXPOSE 7860 22

# 10. Entrypoint/Command - Use the combined startup script
CMD ["/app/startup.sh"]
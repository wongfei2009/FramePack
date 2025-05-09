#!/bin/bash
# Enhanced setup script with better logging

echo "====== FramePack Workspace Setup ======"
echo "Starting workspace initialization..."

# Create main framepack directory on workspace
mkdir -p /workspace/framepack
echo "Created main directory: /workspace/framepack"

# Create necessary subdirectories
mkdir -p /workspace/framepack/local_models
mkdir -p /workspace/framepack/outputs
mkdir -p /workspace/framepack/huggingface_cache/huggingface
mkdir -p /workspace/framepack/huggingface_cache/transformers
mkdir -p /workspace/framepack/huggingface_cache/diffusers
echo "Created all subdirectories"

# Set permissions - make output directory world-writable
chown -R 1000:1000 /workspace/framepack
chmod -R 777 /workspace/framepack/outputs
echo "Set directory permissions"

# Print directory structure for verification
echo "FramePack directories created on network drive:"
ls -la /workspace/framepack

# Print environment variables to verify they're set correctly
echo "====== Environment Variables ======"
echo "HF_HOME: $HF_HOME"
echo "TRANSFORMERS_CACHE: $TRANSFORMERS_CACHE"
echo "DIFFUSERS_CACHE: $DIFFUSERS_CACHE"
echo "LOCAL_MODELS_DIR: $LOCAL_MODELS_DIR"
echo "OUTPUTS_DIR: $OUTPUTS_DIR"
echo "===================================="

# Check GPU availability before starting
echo "====== GPU Information ======"
nvidia-smi
echo "============================"

# Start the application with more memory for CUDA
echo "Starting FramePack application..."
# Change to the app directory first to ensure we're in the right place
cd /app
exec python3 /app/app.py --server 0.0.0.0 --port 7860
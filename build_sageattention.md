# SageAttention Build and Upload Instructions

This repository contains instructions for building [SageAttention](https://github.com/thu-ml/SageAttention) from source and uploading the compiled wheel to Hugging Face for easy installation in Docker containers.

## Why Build from Source?

- Avoid slow compilation during Docker container startup
- Ensure consistent builds across all environments
- Pre-compile with specific CUDA and Python versions

## Prerequisites

- Docker installed on your system with GPU support
- NVIDIA Container Toolkit (nvidia-docker2) installed
- Hugging Face account
- Hugging Face CLI installed (`pip install huggingface_hub`)

## Build Process

### 1. Create the Builder Dockerfile

Create a file named `Dockerfile.build` with the following content:

```dockerfile
# Use RunPod PyTorch image as base
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
    git \
    build-essential \
    ninja-build \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /build

# Install Python build dependencies
RUN pip3 install --no-cache-dir \
    wheel \
    setuptools \
    build \
    ninja \
    huggingface_hub

# Clone the SageAttention repository
RUN git clone https://github.com/thu-ml/SageAttention.git

# Set the entrypoint to bash to keep container running
ENTRYPOINT ["/bin/bash"]
```

### 2. Build the Docker Image

```bash
docker build -t sageattention-builder -f Dockerfile.build .
```

### 3. Run the Container with GPU Access

```bash
# Create a directory to store the built wheel
mkdir -p sage-wheels

# Run the container with GPU access and mount the directory
docker run -it --gpus all -v $(pwd)/sage-wheels:/out sageattention-builder
```

### 4. Build the Wheel Inside the Container

Once inside the container, you'll get a bash prompt. Run these commands to build the wheel:

```bash
# Verify GPU is accessible
nvidia-smi
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Navigate to SageAttention directory
cd /build/SageAttention

# Build the wheel
python setup.py bdist_wheel

# Copy the built wheel to the mounted output directory
cp dist/*.whl /out/

# Exit the container
exit
```

The wheel file will now be available in the `sage-wheels` directory on your host machine.

This will build the wheel file with GPU access and copy it to the `sage-wheels` directory on your host machine.

## Upload to Hugging Face

### 1. Login to Hugging Face

```bash
huggingface-cli login
```

### 2. Create a New Repository

```bash
huggingface-cli repo create sageattention-compiled
```

### 3. Upload the Wheel File

```bash
huggingface-cli upload-large-folder --repo-type=model wongfei2009/sageattention-compiled sage-wheels
```

## Update Your Dockerfile

Replace the pip install line in your original Dockerfile:

```dockerfile
# Original line
RUN pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install sageattention==1.0.6

# New line
RUN pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install https://huggingface.co/wongfei2009/sageattention-compiled/resolve/main/sageattention-2.1.1-cp311-cp311-linux_x86_64.whl
```

Replace `YOUR_USERNAME` with your Hugging Face username and `VERSION` with the actual version number from the built wheel file.

## Additional Notes

### Getting CUDA Information

Inside the container, you can verify CUDA is available with:

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Check CUDA version
python -c "import torch; print(torch.version.cuda)"

# Check device information
python -c "import torch; print(torch.cuda.get_device_properties(0))"
```

### Version Naming Strategy

You may want to include the CUDA version and Python version in your Hugging Face repository organization:

```
/wheels/cuda12.8.1-py3.11/sageattention-VERSION-cp311-cp311-linux_x86_64.whl
```

This structure helps when managing multiple versions.

### Alternative Build Method for Multiple CUDA Architectures

If you need to build for multiple CUDA architectures, you can set the TORCH_CUDA_ARCH_LIST environment variable:

```bash
# Inside the container
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6"
cd /build/SageAttention
python setup.py bdist_wheel
```

This would build for Turing (7.5), Ampere (8.0), and Ada Lovelace (8.6) architectures.

### Verifying the Wheel

You can test the wheel file locally before uploading:

```bash
pip install ./sage-wheels/sageattention-*.whl
python -c "import sageattention; print(sageattention.__version__)"
```

### Building for Different Environments

If you need to build for different CUDA or Python versions, adjust the base image in the Dockerfile.build accordingly.

# SageAttention Build and Upload Instructions for RTX 5090 Support

This guide provides instructions for building [SageAttention](https://github.com/thu-ml/SageAttention) from source with support for the NVIDIA RTX 5090's Blackwell architecture (compute capability sm_120). The compiled wheel can be uploaded to Hugging Face for easy installation in Docker containers.

## Why Build from Source with RTX 5090 Support?

- Enable compatibility with the latest NVIDIA RTX 5090 and its Blackwell architecture
- Avoid slow compilation during Docker container startup
- Ensure consistent builds across all environments
- Pre-compile with specific CUDA and Python versions
- Leverage hardware-specific optimizations for maximum performance

## Prerequisites

- Docker installed on your system with GPU support
- NVIDIA Container Toolkit (nvidia-docker2) installed
- Hugging Face account (optional, for uploading wheels)
- Hugging Face CLI installed (`pip install huggingface_hub`) (optional)
- CUDA 12.8+ for RTX 5090 support

## Build Process

### 1. Build the Docker Image

Build the Docker image with RTX 5090 support:

```bash
docker build -t sageattention-rtx5090-builder -f Dockerfile.build .
```

This will create a Docker image that is preconfigured to build SageAttention with RTX 5090 (sm_120) support.

### 2. Run the Container to Build the Wheel

```bash
# Create a directory to store the built wheel
mkdir -p sage-wheels

# Run the container with GPU access and mount the directory
docker run --gpus all -v $(pwd)/sage-wheels:/out sageattention-rtx5090-builder
```

The container will automatically:
1. Verify GPU accessibility
2. Patch the SageAttention setup.py file for RTX 5090 support
3. Build the wheel with CUDA 12.8 and RTX 5090 compatibility
4. Copy the built wheel to your local `sage-wheels` directory

When the build process completes, you'll find the compiled wheel file in the `sage-wheels` directory on your host machine.

To customize the CUDA architecture targets, you can specify different architectures:

```bash
docker run --gpus all -v $(pwd)/sage-wheels:/out \
  -e TORCH_CUDA_ARCH_LIST="8.6 9.0 12.0+PTX" \
  sageattention-rtx5090-builder
```

## Upload to Hugging Face (Optional)

After building the wheel, you might want to upload it to Hugging Face for easy distribution:

### 1. Login to Hugging Face

```bash
huggingface-cli login
```

### 2. Create a New Repository (if needed)

```bash
huggingface-cli repo create sageattention-compiled
```

### 3. Upload the Wheel File

```bash
huggingface-cli upload-large-folder --repo-type=model wongfei2009/sageattention-compiled sage-wheels
```

## Update Your Dockerfile

To use the compiled wheel in your projects, update your Dockerfile:

```dockerfile
# Original line
RUN pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install sageattention==2.1.1

# New line (if hosted on Hugging Face)
RUN pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install https://huggingface.co/wongfei2009/sageattention-rtx5090/resolve/main/sageattention-2.1.1-cp311-cp311-linux_x86_64.whl

# Or directly if using a local wheel
COPY ./sage-wheels/sageattention-2.1.1-cp311-cp311-linux_x86_64.whl /tmp/
RUN pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install /tmp/sageattention-2.1.1-cp311-cp311-linux_x86_64.whl
```

Make sure to replace the wheel filename with the actual filename generated during the build process.

## Troubleshooting and Verification

### Verifying RTX 5090 Support

To verify that the built wheel properly supports the RTX 5090:

```python
import torch
import sageattention

# Check CUDA availability
print("CUDA available:", torch.cuda.is_available())

# Check device information
print("Device name:", torch.cuda.get_device_name(0))
print("Device capability:", torch.cuda.get_device_capability(0))

# Verify SageAttention version
print("SageAttention version:", sageattention.__version__)

# Test if SageAttention is properly detecting the RTX 5090
# This should not show any warnings about unsupported architectures
```

### Known Issues and Solutions

1. **Issue: "CUDA extension not installed for sm_120"**
   - Solution: Make sure your PyTorch version supports CUDA 12.8+ and has RTX 5090 (sm_120) support.

2. **Issue: Compilation fails with missing header files**
   - Solution: Make sure you have the correct CUDA toolkit installed. For RTX 5090, you need CUDA 12.8 or newer.

3. **Issue: "ImportError: cannot import name '_qattn_sm120'"**
   - Solution: The patch didn't properly add RTX 5090 support. Check that the patch_setup.py script ran correctly.

## Advanced Configuration

### Custom CUDA Architecture Selection

You can customize which CUDA architectures are included in the build using the `TORCH_CUDA_ARCH_LIST` environment variable:

```bash
# For RTX 5090 only
export TORCH_CUDA_ARCH_LIST="12.0"

# For RTX 5090 and previous generations
export TORCH_CUDA_ARCH_LIST="8.0 8.6 9.0 12.0+PTX"

# With specific format
export TORCH_CUDA_ARCH_LIST="compute_80;sm_80;compute_86;sm_86;compute_90;sm_90;compute_120;sm_120"
```

### Building for Different CUDA or Python Versions

To build for different CUDA or Python versions, adjust the base image in the Dockerfile.build:

```dockerfile
# For Python 3.10 with CUDA 12.8
FROM runpod/pytorch:2.8.0-py3.10-cuda12.8.1-cudnn-devel-ubuntu22.04

# For older CUDA version (not recommended for RTX 5090)
FROM runpod/pytorch:2.7.0-py3.11-cuda12.1.0-cudnn-devel-ubuntu22.04
```

Remember that RTX 5090 requires CUDA 12.8 or newer for proper support.

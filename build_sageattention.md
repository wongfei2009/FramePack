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

### 1. Prepare Files

First, create or update the following files in your project directory:

**`patch_setup.py`**: This script automatically modifies the SageAttention setup.py file to add RTX 5090 support:

```python
#!/usr/bin/env python3
"""
This script patches the SageAttention setup.py file to add RTX 5090 (sm_120) support.
"""

import os
import re

def patch_setup_py():
    """
    Patch the setup.py file of SageAttention to add sm_120 support for RTX 5090.
    """
    setup_py_path = 'setup.py'
    
    # Check if setup.py exists
    if not os.path.exists(setup_py_path):
        print(f"Error: {setup_py_path} not found!")
        return False
    
    # Read the original file
    with open(setup_py_path, 'r') as f:
        content = f.read()
    
    # Check if it already has sm_120 support
    if 'sm_120' in content or 'compute_120' in content:
        print("setup.py already has sm_120 support. No changes needed.")
        return True
    
    # Patch content for RTX 5090 support
    
    # 1. Add HAS_SM120 detection
    new_content = re.sub(
        r'HAS_SM80 = .+?\n\s*HAS_SM86 = .+?\n\s*HAS_SM90 = .+?\n',
        r'HAS_SM80 = compute_capabilities is not None and any(cc.startswith("8.0") for cc in compute_capabilities)\nHAS_SM86 = compute_capabilities is not None and any(cc.startswith("8.6") for cc in compute_capabilities)\nHAS_SM90 = compute_capabilities is not None and any(cc.startswith("9.0") for cc in compute_capabilities)\nHAS_SM120 = compute_capabilities is not None and any(cc.startswith("12.0") for cc in compute_capabilities)\n',
        content
    )
    
    # 2. Add sm_120 extensions if needed
    sm90_extension_pattern = r'(if HAS_SM90:.*?ext_modules\.append\(.*?\))'
    sm90_extension_match = re.search(sm90_extension_pattern, new_content, re.DOTALL)
    
    if sm90_extension_match:
        sm90_extension = sm90_extension_match.group(1)
        sm120_extension = sm90_extension.replace('HAS_SM90', 'HAS_SM120')
        sm120_extension = sm120_extension.replace('_qattn_sm90', '_qattn_sm120')
        sm120_extension = sm120_extension.replace('pybind_sm90.cpp', 'pybind_sm120.cpp')
        sm120_extension = sm120_extension.replace('qk_int_sv_f8_cuda_sm90.cu', 'qk_int_sv_f8_cuda_sm120.cu')
        
        # Insert SM120 extension block after SM90 block
        new_content = new_content.replace(sm90_extension, sm90_extension + '\n\n# RTX 5090 (Blackwell) extension\n' + sm120_extension)
    
    # 3. Ensure compute_capabilities is properly set
    compute_caps_pattern = r'compute_capabilities = \[(.*?)\]'
    compute_caps_match = re.search(compute_caps_pattern, new_content)
    
    if compute_caps_match:
        compute_caps = compute_caps_match.group(1)
        if '"12.0"' not in compute_caps and "'12.0'" not in compute_caps:
            # Add sm_120 to the compute capabilities list
            new_compute_caps = compute_caps
            if compute_caps.strip():
                new_compute_caps += ', "12.0"'
            else:
                new_compute_caps = '"8.0", "8.6", "9.0", "12.0"'
            new_content = new_content.replace(compute_caps_match.group(0), f'compute_capabilities = [{new_compute_caps}]')
    
    # Write the patched content
    with open(setup_py_path, 'w') as f:
        f.write(new_content)
    
    print("Successfully patched setup.py to add RTX 5090 (sm_120) support.")
    return True

def main():
    """Main function."""
    if not patch_setup_py():
        exit(1)

if __name__ == "__main__":
    main()
```

**`Dockerfile.build`**: This Dockerfile is configured to build SageAttention with RTX 5090 support:

```dockerfile
# Use RunPod PyTorch image as base
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

# Build arguments for flexible configuration
ARG TORCH_CUDA_ARCH_LIST="8.0 8.6 9.0 12.0+PTX"
ARG NVCC_THREADS=8

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all --threads=${NVCC_THREADS}"
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3-dev \
    python3-pip \
    git \
    cmake \
    build-essential \
    ninja-build \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Install Python build dependencies
RUN pip3 install --no-cache-dir \
    wheel \
    setuptools \
    build \
    ninja \
    huggingface_hub

# Set up working directory
WORKDIR /build

# Clone SageAttention repository
RUN git clone https://github.com/thu-ml/SageAttention.git
WORKDIR /build/SageAttention

# Add patched setup.py file to add RTX 5090 support to SageAttention
COPY patch_setup.py /build/patch_setup.py
RUN python3 /build/patch_setup.py

# Script for building the wheel
RUN echo '#!/bin/bash \n\
nvidia-smi \n\
python -c "import torch; print(\"CUDA available:\", torch.cuda.is_available())" \n\
cd /build/SageAttention \n\
export TORCH_CUDA_ARCH_LIST="8.0 8.6 9.0 12.0+PTX" \n\
python setup.py bdist_wheel \n\
if [ -d "/out" ]; then \n\
  cp dist/*.whl /out/ \n\
  echo "Wheel copied to /out directory" \n\
fi' > /build/build_wheel.sh && chmod +x /build/build_wheel.sh

# Set the entrypoint to our build script
ENTRYPOINT ["/build/build_wheel.sh"]
```

### 2. Build the Docker Image

Build the Docker image with RTX 5090 support:

```bash
docker build -t sageattention-rtx5090-builder -f Dockerfile.build .
```

This will create a Docker image that is preconfigured to build SageAttention with RTX 5090 (sm_120) support.

### 3. Run the Container to Build the Wheel

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

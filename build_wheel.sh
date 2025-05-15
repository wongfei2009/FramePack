#!/bin/bash

# Display GPU information
nvidia-smi

# Check if CUDA is available
python -c "import torch; print(\"CUDA available:\", torch.cuda.is_available())"

# Set the CUDA architecture list for RTX 5090 support
export TORCH_CUDA_ARCH_LIST="8.0 8.6 9.0 12.0"

# Navigate to SageAttention directory
cd /build/SageAttention

# Build the wheel
python setup.py bdist_wheel

# Copy the wheel to the mounted directory if it exists
cd /build/SageAttention
if [ -d "/out" ]; then
  cp dist/*.whl /out/
  echo "Wheel copied to /out directory"
  echo "Wheel file: $(ls /out/*.whl)"
fi

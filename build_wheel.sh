#!/bin/bash

#!/bin/bash

# Display GPU information
nvidia-smi

# Check if CUDA is available
python -c "import torch; print(\"CUDA available:\", torch.cuda.is_available())"

# Set the CUDA architecture list for RTX 5090 support
export TORCH_CUDA_ARCH_LIST="8.0 8.6 9.0 12.0+PTX"

# Navigate to SageAttention directory
cd /build/SageAttention

# Apply patches (already done in Dockerfile, but let's make sure)
echo "Applying patches to SageAttention..."
if [ -f "/build/patch_setup.py" ]; then
  echo "Running patch_setup.py..."
  python /build/patch_setup.py
  if [ $? -ne 0 ]; then
    echo "ERROR: Failed to apply setup.py patch!"
    exit 1
  fi
fi

if [ -f "/build/patch_core.py" ]; then
  echo "Running patch_core.py..."
  # For debugging, first list the files in sageattention directory
  ls -la /build/SageAttention/sageattention/
  
  python /build/patch_core.py
  if [ $? -ne 0 ]; then
    echo "ERROR: Failed to apply core.py patch!"
    # Print core.py content for debugging
    echo "First 50 lines of core.py:"
    head -n 50 /build/SageAttention/sageattention/core.py
    exit 1
  fi
fi

# Build the wheel
python setup.py bdist_wheel

# Verify the wheel
echo "Verifying the wheel..."
wheel_file=$(ls dist/*.whl)
wheel_dir="/tmp/wheel_verify"
rm -rf $wheel_dir
mkdir -p $wheel_dir
cd $wheel_dir

# Copy the wheel to the verification directory
cp /build/SageAttention/dist/*.whl ./

# Install the wheel from local path
pip install *.whl

# Quick verification
python -c "
import torch
import sageattention
print('SageAttention version:', sageattention.__version__)
try:
    print('SM120_ENABLED flag check...')
    from sageattention.core import SM120_ENABLED
    print('SM120_ENABLED =', SM120_ENABLED)
except ImportError:
    print('SM120_ENABLED flag not found!')
print('Checking sageattn function...')
import inspect
print(inspect.getsource(sageattention.sageattn))
"

# Copy the wheel to the mounted directory if it exists
cd /build/SageAttention
if [ -d "/out" ]; then
  cp dist/*.whl /out/
  echo "Wheel copied to /out directory"
  echo "Wheel file: $(ls /out/*.whl)"
fi

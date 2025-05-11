#!/bin/bash

# Run SSH setup first
echo "Setting up SSH..."
/app/setup_ssh.sh

# Install SageAttention if not already installed
if [ ! -d "/app/SageAttention" ] || [ ! -f "/app/SageAttention/setup.py" ]; then
    echo "Installing SageAttention at runtime..."
    cd /app
    git clone https://github.com/thu-ml/SageAttention
    cd SageAttention
    pip install -e .
    cd /app
fi

# Then run the main workspace setup
echo "Setting up workspace..."
exec /app/setup_workspace.sh
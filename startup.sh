#!/bin/bash

# Run SSH setup first
echo "Setting up SSH..."
/app/setup_ssh.sh

# Then run the main workspace setup
echo "Setting up workspace..."
exec /app/setup_workspace.sh
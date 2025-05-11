#!/bin/bash

echo "======= SSH Setup ======="

# If PUBLIC_KEY is provided, add it to authorized_keys
if [ -n "$PUBLIC_KEY" ]; then
  echo "$PUBLIC_KEY" > /root/.ssh/authorized_keys
  chmod 600 /root/.ssh/authorized_keys
  echo "Added provided public key to authorized_keys"
else
  echo "No public key provided, continuing with password authentication"
  # Optionally, you could set a default password here if needed
  # echo "root:your_default_password" | chpasswd
fi

# Start SSH service
service ssh start
echo "SSH service started"
echo "======================="
# FramePack Deployment Guide for RunPod

This guide provides instructions for deploying FramePack to RunPod with a focus on secure access via SSH port forwarding.

## 1. Prerequisites

- **Install `runpodctl`**:
  - macOS: `brew install runpod/runpodctl/runpodctl`
  - Linux: `wget -qO- cli.runpod.net | sudo bash`
  - Windows: `wget https://github.com/runpod/runpodctl/releases/latest/download/runpodctl-windows-amd64.exe -O runpodctl.exe`

- **Configure API Key**:
  ```bash
  runpodctl config --apiKey YOUR_API_KEY
  ```

- **Create a Network Volume** in RunPod UI (Secure Cloud > Volumes):
  - Note down the **Volume ID** and **Data Center ID**

## 2. Deploy the Pod (Secure Method)

To deploy without exposing the endpoint publicly (recommended):

```bash
runpodctl create pod \
    --name "framepack-app" \
    --imageName "yourdockerhubusername/framepack-app" \
    --gpuType "NVIDIA GeForce RTX 3090" \
    --gpuCount 1 \
    --secureCloud \
    --dataCenterId "YOUR_DATACENTER_ID" \
    --networkVolumeId "vol-xxxxxxxxxxxxxxxxx" \
    --volumePath "/app" \
    --containerDiskSize 20
    # Note: We intentionally omit the --ports flag for security
```

## 3. SSH Port Forwarding Access

### 3.1 Get SSH Connection Details

From the RunPod web interface:
- Go to your pod details > "Connect" dropdown > "Connect via SSH"
- Copy the SSH command (e.g., `ssh root@123.45.67.89 -p 12345`)

Or using runpodctl:
```bash
runpodctl get pod
runpodctl get pod YOUR_POD_ID
# Look for SSH connection details
```

### 3.2 Create SSH Tunnel

Run this command on your local machine to create the tunnel:

```bash
ssh -L 8080:localhost:7860 root@POD_IP -p SSH_PORT
```

Where:
- `8080` is your desired local port
- `7860` is FramePack's internal port
- `POD_IP` and `SSH_PORT` are from the SSH connection details

Access the application at `http://localhost:8080` while keeping the terminal window open.

### 3.3 Optional: Create an SSH Config Entry

For convenience, add to your `~/.ssh/config`:

```
Host framepack
  HostName POD_IP
  User root
  Port SSH_PORT
  LocalForward 8080 localhost:7860
```

Then simply connect with:
```bash
ssh framepack
```

## 4. Troubleshooting

- **Connection Refused**: Verify pod is running and SSH port is correct
- **Cannot Access Application**: Check if the app is running on port 7860 in the pod
- **Permission Denied**: Verify username, password, or SSH key

## 5. Security Tips

- Use SSH key authentication (optional):
  ```bash
  # Generate key if needed
  ssh-keygen -t ed25519
  
  # Add to RunPod
  runpodctl ssh add-key --publicKey "$(cat ~/.ssh/id_ed25519.pub)"
  ```
- Keep the SSH tunnel running only when needed
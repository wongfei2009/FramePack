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

## 2. Deploy the Pod (Secure Method)

To deploy without exposing the endpoint publicly (recommended):

```bash
runpodctl create pod \
    --name "framepack" \
    --communityCloud \
    --startSSH \
    --gpuType "NVIDIA GeForce RTX 5090" \
    --gpuCount 1 \
    --templateId "sh69ed8ft7"
```

## 3. SSH Port Forwarding Access

### 3.1 Get SSH Connection Details

From the RunPod web interface:
- Go to your pod details > "Connect" dropdown > "Connect via SSH"
- Copy the SSH command (e.g., `ssh root@123.45.67.89 -p 12345`)

Or using runpodctl:
```bash
POD_ID=$(runpodctl get pod | grep -v '^ID' | cut -f1)
```

### 3.2 Create SSH Tunnel

Run this command on your local machine to create the tunnel:

```bash
eval $(runpodctl ssh connect $POD_ID) -L 7860:localhost:7860
```

Access the application at `http://localhost:7860` while keeping the terminal window open.

## 4. Troubleshooting

- **Connection Refused**: Verify pod is running and SSH port is correct
- **Cannot Access Application**: Check if the app is running on port 7860 in the pod
# FramePack RunPod Template

This template provides a ready-to-use environment for running FramePack progressive video generation on RunPod.

## Environment Variables

The following environment variables can be customized:

| Variable | Description | Default Value |
|----------|-------------|---------------|
| `GRADIO_SERVER_NAME` | Gradio server host binding | `0.0.0.0` |
| `GRADIO_SERVER_PORT` | Gradio server port | `7860` |
| `HF_HOME` | Hugging Face cache directory | `/workspace/framepack/huggingface_cache/huggingface` |
| `TRANSFORMERS_CACHE` | Transformers models cache | `/workspace/framepack/huggingface_cache/transformers` |
| `DIFFUSERS_CACHE` | Diffusers models cache | `/workspace/framepack/huggingface_cache/diffusers` |
| `LOCAL_MODELS_DIR` | Directory for local model storage | `/workspace/framepack/local_models` |
| `OUTPUTS_DIR` | Directory where generated videos are saved | `/workspace/framepack/outputs` |

## Deployment Steps

### 1. Prerequisites

- **Install `runpodctl`**:
  - macOS: `brew install runpod/runpodctl/runpodctl`
  - Linux: `wget -qO- cli.runpod.net | sudo bash`
  - Windows: Download from [GitHub Releases](https://github.com/runpod/runpodctl/releases)

- **Configure API Key**:
  ```bash
  runpodctl config --apiKey YOUR_API_KEY
  ```

### 2. Deploy the Pod (Secure Method)

Deploy FramePack without exposing the endpoint publicly (recommended):

```bash
runpodctl create pod \
    --name "framepack-video-generator" \
    --communityCloud \
    --startSSH \
    --gpuType "NVIDIA GeForce RTX 5090" \
    --gpuCount 1 \
    --imageName "wongfei2009/framepack:latest" \
    --containerDiskSize 5 \
    --volumeSize 55 \
    --volumePath "/workspace"
```

### 3. SSH Port Forwarding Access

#### 3.1 Get SSH Connection Details

From the RunPod web interface:
- Go to your pod details > "Connect" dropdown > "SSH over exposed TCP"
- Copy the SSH command (e.g., `ssh root@80.15.7.37 -p 48896 -i ~/.ssh/id_ed25519`)

#### 3.2 Create SSH Tunnel

Run this command on your local machine to create the tunnel:

```bash
ssh root@80.15.7.37 -p 48896 -i ~/.ssh/id_ed25519 -L 7860:localhost:7860
```

#### 3.3 Access FramePack

While keeping the terminal window with the SSH tunnel open, access FramePack at:

```
http://localhost:7860
```

## Using FramePack

Once you have accessed the FramePack interface, you can:

1. Upload an input image
2. Enter a text prompt to guide the video generation
3. Adjust generation settings:
   - Frame count
   - FPS
   - Section controls
4. Start generation and monitor progress
5. Download the generated video

### Section Controls

FramePack supports fine-grained control over different segments of the generated video:

- Specify different reference images for different sections
- Use different text prompts for each section
- Create videos with dynamic scene changes or different visual styles

For more information on section controls, refer to the in-app documentation.

## Troubleshooting

- **Connection Refused**: Verify pod is running and SSH port is correct
- **Cannot Access Application**: Check if the app is running on port 7860 in the pod
- **Slow Model Loading**: The first run may take time to download models to the cache
- **Out of Memory Errors**: Reduce frame count or video resolution

## Technical Information

FramePack uses a unique progressive generation approach for efficient video generation:

- Generates videos section-by-section instead of all frames at once
- Maintains constant memory usage regardless of video length
- Provides immediate visual feedback during generation
- Uses context packing for efficient temporal consistency
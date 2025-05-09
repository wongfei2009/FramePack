# FramePack Deployment Guide for RunPod using runpodctl

This guide provides instructions for deploying the FramePack Docker image to RunPod using the `runpodctl` command-line interface, with a focus on setting up persistent storage and correct port configuration.

**Docker Image:** `yourdockerhubusername/framepack-app` (Replace with your actual image name)

## 1. Prerequisites

Before deploying, ensure you have:

1.  **Installed `runpodctl`**:
    * **macOS (Homebrew):** `brew install runpod/runpodctl/runpodctl`
    * **Linux (wget):** `wget -qO- cli.runpod.net | sudo bash`
    * **Windows (PowerShell):** Download the `.exe` from the [RunPod CLI releases page](https://github.com/runpod/runpodctl/releases) and add it to your system's PATH.
        Example: `wget https://github.com/runpod/runpodctl/releases/latest/download/runpodctl-windows-amd64.exe -O runpodctl.exe`

2.  **Configured `runpodctl` with your API Key**:
    * Obtain your API key from your RunPod account settings (Settings > API Keys).
    * Configure `runpodctl` by running:
        ```bash
        runpodctl config --apiKey YOUR_API_KEY
        ```
        Replace `YOUR_API_KEY` with the actual key.

3.  **Docker Image Pushed to a Registry**:
    * Your FramePack Docker image (e.g., `yourdockerhubusername/framepack-app`) must be pushed to a container registry accessible by RunPod (like Docker Hub).

4.  **Network Volume Created (for Persistent Storage)**:
    * FramePack requires persistent storage for models and cache. This is best handled using a RunPod Network Volume.
    * **Action**: Via the RunPod web UI, navigate to **Secure Cloud > Volumes**.
    * Click **Create Volume**.
    * Configure the volume:
        * **Name**: e.g., `framepack-data-volume`
        * **Size**: Choose an appropriate size (e.g., 50 GB, 100 GB, or more, depending on your model storage needs).
        * **Data Center**: Select your preferred data center. **Crucially, ensure the pod will be deployed in this same data center.**
    * **Important**: After creation, note down the **Volume ID** (e.g., `vol-xxxxxxxxxxxxxxxxx`) and the **Data Center ID** (e.g., `US-KS1`) of this volume. You will need both for the `runpodctl` deployment command.
    * As of the latest review, `runpodctl` does not provide commands to create or manage network volumes directly; this step must be performed via the RunPod web UI.

## 2. Deploying the Pod with `runpodctl`

The primary command for deploying your FramePack application as a new pod is `runpodctl create pod`. This command allows you to specify all necessary configurations, including GPU type, Docker image, port mapping, and importantly, the attachment of your pre-created Network Volume for persistent storage.

### Understanding Key Parameters for FramePack Deployment:

Your FramePack application (as per your Dockerfile and application logic) expects to use paths like `/app/local_models` and `/app/huggingface_cache`. To ensure data persistence across pod restarts or new deployments, your Network Volume must be mounted to `/app` inside the container.

Here's a breakdown of the essential `runpodctl create pod` flags confirmed from the `runpodctl` source code:

* `--name "your-pod-name"`: A friendly name for your pod (e.g., "framepack-prod").
* `--imageName "yourdockerhubusername/framepack-app:tag"`: The full path to your Docker image on a registry (e.g., Docker Hub). **(Required)**
* `--gpuType "GPU_TYPE_ID"`: The specific GPU type ID you want to use (e.g., "NVIDIA GeForce RTX 3090"). You can find available types via the RunPod UI or potentially `runpodctl get cloud`. **(Required)**
* `--gpuCount <number>`: Number of GPUs for the pod (default: 1).
* `--secureCloud`: Use this flag to deploy to Secure Cloud. Network Volumes are typically located in Secure Cloud, so this is recommended.
* `--dataCenterId "YOUR_DATACENTER_ID"`: Specify the ID of the data center where your pod should be created. **This MUST match the Data Center ID of your Network Volume** to allow successful attachment.
* `--networkVolumeId "vol-xxxxxxxxxxxxxxxxx"`: The ID of your pre-created Network Volume. This is crucial for persistent storage.
* `--volumePath "/app"`: The mount point *inside the container* for the volume specified by `--networkVolumeId`. For FramePack, this **must be `/app`**.
* `--ports "7860/http"`: Exposes the container's port `7860` (where Gradio listens) for HTTP access. Your app will be accessible via a URL like `https://<YOUR_POD_ID>-7860.proxy.runpod.net`.
* `--containerDiskSize <GB>`: (Default: 20 GB) Size for the pod's ephemeral system disk. This disk is generally for the OS and temporary files, not persistent application data if you're using a network volume for `/app`.
* `--volumeSize <GB>`: (Default: 1 GB) This defines the size of the pod's *own* local persistent storage, which is separate from the Network Volume. If your Network Volume is mounted at `/app` and contains all your primary persistent data, this local pod volume (often mounted at `/runpod` by default if not explicitly mapped elsewhere by `--volumePath` without `--networkVolumeId`) might be of minimal use for FramePack's core data. You can typically leave it at its default or a small size.
* `--env "KEY=VALUE"`: (Optional) Allows you to set environment variables in the container. You can use this multiple times (e.g., `--env "HF_HOME=/app/huggingface_cache"`). However, it's often better to define such environment variables directly in your Dockerfile.

### Verified `runpodctl create pod` Command for FramePack:

Remember to replace placeholder values (like `yourdockerhubusername/framepack-app:latest`, GPU type, volume ID, and data center ID) with your actual information.

```bash
runpodctl create pod \
    --name "framepack-app-prod" \
    --imageName "yourdockerhubusername/framepack-app" \
    --gpuType "NVIDIA GeForce RTX 3090" \
    --gpuCount 1 \
    --secureCloud \
    --dataCenterId "YOUR_NETWORK_VOLUME_DATACENTER_ID" \
    \
    # --- Persistent Storage (Network Volume) ---
    --networkVolumeId "vol-xxxxxxxxxxxxxxxxx" \
    --volumePath "/app" \
    \
    # --- Port Configuration ---
    --ports "7860/http" \
    \
    # --- Other Pod Sizing Parameters (Defaults are often fine) ---
    --containerDiskSize 20 \
    # --volumeSize 5 \ # Optional: Pod's local persistent volume size in GB.
                        # Your main data is on the Network Volume at /app.
    # --minVcpuCount 2 \
    # --minMemoryInGb 8 \
    # --env "HF_HOME=/app/huggingface_cache" # Example if not set in Dockerfile
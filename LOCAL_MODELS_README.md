# Using Local Hunyuan Video Models

This project has been modified to support local model loading. This README explains how to set up and use local Hunyuan video models instead of downloading them automatically from Hugging Face.

## Directory Structure

The project will look for models in the `local_models` directory with the following structure:

```
local_models/
├── text_encoders/
│   ├── clip_l.safetensors
│   └── llava_llama3_fp8_scaled.safetensors
├── vae/
│   └── hunyuan_video_vae_bf16.safetensors
├── clip_vision/
│   └── sigclip_vision_patch14_384.safetensors
└── diffusion_models/
    └── FramePackI2V_HY_bf16.safetensors
```

## How to Set Up Local Models

### Using single file safetensors

Download one of the following files for the main transformer model:
```
https://huggingface.co/Kijai/HunyuanVideo_comfy/resolve/main/FramePackI2V_HY_bf16.safetensors
```

Place it in:
```
local_models/diffusion_models/
```

### For required text encoders, VAE and sigclip vision models:

Download these required support models:
```
https://huggingface.co/Comfy-Org/HunyuanVideo_repackaged/resolve/main/split_files/text_encoders/clip_l.safetensors
https://huggingface.co/Comfy-Org/HunyuanVideo_repackaged/resolve/main/split_files/text_encoders/llava_llama3_fp16.safetensors
https://huggingface.co/Comfy-Org/HunyuanVideo_repackaged/resolve/main/split_files/vae/hunyuan_video_vae_bf16.safetensors
https://huggingface.co/Comfy-Org/sigclip_vision_384/resolve/main/sigclip_vision_patch14_384.safetensors
```

Place them in the appropriate directories as shown in the directory structure above.

## Automatic Download Script

For convenience, this project includes a Python script to automatically download all required models:

```bash
# Install dependencies if you haven't already
pip install requests tqdm huggingface_hub

# Download all models with default settings (bf16 transformer model)
python download_hunyuan_models.py

# Specify a different download directory
python download_hunyuan_models.py --base-dir /path/to/models
```

## Troubleshooting

If you encounter issues with local model loading:

1. Check the console output to see which models are failing to load locally
2. Verify that all required files are present in each model directory
3. Ensure file permissions allow the application to read the model files
4. Try deleting and re-downloading the problematic model
# Using Local Hunyuan Video Models

This project has been modified to support local model loading. This README explains how to set up and use local Hunyuan video models instead of downloading them automatically from Hugging Face.

## Directory Structure

The project will look for models in the `local_models` directory with the following structure:

```
local_models/
├── HunyuanVideo/
│   ├── text_encoder/
│   ├── text_encoder_2/
│   ├── tokenizer/
│   ├── tokenizer_2/
│   └── vae/
├── flux_redux_bfl/
│   ├── feature_extractor/
│   └── image_encoder/
└── FramePackI2V_HY/
```

## How to Set Up Local Models

1. First, download the models from Hugging Face or obtain them locally.

2. Place the models in the appropriate directories shown above.

3. Make sure each model directory contains all necessary files (config.json, model weights, etc.)

4. Run the application as usual with `python demo_gradio.py`

The application will now:
- First attempt to load models from the local directory
- Fall back to downloading from Hugging Face if the local models don't exist or are invalid

## Manually Preparing Models

If you want to download the models manually, you can use the Hugging Face CLI:

```bash
# Install the Hugging Face Hub CLI if you haven't already
pip install huggingface_hub

# Download the models with their subdirectories
huggingface-cli download hunyuanvideo-community/HunyuanVideo --local-dir ./local_models/HunyuanVideo --local-dir-use-symlinks False
huggingface-cli download lllyasviel/flux_redux_bfl --local-dir ./local_models/flux_redux_bfl --local-dir-use-symlinks False
huggingface-cli download lllyasviel/FramePackI2V_HY --local-dir ./local_models/FramePackI2V_HY --local-dir-use-symlinks False
```

## Troubleshooting

If you encounter issues with local model loading:

1. Check the console output to see which models are failing to load locally
2. Verify that all required files are present in each model directory
3. Ensure file permissions allow the application to read the model files
4. Try deleting and re-downloading the problematic model
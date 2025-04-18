#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to download Hunyuan Video Models for FramePack local use
"""

import os
import argparse
import requests
from tqdm import tqdm
from pathlib import Path
import time

# Define model URLs
MODEL_URLS = {
    "diffusion_models/FramePackI2V_HY_bf16.safetensors": 
        "https://huggingface.co/Kijai/HunyuanVideo_comfy/resolve/main/FramePackI2V_HY_bf16.safetensors",
    "text_encoders/clip_l.safetensors": 
        "https://huggingface.co/Comfy-Org/HunyuanVideo_repackaged/resolve/main/split_files/text_encoders/clip_l.safetensors",
    "text_encoders/llava_llama3_fp16.safetensors": 
        "https://huggingface.co/Comfy-Org/HunyuanVideo_repackaged/resolve/main/split_files/text_encoders/llava_llama3_fp16.safetensors",
    "vae/hunyuan_video_vae_bf16.safetensors": 
        "https://huggingface.co/Comfy-Org/HunyuanVideo_repackaged/resolve/main/split_files/vae/hunyuan_video_vae_bf16.safetensors",
    "clip_vision/sigclip_vision_patch14_384.safetensors": 
        "https://huggingface.co/Comfy-Org/sigclip_vision_384/resolve/main/sigclip_vision_patch14_384.safetensors",
}

def download_file(url, file_path, chunk_size=8192):
    """
    Download a file from URL to the specified path with a progress bar
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Check if file already exists
    if os.path.exists(file_path):
        print(f"File {file_path} already exists. Skipping download.")
        return True
    
    print(f"Downloading {url} to {file_path}")
    
    try:
        # Start with a HEAD request to get file size
        response = requests.head(url, allow_redirects=True)
        file_size = int(response.headers.get('Content-Length', 0))
        
        # Now download the file
        response = requests.get(url, stream=True, allow_redirects=True)
        response.raise_for_status()
        
        with open(file_path, 'wb') as f:
            with tqdm(total=file_size, unit='iB', unit_scale=True, desc=os.path.basename(file_path)) as bar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
        
        print(f"Successfully downloaded {file_path}")
        return True
        
    except requests.RequestException as e:
        print(f"Error downloading {url}: {e}")
        # If file was partially downloaded, remove it
        if os.path.exists(file_path):
            os.remove(file_path)
        return False

def main():
    parser = argparse.ArgumentParser(description="Download Hunyuan Video Models for local use")
    parser.add_argument("--base-dir", type=str, default="local_models", 
                        help="Base directory to store downloaded models")
    parser.add_argument("--retry", type=int, default=3,
                        help="Number of retry attempts for failed downloads")
    parser.add_argument("--retry-delay", type=int, default=5,
                        help="Delay in seconds between retry attempts")
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    
    print(f"Downloading models to {base_dir.absolute()}")
    
    # Create the base directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)
    
    # Download each model
    for model_path, url in MODEL_URLS.items():
        full_path = base_dir / model_path
        
        # Try multiple times with delay to handle temporary failures
        success = False
        for attempt in range(args.retry):
            if attempt > 0:
                print(f"Retry attempt {attempt+1}/{args.retry} after {args.retry_delay} seconds...")
                time.sleep(args.retry_delay)
                
            success = download_file(url, str(full_path))
            if success:
                break
        
        if not success:
            print(f"Failed to download {model_path} after {args.retry} attempts.")
    
    print("\nDownload summary:")
    missing_files = []
    
    for model_path in MODEL_URLS.keys():
        full_path = base_dir / model_path
        if os.path.exists(full_path):
            file_size = os.path.getsize(full_path) / (1024 * 1024)  # Convert to MB
            print(f"✓ {model_path} ({file_size:.2f} MB)")
        else:
            print(f"✗ {model_path} (MISSING)")
            missing_files.append(model_path)
    
    if missing_files:
        print("\nWARNING: The following files are missing:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nPlease try downloading them manually or run the script again.")
        return 1
    else:
        print("\nAll models successfully downloaded!")
        print("You can now use local models with FramePack.")
        return 0

if __name__ == "__main__":
    exit(main())

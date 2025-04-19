"""
Model loading and management for FramePack.
"""

import os
import torch
from diffusers import AutoencoderKLHunyuanVideo
from transformers import (
    LlamaModel, 
    CLIPTextModel, 
    LlamaTokenizerFast, 
    CLIPTokenizer,
    SiglipImageProcessor, 
    SiglipVisionModel
)
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.memory import (
    cpu, gpu, get_cuda_free_memory_gb, 
    move_model_to_device_with_memory_preservation, 
    offload_model_from_device_for_memory_preservation, 
    unload_complete_models, 
    load_model_as_complete,
    DynamicSwapInstaller
)
from diffusers_helper.optimization import (
    configure_teacache, optimize_for_inference, 
    aggressive_memory_cleanup, warmup_model
)

# Define a function to load models locally first or download if not found
def load_model_locally_or_download(model_cls, model_id, subfolder=None, **kwargs):
    """
    Load a model from local path if available, or download from HuggingFace.
    
    Args:
        model_cls: The model class to instantiate
        model_id: HuggingFace model ID
        subfolder: Optional subfolder in the model repository
        **kwargs: Additional arguments for model loading
        
    Returns:
        Instantiated model
    """
    # Extract model name from model_id (e.g. "hunyuanvideo-community/HunyuanVideo" -> "HunyuanVideo")
    model_name = model_id.split('/')[-1]
    
    # Construct local path - use the same directory structure as HF
    local_path = os.path.join(get_local_models_dir(), model_name)
    if subfolder:
        local_path = os.path.join(local_path, subfolder)
    
    try:
        # Try loading locally first
        print(f"Attempting to load {model_name}{f'/{subfolder}' if subfolder else ''} from local path: {local_path}")
        return model_cls.from_pretrained(local_path, **kwargs)
    except (OSError, ValueError, FileNotFoundError) as e:
        print(f"Could not load model from {local_path}, downloading from HF: {e}")
        return model_cls.from_pretrained(model_id, subfolder=subfolder, **kwargs)

def get_local_models_dir():
    """
    Get the local models directory, creating it if it doesn't exist.
    
    Returns:
        Path to local models directory
    """
    local_models_dir = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'local_models')))
    os.makedirs(local_models_dir, exist_ok=True)
    return local_models_dir

def configure_sage_attention(transformer, has_sage_attn=False):
    """
    Configure Sage Attention for the transformer model if available.
    
    Args:
        transformer: The transformer model
        has_sage_attn: Whether Sage Attention is available
    """
    # print("Activating Sage Attention...")
    try:
        if has_sage_attn:
            # Method 1: Try to set it directly if using appropriate attention mechanism
            if hasattr(transformer, 'set_use_sage_attention'):
                transformer.set_use_sage_attention(True)
                # print("Sage Attention activated successfully via direct method")
            # Method 2: Try to set the attention processor
            elif hasattr(transformer, "set_attn_processor"):
                transformer.set_attn_processor({"sage_attention": {}})
                # print("Sage Attention activated via attention processor")
            else:
                # print("This model already uses Sage Attention by default when available")
                pass
        else:
            # print("Sage Attention module detected but will be used in fallback mode")
            pass
    except Exception as e:
        # print(f"Failed to activate Sage Attention: {e}")
        # print("Continuing without Sage Attention optimization")
        pass

class FramePackModels:
    """
    Class to manage all models required for FramePack.
    """
    def __init__(self):
        self.text_encoder = None
        self.text_encoder_2 = None
        self.tokenizer = None
        self.tokenizer_2 = None
        self.vae = None
        self.feature_extractor = None
        self.image_encoder = None
        self.transformer = None
        
        # Set environment variables for model loading
        self.local_models_dir = get_local_models_dir()
        os.environ['HF_HOME'] = self.local_models_dir
        
        # Determine VRAM availability
        self.free_mem_gb = get_cuda_free_memory_gb(gpu)
        self.high_vram = self.free_mem_gb > 60
        
        print(f'Free VRAM {self.free_mem_gb} GB')
        print(f'High-VRAM Mode: {self.high_vram}')
    
    def load_models(self, has_sage_attn=False):
        """
        Load all required models.
        
        Args:
            has_sage_attn: Whether Sage Attention is available
        """
        # Load text encoders and tokenizers
        self.text_encoder = load_model_locally_or_download(
            LlamaModel, 
            "hunyuanvideo-community/HunyuanVideo", 
            subfolder='text_encoder', 
            torch_dtype=torch.float16
        ).cpu()

        self.text_encoder_2 = load_model_locally_or_download(
            CLIPTextModel, 
            "hunyuanvideo-community/HunyuanVideo", 
            subfolder='text_encoder_2', 
            torch_dtype=torch.float16
        ).cpu()

        self.tokenizer = load_model_locally_or_download(
            LlamaTokenizerFast, 
            "hunyuanvideo-community/HunyuanVideo", 
            subfolder='tokenizer'
        )

        self.tokenizer_2 = load_model_locally_or_download(
            CLIPTokenizer, 
            "hunyuanvideo-community/HunyuanVideo", 
            subfolder='tokenizer_2'
        )

        # Load VAE
        self.vae = load_model_locally_or_download(
            AutoencoderKLHunyuanVideo, 
            "hunyuanvideo-community/HunyuanVideo", 
            subfolder='vae', 
            torch_dtype=torch.float16
        ).cpu()

        # Load feature extractor and image encoder
        self.feature_extractor = load_model_locally_or_download(
            SiglipImageProcessor, 
            "lllyasviel/flux_redux_bfl", 
            subfolder='feature_extractor'
        )

        self.image_encoder = load_model_locally_or_download(
            SiglipVisionModel, 
            "lllyasviel/flux_redux_bfl", 
            subfolder='image_encoder', 
            torch_dtype=torch.float16
        ).cpu()

        # Load transformer
        self.transformer = load_model_locally_or_download(
            HunyuanVideoTransformer3DModelPacked, 
            'lllyasviel/FramePackI2V_HY', 
            torch_dtype=torch.bfloat16
        ).cpu()

        # Set models to eval mode
        self.vae.eval()
        self.text_encoder.eval()
        self.text_encoder_2.eval()
        self.image_encoder.eval()
        self.transformer.eval()

        # Configure Sage Attention
        configure_sage_attention(self.transformer, has_sage_attn)

        # Configure models based on VRAM availability
        self._configure_models()
        
    def _configure_models(self):
        """Configure models based on available VRAM."""
        # VAE configuration
        if not self.high_vram:
            self.vae.enable_slicing()
            self.vae.enable_tiling()

        # Set transformer output to high quality for inference
        self.transformer.high_quality_fp32_output_for_inference = True

        # Set model precision
        self.transformer.to(dtype=torch.bfloat16)
        self.vae.to(dtype=torch.float16)
        self.image_encoder.to(dtype=torch.float16)
        self.text_encoder.to(dtype=torch.float16)
        self.text_encoder_2.to(dtype=torch.float16)

        # Disable gradients for all models
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        self.image_encoder.requires_grad_(False)
        self.transformer.requires_grad_(False)

        # Device placement based on VRAM
        if not self.high_vram:
            # Use dynamic swapping for low VRAM systems
            DynamicSwapInstaller.install_model(self.transformer, device=gpu)
            DynamicSwapInstaller.install_model(self.text_encoder, device=gpu)
        else:
            # Move models to GPU for high VRAM systems
            self.text_encoder.to(gpu)
            self.text_encoder_2.to(gpu)
            self.image_encoder.to(gpu)
            self.vae.to(gpu)
            self.transformer.to(gpu)

    def prepare_for_inference(self, gpu_memory_preservation, use_teacache, steps, rel_l1_thresh=0.15):
        """
        Prepare models for inference.
        
        Args:
            gpu_memory_preservation: Amount of GPU memory to preserve during inference
            use_teacache: Whether to use TeaCache for acceleration
            steps: Number of inference steps
            rel_l1_thresh: Threshold for TeaCache relative L1 distance (lower = faster but lower quality)
        """
        if not self.high_vram:
            # Clean up memory before loading transformer
            unload_complete_models()
            aggressive_memory_cleanup()
            
            # Load transformer with optimized memory settings
            print(f"Loading transformer with {gpu_memory_preservation}GB memory preservation")
            move_model_to_device_with_memory_preservation(
                self.transformer, 
                target_device=gpu, 
                preserved_memory_gb=gpu_memory_preservation
            )

        # Configure TeaCache if enabled
        if use_teacache:
            free_mem = get_cuda_free_memory_gb(gpu)
            print(f"Configuring TeaCache with {free_mem:.1f}GB available VRAM, {steps} steps, and threshold {rel_l1_thresh:.4f}")
            
            # Try direct initialization first - safer approach
            try:
                # Directly initialize TeaCache with parameters
                self.transformer.initialize_teacache(
                    enable_teacache=True,
                    num_steps=steps,
                    rel_l1_thresh=float(rel_l1_thresh)
                )
                print(f"Direct TeaCache initialization successful with threshold={rel_l1_thresh:.4f}")
            except Exception as e:
                print(f"Direct initialization failed: {e}, falling back to configure_teacache")
                # Fall back to the helper function
                configure_teacache(self.transformer, vram_gb=free_mem, steps=steps, rel_l1_thresh=rel_l1_thresh)
            
            # Track memory after TeaCache configuration
            if torch.cuda.is_available():
                current_mem = torch.cuda.memory_allocated() / (1024**3)
                print(f"Memory after TeaCache configuration: {current_mem:.2f}GB")
        else:
            self.transformer.initialize_teacache(enable_teacache=False)
            print("TeaCache disabled as per user request")
            
        # Apply additional inference optimizations
        optimize_for_inference(self.transformer, high_vram=self.high_vram)

from diffusers_helper.hf_login import login

import os
import sys
import importlib.util
from custom_asyncio_policy import apply_asyncio_fixes

# Apply custom asyncio fixes before any other imports that might use asyncio
apply_asyncio_fixes()

# Debug helper to check why sageattention isn't being imported
def debug_import_paths(module_name="sageattention"):
    """Print debugging information about Python's import system for a specific module"""
    print(f"\n--- Debug import paths for {module_name} ---")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    
    # Check if the module is already imported
    if module_name in sys.modules:
        print(f"Module {module_name} is already imported at: {sys.modules[module_name].__file__}")
        return True
    
    # Try to find the module in sys.path
    found = False
    for path in sys.path:
        spec = importlib.util.find_spec(module_name, [path])
        if spec is not None:
            print(f"Module {module_name} found at: {spec.origin}")
            found = True
            break
    
    if not found:
        print(f"Module {module_name} not found in any of these paths:")
        for i, path in enumerate(sys.path):
            print(f"  {i}: {path}")
    
    print("--- End debug info ---\n")
    return found

# Run the debug function to see why sageattention isn't being imported
debug_import_paths("sageattention")

# Define a function to load models locally first or download if not found
def load_model_locally_or_download(model_cls, model_id, subfolder=None, **kwargs):
    # Extract model name from model_id (e.g. "hunyuanvideo-community/HunyuanVideo" -> "HunyuanVideo")
    model_name = model_id.split('/')[-1]
    
    # Construct local path - use the same directory structure as HF
    local_path = os.path.join(LOCAL_MODELS_DIR, model_name)
    if subfolder:
        local_path = os.path.join(local_path, subfolder)
    
    try:
        # Try loading locally first
        print(f"Attempting to load {model_name}{f'/{subfolder}' if subfolder else ''} from local path: {local_path}")
        return model_cls.from_pretrained(local_path, **kwargs)
    except (OSError, ValueError, FileNotFoundError) as e:
        print(f"Could not load model from {local_path}, downloading from HF: {e}")
        return model_cls.from_pretrained(model_id, subfolder=subfolder, **kwargs)

# Try to import sage attention
try:
    from sageattention import sageattn, sageattn_varlen
    print("Successfully imported Sage Attention")
    _has_sage_attn = True
except ImportError:
    # Create dummy functions that do nothing but allow the code to run
    def sageattn(*args, **kwargs):
        return None
    def sageattn_varlen(*args, **kwargs):
        return None
    sageattn.is_available = lambda: False
    sageattn_varlen.is_available = lambda: False
    _has_sage_attn = False
    print("Sage Attention not imported - install with 'pip install sageattention==1.0.6'")

# Create local_models directory if it doesn't exist
LOCAL_MODELS_DIR = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './local_models')))
os.makedirs(LOCAL_MODELS_DIR, exist_ok=True)

# Set HF_HOME to use our local_models directory instead of creating a separate hf_download folder
os.environ['HF_HOME'] = LOCAL_MODELS_DIR

import gradio as gr
import torch
import traceback
import einops
import safetensors.torch as sf
import numpy as np
import argparse
import math
import time

from PIL import Image
from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, state_dict_weighted_merge, state_dict_offset_merge, generate_timestamp
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device, DynamicSwapInstaller, unload_complete_models, load_model_as_complete
from diffusers_helper.thread_utils import AsyncStream, async_run
from diffusers_helper.gradio.progress_bar import make_progress_bar_css, make_progress_bar_html
from diffusers_helper.optimization import configure_teacache, optimize_for_inference, aggressive_memory_cleanup, warmup_model
from diffusers_helper.benchmarking import performance_tracker
from diffusers_helper.optimization import configure_teacache, optimize_for_inference
from transformers import SiglipImageProcessor, SiglipVisionModel
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket


parser = argparse.ArgumentParser()
parser.add_argument('--share', action='store_true')
parser.add_argument("--server", type=str, default='0.0.0.0')
parser.add_argument("--port", type=int, default=7860)
args = parser.parse_args()

print(args)

free_mem_gb = get_cuda_free_memory_gb(gpu)
high_vram = free_mem_gb > 60

print(f'Free VRAM {free_mem_gb} GB')
print(f'High-VRAM Mode: {high_vram}')

# Load models using our function that checks local paths first
text_encoder = load_model_locally_or_download(
    LlamaModel, 
    "hunyuanvideo-community/HunyuanVideo", 
    subfolder='text_encoder', 
    torch_dtype=torch.float16
).cpu()

text_encoder_2 = load_model_locally_or_download(
    CLIPTextModel, 
    "hunyuanvideo-community/HunyuanVideo", 
    subfolder='text_encoder_2', 
    torch_dtype=torch.float16
).cpu()

tokenizer = load_model_locally_or_download(
    LlamaTokenizerFast, 
    "hunyuanvideo-community/HunyuanVideo", 
    subfolder='tokenizer'
)

tokenizer_2 = load_model_locally_or_download(
    CLIPTokenizer, 
    "hunyuanvideo-community/HunyuanVideo", 
    subfolder='tokenizer_2'
)

vae = load_model_locally_or_download(
    AutoencoderKLHunyuanVideo, 
    "hunyuanvideo-community/HunyuanVideo", 
    subfolder='vae', 
    torch_dtype=torch.float16
).cpu()

feature_extractor = load_model_locally_or_download(
    SiglipImageProcessor, 
    "lllyasviel/flux_redux_bfl", 
    subfolder='feature_extractor'
)

image_encoder = load_model_locally_or_download(
    SiglipVisionModel, 
    "lllyasviel/flux_redux_bfl", 
    subfolder='image_encoder', 
    torch_dtype=torch.float16
).cpu()

transformer = load_model_locally_or_download(
    HunyuanVideoTransformer3DModelPacked, 
    'lllyasviel/FramePackI2V_HY', 
    torch_dtype=torch.bfloat16
).cpu()

vae.eval()
text_encoder.eval()
text_encoder_2.eval()
image_encoder.eval()
transformer.eval()

# Enable sage attention for the transformer model
print("Activating Sage Attention...")
try:
    if _has_sage_attn:
        # Method 1: Try to set it directly if using appropriate attention mechanism
        if hasattr(transformer, 'set_use_sage_attention'):
            transformer.set_use_sage_attention(True)
            print("Sage Attention activated successfully via direct method")
        # Method 2: Try to set the attention processor
        elif hasattr(transformer, "set_attn_processor"):
            transformer.set_attn_processor({"sage_attention": {}})
            print("Sage Attention activated via attention processor")
        else:
            print("This model already uses Sage Attention by default when available")
    else:
        print("Sage Attention module detected but will be used in fallback mode")
except Exception as e:
    print(f"Failed to activate Sage Attention: {e}")
    print("Continuing without Sage Attention optimization")

if not high_vram:
    vae.enable_slicing()
    vae.enable_tiling()

transformer.high_quality_fp32_output_for_inference = True
print('transformer.high_quality_fp32_output_for_inference = True')

transformer.to(dtype=torch.bfloat16)
vae.to(dtype=torch.float16)
image_encoder.to(dtype=torch.float16)
text_encoder.to(dtype=torch.float16)
text_encoder_2.to(dtype=torch.float16)

vae.requires_grad_(False)
text_encoder.requires_grad_(False)
text_encoder_2.requires_grad_(False)
image_encoder.requires_grad_(False)
transformer.requires_grad_(False)

if not high_vram:
    # DynamicSwapInstaller is same as huggingface's enable_sequential_offload but 3x faster
    DynamicSwapInstaller.install_model(transformer, device=gpu)
    DynamicSwapInstaller.install_model(text_encoder, device=gpu)
else:
    text_encoder.to(gpu)
    text_encoder_2.to(gpu)
    image_encoder.to(gpu)
    vae.to(gpu)
    transformer.to(gpu)

stream = AsyncStream()

outputs_folder = './outputs/'
os.makedirs(outputs_folder, exist_ok=True)


@torch.no_grad()
def worker(input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache):
    # Ensure all parameters are the right type
    steps = int(steps)
    gpu_memory_preservation = float(gpu_memory_preservation)
    use_teacache = bool(use_teacache)
    # Use a fixed teacache threshold
    thresh_value = 0.15  # Optimal default value for most cases
    # Reset performance tracker at the start of each run
    performance_tracker.reset()
    performance_tracker.start_timer()
    
    total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))

    job_id = generate_timestamp()

    stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Starting ...'))))
    
    # Track initial memory usage
    mem_stats = performance_tracker.track_memory("initial")
    print(f"Initial memory: {mem_stats['current_gb']:.2f}GB used, {mem_stats['free_gb']:.2f}GB free")
    
    # Perform model warmup to optimize GPU performance
    if high_vram:
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Warming up model...'))))
        performance_tracker.start_timer("warmup")
        transformer.to(gpu)
        warmup_model(transformer, device=gpu, dtype=torch.bfloat16)
        if not high_vram:
            transformer.to(cpu)
        aggressive_memory_cleanup()
        warmup_time = performance_tracker.end_timer("warmup")
        print(f"Model warmup completed in {warmup_time:.2f} seconds")

    try:
        # Clean GPU
        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )

        # Text encoding
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Text encoding ...'))))
        
        performance_tracker.start_timer("text_encoding")
        performance_tracker.track_memory("before_text_encoding")

        if not high_vram:
            fake_diffusers_current_device(text_encoder, gpu)  # since we only encode one text - that is one model move and one encode, offload is same time consumption since it is also one load and one encode.
            load_model_as_complete(text_encoder_2, target_device=gpu)

        llama_vec, clip_l_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
        
        text_encoding_time = performance_tracker.end_timer("text_encoding")
        performance_tracker.track_memory("after_text_encoding")
        print(f"Text encoding completed in {text_encoding_time:.2f} seconds")

        if cfg == 1:
            llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
        else:
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

        # Processing input image

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Image processing ...'))))

        H, W, C = input_image.shape
        height, width = find_nearest_bucket(H, W, resolution=640)
        input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)

        Image.fromarray(input_image_np).save(os.path.join(outputs_folder, f'{job_id}.png'))

        input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]

        # VAE encoding

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'VAE encoding ...'))))

        if not high_vram:
            load_model_as_complete(vae, target_device=gpu)

        # Time the VAE encoding to identify performance bottlenecks
        vae_start_time = time.time()
        start_latent = vae_encode(input_image_pt, vae)
        vae_time = time.time() - vae_start_time
        print(f"VAE encoding completed in {vae_time:.2f} seconds")
        
        # Aggressive memory cleanup after VAE encoding
        if not high_vram:
            vae.to(cpu)
            aggressive_memory_cleanup()

        # CLIP Vision

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'CLIP Vision encoding ...'))))

        if not high_vram:
            load_model_as_complete(image_encoder, target_device=gpu)

        # Time the CLIP Vision encoding
        clip_start_time = time.time()
        image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state
        clip_time = time.time() - clip_start_time
        print(f"CLIP Vision encoding completed in {clip_time:.2f} seconds")
        
        # Aggressive memory cleanup after CLIP encoding
        if not high_vram:
            image_encoder.to(cpu)
            aggressive_memory_cleanup()

        # Dtype

        llama_vec = llama_vec.to(transformer.dtype)
        llama_vec_n = llama_vec_n.to(transformer.dtype)
        clip_l_pooler = clip_l_pooler.to(transformer.dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)

        # Sampling

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Start sampling ...'))))

        rnd = torch.Generator("cpu").manual_seed(seed)
        num_frames = latent_window_size * 4 - 3

        history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()
        history_pixels = None
        total_generated_latent_frames = 0

        latent_paddings = reversed(range(total_latent_sections))

        if total_latent_sections > 4:
            # In theory the latent_paddings should follow the above sequence, but it seems that duplicating some
            # items looks better than expanding it when total_latent_sections > 4
            # One can try to remove below trick and just
            # use `latent_paddings = list(reversed(range(total_latent_sections)))` to compare
            latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]

        for latent_padding in latent_paddings:
            is_last_section = latent_padding == 0
            latent_padding_size = latent_padding * latent_window_size

            if stream.input_queue.top() == 'end':
                stream.output_queue.push(('end', None))
                return

            print(f'latent_padding_size = {latent_padding_size}, is_last_section = {is_last_section}')

            indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
            clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

            clean_latents_pre = start_latent.to(history_latents)
            clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
            clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)

            if not high_vram:
                # Clean up memory before loading transformer
                unload_complete_models()
                aggressive_memory_cleanup()
                
                # Load transformer with optimized memory settings
                print(f"Loading transformer with {gpu_memory_preservation}GB memory preservation")
                move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

            # Track memory before sampling
            performance_tracker.track_memory("before_sampling")
            
            # Apply enhanced TeaCache settings with dynamic parameters
            if use_teacache:
                free_mem = get_cuda_free_memory_gb(gpu)
                # Pass additional context to the enhanced TeaCache configuration
                print(f"Configuring TeaCache with {free_mem:.1f}GB available VRAM, {steps} steps, and threshold {thresh_value:.4f}")
                
                # Use the converted threshold value with the updated function
                configure_teacache(transformer, vram_gb=free_mem, steps=steps, rel_l1_thresh=thresh_value)
                
                # Track memory after TeaCache configuration
                if torch.cuda.is_available():
                    current_mem = torch.cuda.memory_allocated() / (1024**3)
                    print(f"Memory after TeaCache configuration: {current_mem:.2f}GB")
            else:
                transformer.initialize_teacache(enable_teacache=False)
                print("TeaCache disabled as per user request")
                
            # Apply additional inference optimizations
            optimize_for_inference(transformer, high_vram=high_vram)
            
            # Start sampling timer
            performance_tracker.start_timer("sampling")

            def callback(d):
                preview = d['denoised']
                preview = vae_decode_fake(preview)

                preview = (preview * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                preview = einops.rearrange(preview, 'b c t h w -> (b h) (t w) c')

                if stream.input_queue.top() == 'end':
                    stream.output_queue.push(('end', None))
                    raise KeyboardInterrupt('User ends the task.')

                current_step = d['i'] + 1
                percentage = int(100.0 * current_step / steps)
                
                # Calculate ETA and track performance
                if not hasattr(callback, 'start_time'):
                    callback.start_time = time.time()
                    callback.times = []
                    callback.last_step_time = callback.start_time
                    callback.cache_hits = 0
                    callback.cache_misses = 0
                    callback.cache_queries = 0
                    eta_str = ""
                else:
                    current_time = time.time()
                    step_time = current_time - callback.last_step_time
                    callback.last_step_time = current_time
                    callback.times.append(current_time)
                    
                    # Track step time in performance tracker
                    performance_tracker.track_step_time(step_time)
                    
                    # Track TeaCache statistics if available in the denoising data
                    if 'cache_info' in d:
                        cache_info = d['cache_info']
                        if 'hits' in cache_info and 'misses' in cache_info:
                            # Track new hits/misses since last step
                            new_hits = cache_info['hits'] - getattr(callback, 'last_hits', 0)
                            new_misses = cache_info['misses'] - getattr(callback, 'last_misses', 0)
                            new_queries = new_hits + new_misses
                            
                            # Update totals
                            callback.cache_hits += new_hits
                            callback.cache_misses += new_misses
                            callback.cache_queries += new_queries
                            
                            # Store current values for next diff
                            callback.last_hits = cache_info['hits']
                            callback.last_misses = cache_info['misses']
                            
                            # Always record in performance tracker even if no new queries
                            # This ensures statistics are tracked for every step
                            performance_tracker.track_cache_stats(new_hits, new_misses, max(1, new_queries))
                            
                            # Print additional debug info about cache state
                            print(f"Step {current_step}/{steps} TeaCache: +{new_hits}/{new_queries} hits, total {callback.cache_hits}/{callback.cache_queries} ({callback.cache_hits/max(1,callback.cache_queries)*100:.1f}%)")
                    
                    if len(callback.times) > 2:
                        # Calculate time per step
                        avg_time = (callback.times[-1] - callback.start_time) / current_step
                        remaining_steps = steps - current_step
                        eta_seconds = avg_time * remaining_steps
                        
                        # Format ETA nicely
                        if eta_seconds < 60:
                            eta_str = f", ETA: {int(eta_seconds)}s"
                        else:
                            eta_str = f", ETA: {int(eta_seconds // 60)}m {int(eta_seconds % 60)}s"
                            
                        # Add frame generation rate
                        fps = current_step / (callback.times[-1] - callback.start_time)
                        eta_str += f", {fps:.2f} steps/sec"
                        
                        # Add cache hit rate if available
                        if hasattr(callback, 'cache_queries') and callback.cache_queries > 0:
                            hit_rate = callback.cache_hits / callback.cache_queries * 100
                            eta_str += f", Cache: {hit_rate:.1f}% hits"
                    else:
                        eta_str = ""
                
                # Show memory usage in the progress
                if torch.cuda.is_available():
                    current_mem = torch.cuda.memory_allocated() / (1024**3)
                    max_mem = torch.cuda.max_memory_allocated() / (1024**3)
                    mem_str = f" (VRAM: {current_mem:.1f}GB, peak: {max_mem:.1f}GB)"
                else:
                    mem_str = ""
                    
                hint = f'Sampling {current_step}/{steps}{eta_str}{mem_str}'
                desc = f'Total generated frames: {int(max(0, total_generated_latent_frames * 4 - 3))}, Video length: {max(0, (total_generated_latent_frames * 4 - 3) / 30) :.2f} seconds (FPS-30). The video is being extended now ...'
                stream.output_queue.push(('progress', (preview, desc, make_progress_bar_html(percentage, hint))))
                return

            generated_latents = sample_hunyuan(
                transformer=transformer,
                sampler='unipc',
                width=width,
                height=height,
                frames=num_frames,
                real_guidance_scale=cfg,
                distilled_guidance_scale=gs,
                guidance_rescale=rs,
                # shift=3.0,
                num_inference_steps=steps,
                generator=rnd,
                prompt_embeds=llama_vec,
                prompt_embeds_mask=llama_attention_mask,
                prompt_poolers=clip_l_pooler,
                negative_prompt_embeds=llama_vec_n,
                negative_prompt_embeds_mask=llama_attention_mask_n,
                negative_prompt_poolers=clip_l_pooler_n,
                device=gpu,
                dtype=torch.bfloat16,
                image_embeddings=image_encoder_last_hidden_state,
                latent_indices=latent_indices,
                clean_latents=clean_latents,
                clean_latent_indices=clean_latent_indices,
                clean_latents_2x=clean_latents_2x,
                clean_latent_2x_indices=clean_latent_2x_indices,
                clean_latents_4x=clean_latents_4x,
                clean_latent_4x_indices=clean_latent_4x_indices,
                callback=callback,
            )

            if is_last_section:
                generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)

            total_generated_latent_frames += int(generated_latents.shape[2])
            history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)

            if not high_vram:
                offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=8)
                load_model_as_complete(vae, target_device=gpu)

            # End sampling timer when reaching decoding stage
            sampling_time = performance_tracker.end_timer("sampling")
            print(f"Sampling completed in {sampling_time:.2f} seconds")
            
            # Capture final TeaCache stats directly from the model for accurate reporting
            if use_teacache and hasattr(transformer, 'cache_hits'):
                # Get the current cache stats from the model
                total_hits = transformer.cache_hits
                total_misses = transformer.cache_misses
                total_queries = transformer.cache_queries
                
                # Only track if there are new queries since last check
                if hasattr(callback, 'last_model_queries'):
                    new_queries = total_queries - callback.last_model_queries
                    new_hits = total_hits - callback.last_model_hits
                    new_misses = total_misses - callback.last_model_misses
                else:
                    new_queries = total_queries
                    new_hits = total_hits
                    new_misses = total_misses
                
                # Store the values for next check
                callback.last_model_queries = total_queries
                callback.last_model_hits = total_hits
                callback.last_model_misses = total_misses
                
                # Only record if we have new queries
                if new_queries > 0:
                    performance_tracker.track_cache_stats(new_hits, new_misses, new_queries)
                    print(f"Final sampling TeaCache stats: +{new_hits}/{new_queries} hits")
            
            real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]

            # Track VAE decoding time
            performance_tracker.start_timer("vae_decode")
            performance_tracker.track_memory("before_vae_decode")
            
            if history_pixels is None:
                history_pixels = vae_decode(real_history_latents, vae).cpu()
            else:
                section_latent_frames = (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
                overlapped_frames = latent_window_size * 4 - 3

                current_pixels = vae_decode(real_history_latents[:, :, :section_latent_frames], vae).cpu()
                history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)
                
            vae_decode_time = performance_tracker.end_timer("vae_decode")
            performance_tracker.track_memory("after_vae_decode")
            print(f"VAE decoding completed in {vae_decode_time:.2f} seconds")

            if not high_vram:
                # More aggressive memory cleanup after section processing
                unload_complete_models()
                aggressive_memory_cleanup()
                
                # Report memory usage
                if torch.cuda.is_available():
                    current_mem = torch.cuda.memory_allocated() / (1024**3)
                    max_mem = torch.cuda.max_memory_allocated() / (1024**3)
                    print(f"Memory after section: current={current_mem:.2f}GB, peak={max_mem:.2f}GB")

            output_filename = os.path.join(outputs_folder, f'{job_id}_{total_generated_latent_frames}.mp4')

            # Track video saving time
            performance_tracker.start_timer("video_save")
            save_bcthw_as_mp4(history_pixels, output_filename, fps=30)
            video_save_time = performance_tracker.end_timer("video_save")
            
            print(f'Decoded. Current latent shape {real_history_latents.shape}; pixel shape {history_pixels.shape}')
            print(f'Video saved in {video_save_time:.2f} seconds')

            stream.output_queue.push(('file', output_filename))

            if is_last_section:
                # Finish the sampling timer if still running
                if hasattr(performance_tracker, "sampling_start"):
                    performance_tracker.end_timer("sampling")
                break
                
    except:
        traceback.print_exc()

        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )

    # Print performance summary at the end
    performance_tracker.print_summary()
    
    # Manually check TeaCache stats on the transformer to ensure they were captured
    if use_teacache and hasattr(transformer, 'cache_hits'):
        print("\n----- TeaCache Model Statistics -----")
        print(f"Direct from model: hits={transformer.cache_hits}, misses={transformer.cache_misses}, queries={transformer.cache_queries}")
        if transformer.cache_queries > 0:
            hit_rate = transformer.cache_hits / transformer.cache_queries * 100
            print(f"Hit rate: {hit_rate:.2f}%")
            # No longer calculating estimated time saved as it's inaccurate
        print("-------------------------------------\n")
    
    stream.output_queue.push(('end', None))
    return


def process(input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache):
    # Use default value for teacache_threshold
    teacache_threshold = 0.15  # Default threshold value
    global stream
    assert input_image is not None, 'No input image!'

    yield None, None, '', '', gr.update(interactive=False), gr.update(interactive=True)

    stream = AsyncStream()

    async_run(worker, input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache)

    output_filename = None

    while True:
        flag, data = stream.output_queue.next()

        if flag == 'file':
            output_filename = data
            yield output_filename, gr.update(), gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=True)

        if flag == 'progress':
            preview, desc, html = data
            yield gr.update(), gr.update(visible=True, value=preview), desc, html, gr.update(interactive=False), gr.update(interactive=True)

        if flag == 'end':
            yield output_filename, gr.update(visible=False), gr.update(), '', gr.update(interactive=True), gr.update(interactive=False)
            break


def end_process():
    stream.input_queue.push('end')


quick_prompts = [
    'The girl dances gracefully, with clear movements, full of charm.',
    'A character doing some simple body movements.',
]
quick_prompts = [[x] for x in quick_prompts]


css = make_progress_bar_css()
block = gr.Blocks(css=css).queue()
with block:
    gr.Markdown('# FramePack')
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(sources='upload', type="numpy", label="Image", height=320)
            prompt = gr.Textbox(label="Prompt", value='')
            example_quick_prompts = gr.Dataset(samples=quick_prompts, label='Quick List', samples_per_page=1000, components=[prompt])
            example_quick_prompts.click(lambda x: x[0], inputs=[example_quick_prompts], outputs=prompt, show_progress=False, queue=False)

            with gr.Row():
                start_button = gr.Button(value="Start Generation")
                end_button = gr.Button(value="End Generation", interactive=False)

            with gr.Group():
                use_teacache = gr.Checkbox(label='Use TeaCache', value=True, info='Accelerates generation by reusing computation across steps. Faster speed, but may slightly affect quality of fine details like hands and fingers.')

                n_prompt = gr.Textbox(label="Negative Prompt", value="", visible=False)  # Not used
                seed = gr.Number(label="Seed", value=31337, precision=0)

                total_second_length = gr.Slider(label="Total Video Length (Seconds)", minimum=1, maximum=120, value=5, step=0.1)
                latent_window_size = gr.Slider(label="Latent Window Size", minimum=1, maximum=33, value=9, step=1, visible=False)  # Should not change
                steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=25, step=1, info='Changing this value is not recommended.')

                cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=1.0, step=0.01, visible=False)  # Should not change
                gs = gr.Slider(label="Distilled CFG Scale", minimum=1.0, maximum=32.0, value=10.0, step=0.01, info='Changing this value is not recommended.')
                rs = gr.Slider(label="CFG Re-Scale", minimum=0.0, maximum=1.0, value=0.0, step=0.01, visible=False)  # Should not change

                gpu_memory_preservation = gr.Slider(label="GPU Inference Preserved Memory (GB) (larger means slower)", minimum=6, maximum=128, value=6, step=0.1, info="Set this number to a larger value if you encounter OOM. Larger value causes slower speed.")

        with gr.Column():
            preview_image = gr.Image(label="Next Latents", height=200, visible=False)
            result_video = gr.Video(label="Finished Frames", autoplay=True, show_share_button=False, height=512, loop=True)
            gr.Markdown('Note that the ending actions will be generated before the starting actions due to the inverted sampling. If the starting action is not in the video, you just need to wait, and it will be generated later.')
            progress_desc = gr.Markdown('', elem_classes='no-generating-animation')
            progress_bar = gr.HTML('', elem_classes='no-generating-animation')
    ips = [input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache]
    # Update worker call to include threshold
    async_run_fn = lambda *args: async_run(worker, *args)
    start_button.click(fn=process, inputs=ips, outputs=[result_video, preview_image, progress_desc, progress_bar, start_button, end_button])
    end_button.click(fn=end_process)


# Launch with standard parameters
block.launch(
    server_name=args.server,
    server_port=args.port,
    share=args.share,
    max_threads=20    # Increase number of worker threads for handling requests
)

"""
Worker module for video generation in FramePack.
"""

import os
import time
import torch
import traceback
import einops
import numpy as np
from PIL import Image

from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import (
    cpu, gpu, offload_model_from_device_for_memory_preservation, 
    unload_complete_models, 
    load_model_as_complete
)
from framepack.utils import prepare_generation_subfolder
from diffusers_helper.hunyuan import (
    encode_prompt_conds, 
    vae_decode, 
    vae_encode, 
    vae_decode_fake
)
from diffusers_helper.utils import (
    save_bcthw_as_mp4, 
    soft_append_bcthw, 
    crop_or_pad_yield_mask,
    resize_and_center_crop,
    generate_timestamp
)
from diffusers_helper.bucket_tools import find_nearest_bucket
from diffusers_helper.optimization import aggressive_memory_cleanup
from diffusers_helper.benchmarking import performance_tracker
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.gradio.progress_bar import make_progress_bar_html

@torch.no_grad()
@torch.no_grad()
def worker(input_image, end_frame, prompt, n_prompt, seed, total_latent_sections, latent_window_size, 
           steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, teacache_thresh, resolution_scale, mp4_crf,
           keep_section_videos, end_frame_strength, movement_scale, section_settings=None, lora_path=None, lora_multiplier=0.8, 
           fp8_optimization=False, models=None, stream=None, outputs_folder='./outputs/'):
    """
    Worker function for generating videos with FramePack.
    
    Args:
        input_image: Input image as numpy array
        end_frame: Optional end frame as numpy array
        prompt: Text prompt for generation
        n_prompt: Negative prompt
        seed: Random seed
        total_latent_sections: Number of sections to generate (instead of video length in seconds)
        latent_window_size: Size of latent window
        steps: Number of denoising steps
        cfg: CFG scale
        gs: Distilled CFG scale
        rs: CFG Re-Scale
        gpu_memory_preservation: Memory to preserve in GB
        use_teacache: Whether to use TeaCache
        teacache_thresh: Threshold for TeaCache
        resolution_scale: Scale factor for resolution
        mp4_crf: MP4 compression quality
        keep_section_videos: Whether to keep intermediate section videos
        end_frame_strength: Strength of end frame influence
        section_settings: Optional settings for individual sections
        lora_path: Path to LoRA file to apply to the model
        lora_multiplier: Multiplier for LoRA weights
        fp8_optimization: Whether to apply FP8 optimization
        models: FramePackModels instance
        stream: AsyncStream for communication
        outputs_folder: Folder to save outputs
        
    Returns:
        Path to the output video file
    """
    # Ensure all parameters are the right type
    steps = int(steps)
    gpu_memory_preservation = float(gpu_memory_preservation)
    use_teacache = bool(use_teacache)
    
    # Process resolution scale
    resolution_scale_factor = 0.5 if resolution_scale == "Half (0.5x)" else 1.0
    print(f"Selected resolution scale: {resolution_scale} (factor: {resolution_scale_factor})")
    
    # Get TeaCache threshold from UI
    thresh_value = float(teacache_thresh)
    
    # Reset performance tracker
    performance_tracker.reset()
    performance_tracker.start_timer()
    
    # Use the directly specified number of latent sections
    total_latent_sections = int(max(total_latent_sections, 1))

    # Generate job ID and create subfolder for this generation
    generation_folder, job_id = prepare_generation_subfolder(outputs_folder, None)
    
    # Log the subfolder creation
    print(f"Creating output subfolder: {generation_folder}")

    stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Starting ...'))))
    
    # Process section settings
    section_map = None
    if section_settings is not None and len(section_settings) > 0:
        section_map = {}
        for row in section_settings:
            if row and len(row) >= 3 and row[0] is not None:
                try:
                    sec_num = int(row[0])
                    img = row[1]
                    prompt_text = row[2] if row[2] is not None else ""
                    
                    # Only add entries that have a valid section number and either an image or prompt
                    if sec_num >= 0 and (img is not None or (prompt_text and prompt_text.strip())):
                        section_map[sec_num] = (img, prompt_text)
                except (ValueError, TypeError) as e:
                    print(f"Error processing section row: {e}")
        
        print(f"Section settings processed: {len(section_map)} sections configured")
        for sec_num, (img, sec_prompt) in section_map.items():
            has_img = img is not None
            has_prompt = sec_prompt is not None and sec_prompt.strip() != ""
            print(f"  Section {sec_num}: Image: {'✓' if has_img else '✗'}, Prompt: {'✓' if has_prompt else '✗'}")
            
        # Print sorted list of section keys for debugging
        print(f"  Section keys in order: {sorted(section_map.keys())}")
    
    # Track initial memory usage
    mem_stats = performance_tracker.track_memory("initial")
    print(f"Initial memory: {mem_stats['current_gb']:.2f}GB used, {mem_stats['free_gb']:.2f}GB free")
    
    # Model warmup for high VRAM systems
    if models.high_vram:
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Warming up model...'))))
        performance_tracker.start_timer("warmup")
        models.transformer.to(gpu)
        if not models.high_vram:
            models.transformer.to(cpu)
        aggressive_memory_cleanup()
        warmup_time = performance_tracker.end_timer("warmup")
        print(f"Model warmup completed in {warmup_time:.2f} seconds")

    try:
        # Clean GPU memory
        if not models.high_vram:
            unload_complete_models(
                models.text_encoder, models.text_encoder_2, 
                models.image_encoder, models.vae, models.transformer
            )

        # Text encoding
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Text encoding ...'))))
        
        performance_tracker.start_timer("text_encoding")
        performance_tracker.track_memory("before_text_encoding")

        if not models.high_vram:
            from diffusers_helper.memory import fake_diffusers_current_device
            fake_diffusers_current_device(models.text_encoder, gpu)
            load_model_as_complete(models.text_encoder_2, target_device=gpu)

        # Encode prompts
        llama_vec, clip_l_pooler = encode_prompt_conds(
            prompt, models.text_encoder, models.text_encoder_2, 
            models.tokenizer, models.tokenizer_2
        )
        
        text_encoding_time = performance_tracker.end_timer("text_encoding")
        performance_tracker.track_memory("after_text_encoding")
        print(f"Text encoding completed in {text_encoding_time:.2f} seconds")

        # Handle negative prompt based on whether it's provided, not based on cfg value
        has_negative_prompt = n_prompt.strip() != ""
                    
        # Encode the negative prompt if provided, otherwise use zeros
        if has_negative_prompt:
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(
                n_prompt, models.text_encoder, models.text_encoder_2, 
                models.tokenizer, models.tokenizer_2
            )
        else:
            llama_vec_n = torch.zeros_like(llama_vec)
            clip_l_pooler_n = torch.zeros_like(clip_l_pooler)

        # Process attention masks
        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

        # Process input image
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Image processing ...'))))

        # Apply resolution scale to bucket selection
        H, W, C = input_image.shape
        
        # Calculate target resolution based on scale factor
        # Use exact values rather than scaling to avoid rounding issues
        if resolution_scale == "Half (0.5x)":
            target_resolution = 320  # Exactly half of 640
        else:
            target_resolution = 640  # Default full resolution
            
        print(f"Input image dimensions: {W}x{H}")
        print(f"Target resolution parameter: {target_resolution}")
        
        # Get bucket dimensions
        height, width = find_nearest_bucket(H, W, resolution=target_resolution)
        print(f"Selected bucket resolution: {width}x{height}")
        
        # Resize the image
        input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)
        print(f"Resized image to: {width}x{height}")

        # Save processed input image
        Image.fromarray(input_image_np).save(os.path.join(generation_folder, f'{job_id}.png'))

        # Convert to PyTorch tensor
        # Convert to PyTorch tensor
        input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]

        # Process end frame if provided
        end_frame_latent = None
        if end_frame is not None:
            print("Processing end frame...")
            # Resize the end frame to match input dimensions
            end_frame_np = resize_and_center_crop(end_frame, target_width=width, target_height=height)
            
            # Save processed end frame image
            Image.fromarray(end_frame_np).save(os.path.join(generation_folder, f'{job_id}_end.png'))
            
            # Convert to PyTorch tensor
            end_frame_pt = torch.from_numpy(end_frame_np).float() / 127.5 - 1
            end_frame_pt = end_frame_pt.permute(2, 0, 1)[None, :, None]

        # VAE encoding
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'VAE encoding ...'))))

        if not models.high_vram:
            load_model_as_complete(models.vae, target_device=gpu)

        # Encode input image with VAE
        vae_start_time = time.time()
        start_latent = vae_encode(input_image_pt, models.vae)
        
        # Encode end frame if provided
        if end_frame is not None:
            end_frame_latent = vae_encode(end_frame_pt, models.vae)
            print("End frame encoded successfully")
        
        # Process section-specific images if provided
        section_latents = None
        if section_map:
            section_latents = {}
            for sec_num, (img, _) in section_map.items():
                if img is not None:
                    try:
                        # Preprocess and encode section image
                        print(f"Processing image for section {sec_num}...")
                        img_np = resize_and_center_crop(img, target_width=width, target_height=height)
                        
                        # Save processed section image
                        Image.fromarray(img_np).save(os.path.join(generation_folder, f'{job_id}_section_{sec_num}.png'))
                        
                        # Convert to PyTorch tensor
                        img_pt = torch.from_numpy(img_np).float() / 127.5 - 1
                        img_pt = img_pt.permute(2, 0, 1)[None, :, None]
                        
                        # Encode with VAE
                        latent = vae_encode(img_pt, models.vae)
                        section_latents[sec_num] = latent
                        print(f"Section {sec_num} image encoded successfully")
                        print(f"  - Latent shape: {latent.shape}")
                        print(f"  - Value range: min={latent.min().item():.4f}, max={latent.max().item():.4f}, mean={latent.mean().item():.4f}")
                    except Exception as e:
                        print(f"Error encoding section {sec_num} image: {e}")
            
            if section_latents:
                print(f"Processed {len(section_latents)}/{len(section_map)} section images")
                print(f"Section latents keys: {sorted(section_latents.keys())}")
            else:
                print("No section images were successfully processed")
            
        vae_time = time.time() - vae_start_time
        print(f"VAE encoding completed in {vae_time:.2f} seconds")
        
        # Memory cleanup after VAE encoding
        if not models.high_vram:
            models.vae.to(cpu)
            aggressive_memory_cleanup()

        # CLIP Vision encoding
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'CLIP Vision encoding ...'))))

        if not models.high_vram:
            load_model_as_complete(models.image_encoder, target_device=gpu)

        # Encode image with CLIP Vision
        clip_start_time = time.time()
        image_encoder_output = hf_clip_vision_encode(
            input_image_np, models.feature_extractor, models.image_encoder
        )
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state
        clip_time = time.time() - clip_start_time
        print(f"CLIP Vision encoding completed in {clip_time:.2f} seconds")
        
        # Memory cleanup after CLIP encoding
        if not models.high_vram:
            models.image_encoder.to(cpu)
            aggressive_memory_cleanup()

        # Convert embeddings to correct dtype
        llama_vec = llama_vec.to(models.transformer.dtype)
        llama_vec_n = llama_vec_n.to(models.transformer.dtype)
        clip_l_pooler = clip_l_pooler.to(models.transformer.dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(models.transformer.dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(models.transformer.dtype)

        # Start sampling
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Start sampling ...'))))

        # Initialize random generator with seed
        rnd = torch.Generator("cpu").manual_seed(seed)
        num_frames = latent_window_size * 4 - 3

        # Initialize history buffers for forward-only sampling
        history_latents = torch.zeros(
            size=(1, 16, 16 + 2 + 1, height // 8, width // 8), 
            dtype=torch.float32
        ).cpu()
        history_pixels = None

        # Add start latent to history for forward-only approach
        history_latents = torch.cat([history_latents, start_latent.to(history_latents)], dim=2)
        total_generated_latent_frames = 1

        # Generate video in progressive sections
        # Define helper function for getting section-specific prompts - adapted for forward-only approach
        def get_section_prompt(section_index, section_map, default_llama_vec, default_clip_pooler, default_mask):
            """Get section-specific prompt encoding if available for forward-only sampling"""
            if not section_map:
                return default_llama_vec, default_clip_pooler, default_mask
                
            # Find the section <= current section. This works the same for forward-only approach
            valid_keys = [k for k in section_map.keys() if k <= section_index]
            if not valid_keys:
                return default_llama_vec, default_clip_pooler, default_mask
                
            # Get the closest previous section
            use_key = max(valid_keys)
            img, section_prompt_text = section_map[use_key]
            
            # Skip if no prompt or empty
            if not section_prompt_text or not section_prompt_text.strip():
                return default_llama_vec, default_clip_pooler, default_mask
                
            # Encode section-specific prompt
            print(f"Using prompt from section {use_key} for section {section_index}")
            print(f"Section prompt: {section_prompt_text[:50]}...")
            
            if not models.high_vram:
                from diffusers_helper.memory import fake_diffusers_current_device
                fake_diffusers_current_device(models.text_encoder, gpu)
                load_model_as_complete(models.text_encoder_2, target_device=gpu)
                
            # Encode the section-specific prompt
            try:
                section_llama_vec, section_clip_pooler = encode_prompt_conds(
                    section_prompt_text, models.text_encoder, models.text_encoder_2, 
                    models.tokenizer, models.tokenizer_2
                )
                
                # Process attention mask
                section_llama_vec, section_mask = crop_or_pad_yield_mask(section_llama_vec, length=512)
                
                # Convert to correct dtype
                section_llama_vec = section_llama_vec.to(models.transformer.dtype)
                section_clip_pooler = section_clip_pooler.to(models.transformer.dtype)
                
                return section_llama_vec, section_clip_pooler, section_mask
            except Exception as e:
                print(f"Error encoding section prompt: {e}")
                return default_llama_vec, default_clip_pooler, default_mask
        
        # Define helper function for getting section-specific latents - adapted for forward-only approach
        def get_section_latent(section_index, section_map, section_latents, default_latent):
            """Get the appropriate latent for the current section in forward-only sampling"""
            if not section_map or not section_latents or len(section_latents) == 0:
                return default_latent
            
            try:
                # Find the nearest section number <= current section index
                valid_keys = [k for k in section_latents.keys() if k <= section_index]
                
                if valid_keys:
                    use_key = max(valid_keys)  # Get the closest previous section
                    print(f"Using image from section {use_key} for section {section_index}")
                    section_latent = section_latents[use_key]
                    print(f"Section latent shape: {section_latent.shape}, min: {section_latent.min().item():.4f}, max: {section_latent.max().item():.4f}")
                    return section_latent
            except Exception as e:
                print(f"Error selecting section latent: {e}")
                
            return default_latent
        
        # Generate video in progressive sections with forward-only approach
        output_filename = None
        for section_index in range(total_latent_sections):
            # Log section information
            print(f"\n== Processing section {section_index}/{total_latent_sections-1} ==")

            # Check for user termination
            if stream.input_queue.top() == 'end':
                print("User requested to end generation - exiting generation loop")
                # Clear the 'end' signal from the queue to allow future generations
                stream.input_queue.pop()
                stream.output_queue.push(('end', None))
                return output_filename

            # If we have an end frame and this is the first section, update it with appropriate strength
            if section_index == 0 and end_frame_latent is not None:
                print("Applying end frame influence for forward-only sampling")
                if end_frame_strength != 1.0:
                    print(f"Applying EndFrame influence at {end_frame_strength:.2f}x strength")
                    modified_end_frame_latent = end_frame_latent * end_frame_strength
                    history_latents[:, :, 0:1, :, :] = modified_end_frame_latent
                else:
                    # Normal processing with full influence
                    history_latents[:, :, 0:1, :, :] = end_frame_latent

            # Prepare indices for forward-only sampling approach
            indices = torch.arange(0, sum([1, 16, 2, 1, latent_window_size])).unsqueeze(0)
            clean_latent_indices_start, clean_latent_4x_indices, clean_latent_2x_indices, clean_latent_1x_indices, latent_indices = indices.split([1, 16, 2, 1, latent_window_size], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_start, clean_latent_1x_indices], dim=1)
            
            # Get clean latents from history
            clean_latents_4x, clean_latents_2x, clean_latents_1x = history_latents[:, :, -sum([16, 2, 1]):, :, :].split([16, 2, 1], dim=2)
            clean_latents = torch.cat([start_latent.to(history_latents), clean_latents_1x], dim=2)

            # Log section processing for forward-only approach
            print(f"\n== Processing section {section_index}/{total_latent_sections-1} ==")
            print(f"  - First section: {section_index == 0}")
            print(f"  - Last section: {section_index == total_latent_sections-1}")
            print(f"  - Use end latent: {section_index == 0 and end_frame is not None}")
            
            # Get section-specific latent if available
            current_latent = get_section_latent(section_index, section_map, section_latents, start_latent)
            print(f"  - Using latent for section {section_index}: shape {current_latent.shape}")
            
            # Get section-specific prompt if available
            current_llama_vec, current_clip_pooler, current_mask = get_section_prompt(
                section_index, section_map, llama_vec, clip_l_pooler, llama_attention_mask
            )
            print(f"  - Using prompt for section {section_index}: shape {current_llama_vec.shape}")

            # Prepare model for inference with custom threshold and LoRA/FP8 if specified
            models.prepare_for_inference(
                gpu_memory_preservation, use_teacache, steps, thresh_value,
                lora_path, lora_multiplier, fp8_optimization
            )
            
            # Track memory before sampling
            performance_tracker.track_memory("before_sampling")
            
            # Start sampling timer
            performance_tracker.start_timer("sampling")

            # Define callback function for sampling updates
            def callback(d):
                preview = d['denoised']
                preview = vae_decode_fake(preview)

                preview = (preview * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                preview = einops.rearrange(preview, 'b c t h w -> (b h) (t w) c')

                # Check for user termination
                if stream.input_queue.top() == 'end':
                    print("User requested to end generation - cancelling in callback")
                    # Clear the 'end' signal from the queue to allow future generations
                    stream.input_queue.pop()
                    stream.output_queue.push(('end', None))
                    # Use a special message that we catch in thread_utils.py
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
                    # Update timing information
                    current_time = time.time()
                    step_time = current_time - callback.last_step_time
                    callback.last_step_time = current_time
                    callback.times.append(current_time)
                    
                    # Track step time
                    performance_tracker.track_step_time(step_time)
                    
                    # Track TeaCache statistics if available
                    if 'cache_info' in d:
                        cache_info = d['cache_info']
                        if 'hits' in cache_info and 'misses' in cache_info:
                            # Calculate new hits/misses
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
                            
                            # Record in performance tracker
                            performance_tracker.track_cache_stats(new_hits, new_misses, max(1, new_queries))
                            
                            # Debug info
                            print(f"Step {current_step}/{steps} TeaCache: +{new_hits}/{new_queries} hits, "
                                 f"total {callback.cache_hits}/{callback.cache_queries} "
                                 f"({callback.cache_hits/max(1,callback.cache_queries)*100:.1f}%)")
                    
                    # Calculate ETA
                    if len(callback.times) > 2:
                        # Time per step
                        avg_time = (callback.times[-1] - callback.start_time) / current_step
                        remaining_steps = steps - current_step
                        eta_seconds = avg_time * remaining_steps
                        
                        # Format ETA
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
                
                # Show memory usage
                if torch.cuda.is_available():
                    current_mem = torch.cuda.memory_allocated() / (1024**3)
                    max_mem = torch.cuda.max_memory_allocated() / (1024**3)
                    mem_str = f" (VRAM: {current_mem:.1f}GB, peak: {max_mem:.1f}GB)"
                else:
                    mem_str = ""
                    
                hint = f'Sampling {current_step}/{steps}{eta_str}{mem_str}'
                desc = f'Total generated frames: {int(max(0, total_generated_latent_frames * 4 - 3))}, Video length: {max(0, (total_generated_latent_frames * 4 - 3) / 24) :.2f} seconds (FPS-24). The video is extending now...'
                stream.output_queue.push(('progress', (preview, desc, make_progress_bar_html(percentage, hint))))

            # Sample the section
            generated_latents = sample_hunyuan(
                transformer=models.transformer,
                sampler='unipc',
                width=width,
                height=height,
                frames=num_frames,
                real_guidance_scale=cfg,
                distilled_guidance_scale=gs,
                guidance_rescale=rs,
                has_negative_prompt=has_negative_prompt,  # Pass the flag
                # shift=3.0,
                num_inference_steps=steps,
                generator=rnd,
                prompt_embeds=current_llama_vec,  # Use section-specific prompt
                prompt_embeds_mask=current_mask,   # Use section-specific mask
                prompt_poolers=current_clip_pooler,  # Use section-specific pooler
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
                movement_scale=movement_scale,
            )

            # Update counters and history for forward-only approach
            total_generated_latent_frames += int(generated_latents.shape[2])
            history_latents = torch.cat([history_latents, generated_latents.to(history_latents)], dim=2)

            # Offload transformer and load VAE for decoding
            if not models.high_vram:
                offload_model_from_device_for_memory_preservation(models.transformer, target_device=gpu, preserved_memory_gb=8)
                load_model_as_complete(models.vae, target_device=gpu)

            # End sampling timer
            sampling_time = performance_tracker.end_timer("sampling")
            print(f"Sampling completed in {sampling_time:.2f} seconds")
            
            # Capture TeaCache stats
            if use_teacache and hasattr(models.transformer, 'cache_hits'):
                # Get cache stats
                total_hits = models.transformer.cache_hits
                total_misses = models.transformer.cache_misses
                total_queries = models.transformer.cache_queries
                
                # Calculate new queries since last check
                if hasattr(callback, 'last_model_queries'):
                    new_queries = total_queries - callback.last_model_queries
                    new_hits = total_hits - callback.last_model_hits
                    new_misses = total_misses - callback.last_model_misses
                else:
                    new_queries = total_queries
                    new_hits = total_hits
                    new_misses = total_misses
                
                # Store values for next check
                callback.last_model_queries = total_queries
                callback.last_model_hits = total_hits
                callback.last_model_misses = total_misses
                
                # Record stats if we have new queries
                if new_queries > 0:
                    performance_tracker.track_cache_stats(new_hits, new_misses, new_queries)
                    print(f"Final sampling TeaCache stats: +{new_hits}/{new_queries} hits")
            
            # Process latents for this section with forward-only approach
            real_history_latents = history_latents[:, :, -total_generated_latent_frames:, :, :]

            # Track VAE decoding time
            performance_tracker.start_timer("vae_decode")
            performance_tracker.track_memory("before_vae_decode")
            
            # Decode latents to pixels with forward-only approach
            if history_pixels is None:
                history_pixels = vae_decode(real_history_latents, models.vae).cpu()
            else:
                section_latent_frames = latent_window_size * 2
                overlapped_frames = latent_window_size * 4 - 3

                current_pixels = vae_decode(real_history_latents[:, :, -section_latent_frames:], models.vae).cpu()
                history_pixels = soft_append_bcthw(history_pixels, current_pixels, overlapped_frames)
                
            vae_decode_time = performance_tracker.end_timer("vae_decode")
            performance_tracker.track_memory("after_vae_decode")
            print(f"VAE decoding completed in {vae_decode_time:.2f} seconds")

            # Cleanup after section processing
            if not models.high_vram:
                unload_complete_models()
                aggressive_memory_cleanup()
                
                # Report memory usage
                if torch.cuda.is_available():
                    current_mem = torch.cuda.memory_allocated() / (1024**3)
                    max_mem = torch.cuda.max_memory_allocated() / (1024**3)
                    print(f"Memory after section: current={current_mem:.2f}GB, peak={max_mem:.2f}GB")

            # Save current progress as video
            output_filename = os.path.join(
                generation_folder, 
                f'{job_id}_{total_generated_latent_frames}.mp4'
            )

            # Track video saving time
            performance_tracker.start_timer("video_save")
            save_bcthw_as_mp4(history_pixels, output_filename, fps=24, crf=mp4_crf)
            video_save_time = performance_tracker.end_timer("video_save")
            

            
            print(f'Decoded. Current latent shape {real_history_latents.shape}; pixel shape {history_pixels.shape}')
            print(f'Video saved in {video_save_time:.2f} seconds')

            # Push file to UI with a small delay to ensure the file is fully written
            print(f"Worker: Pushing file update to UI: {output_filename}")
            time.sleep(0.2)  # Small delay to ensure the file is fully written and available
            stream.output_queue.push(('file', output_filename))

            # Stop if this was the last section
            if section_index == total_latent_sections - 1:
                # Finish the sampling timer if still running
                if hasattr(performance_tracker, "sampling_start"):
                    performance_tracker.end_timer("sampling")
                break
                
    except Exception:
        traceback.print_exc()

        if not models.high_vram:
            unload_complete_models(
                models.text_encoder, models.text_encoder_2, 
                models.image_encoder, models.vae, models.transformer
            )

    # Print performance summary at the end
    performance_tracker.print_summary()
    
    # Check TeaCache stats
    if use_teacache and hasattr(models.transformer, 'cache_hits'):
        print("\n----- TeaCache Model Statistics -----")
        print(f"Direct from model: hits={models.transformer.cache_hits}, "
              f"misses={models.transformer.cache_misses}, "
              f"queries={models.transformer.cache_queries}")
        if models.transformer.cache_queries > 0:
            hit_rate = models.transformer.cache_hits / models.transformer.cache_queries * 100
            print(f"Hit rate: {hit_rate:.2f}%")
        print("-------------------------------------\n")
    
    # Calculate total generation time
    total_generation_time = time.time() - performance_tracker.get_start_time()
    
    # Clean up intermediate videos if not keeping them
    if not keep_section_videos:
        try:
            deleted_count = 0
            # Get the final video name
            final_video_path = output_filename
            final_video_name = os.path.basename(final_video_path)
            
            # Find all videos with the same job_id prefix
            for filename in os.listdir(generation_folder):
                # Keep only the final video, delete any other MP4 files with the same job ID
                if filename.startswith(job_id) and filename.endswith('.mp4') and filename != final_video_name:
                    try:
                        file_path = os.path.join(generation_folder, filename)
                        os.remove(file_path)
                        deleted_count += 1
                    except Exception as e:
                        print(f"Error deleting intermediate file {filename}: {e}")
                        
            if deleted_count > 0:
                print(f"Cleaned up {deleted_count} intermediate video files")
        except Exception as e:
            print(f"Error during cleanup: {e}")
    
    # Send completion message with enhanced information
    final_frame_count = int(max(0, total_generated_latent_frames * 4 - 3))
    final_video_length = max(0, (total_generated_latent_frames * 4 - 3) / 24)
    stream.output_queue.push(('end', (final_frame_count, final_video_length, total_generation_time, total_latent_sections)))
    return output_filename

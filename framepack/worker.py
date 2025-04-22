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
def worker(input_image, end_frame, prompt, n_prompt, seed, total_second_length, latent_window_size, 
           steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, teacache_thresh, resolution_scale, mp4_crf, enable_compile,
           models, stream, outputs_folder='./outputs/'):
    """
    Worker function for generating videos with FramePack.
    
    Args:
        input_image: Input image as numpy array
        prompt: Text prompt for generation
        n_prompt: Negative prompt
        seed: Random seed
        total_second_length: Desired video length in seconds
        latent_window_size: Size of latent window
        steps: Number of denoising steps
        cfg: CFG scale
        gs: Distilled CFG scale
        rs: CFG Re-Scale
        gpu_memory_preservation: Memory to preserve in GB
        use_teacache: Whether to use TeaCache
        mp4_crf: MP4 compression quality
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
    
    # Calculate number of latent sections
    total_latent_sections = (total_second_length * 24) / (latent_window_size * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))

    # Generate job ID for file naming
    job_id = generate_timestamp()

    stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Starting ...'))))
    
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
        Image.fromarray(input_image_np).save(os.path.join(outputs_folder, f'{job_id}.png'))

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
            Image.fromarray(end_frame_np).save(os.path.join(outputs_folder, f'{job_id}_end.png'))
            
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

        # Initialize history buffers
        history_latents = torch.zeros(
            size=(1, 16, 1 + 2 + 16, height // 8, width // 8), 
            dtype=torch.float32
        ).cpu()
        history_pixels = None
        total_generated_latent_frames = 0

        # Calculate latent padding sequence
        latent_paddings = reversed(range(total_latent_sections))
        if total_latent_sections > 4:
            # Use an improved padding sequence for longer videos
            latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]

        # Generate video in progressive sections
        # Generate video in progressive sections
        output_filename = None
        for i_section, latent_padding in enumerate(latent_paddings):
            is_first_section = i_section == 0
            is_last_section = latent_padding == 0
            use_end_latent = is_last_section and end_frame is not None
            latent_padding_size = latent_padding * latent_window_size

            # Check for user termination
            if stream.input_queue.top() == 'end':
                stream.output_queue.push(('end', None))
                return output_filename

            print(f'latent_padding_size = {latent_padding_size}, is_last_section = {is_last_section}, is_first_section = {is_first_section}')
            
            # If we have an end frame and this is the first section, add it to history latents
            if is_first_section and end_frame_latent is not None:
                print("Adding end frame to history latents for first section")
                history_latents[:, :, 0:1, :, :] = end_frame_latent

            # Prepare indices for section generation
            # Prepare indices for section generation
            indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
            clean_latent_indices_pre, _, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

            # Prepare clean latents
            clean_latents_pre = start_latent.to(history_latents)
            clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
            clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)

            # Prepare model for inference with custom threshold and compile setting
            models.prepare_for_inference(gpu_memory_preservation, use_teacache, steps, thresh_value, enable_compile)
            
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
                desc = f'Total generated frames: {int(max(0, total_generated_latent_frames * 4 - 3))}, Video length: {max(0, (total_generated_latent_frames * 4 - 3) / 24) :.2f} seconds (FPS-24). The video is being extended now ...'
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

            # Handle last section differently
            if is_last_section:
                generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)

            # Update counters and history
            total_generated_latent_frames += int(generated_latents.shape[2])
            history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)

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
            
            # Process latents for this section
            real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]

            # Track VAE decoding time
            performance_tracker.start_timer("vae_decode")
            performance_tracker.track_memory("before_vae_decode")
            
            # Decode latents to pixels
            if history_pixels is None:
                history_pixels = vae_decode(real_history_latents, models.vae).cpu()
            else:
                section_latent_frames = (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
                overlapped_frames = latent_window_size * 4 - 3

                current_pixels = vae_decode(
                    real_history_latents[:, :, :section_latent_frames], 
                    models.vae
                ).cpu()
                history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)
                
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
                outputs_folder, 
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
            if is_last_section:
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
    
    # Send completion message with total frames count
    final_frame_count = int(max(0, total_generated_latent_frames * 4 - 3))
    final_video_length = max(0, (total_generated_latent_frames * 4 - 3) / 24)
    stream.output_queue.push(('end', (final_frame_count, final_video_length)))
    return output_filename

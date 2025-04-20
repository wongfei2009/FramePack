"""
UI module for FramePack.
"""

import gradio as gr
from diffusers_helper.thread_utils import async_run
from diffusers_helper.gradio.progress_bar import make_progress_bar_css, make_progress_bar_html

from framepack.worker import worker

def create_ui(models, stream):
    """
    Create the gradio UI for FramePack.
    
    Args:
        models: FramePackModels instance
        stream: AsyncStream for communication
        
    Returns:
        Gradio Blocks instance
    """
    # Define sample prompts
    quick_prompts = [
        'The girl dances gracefully, with clear movements, full of charm.',
        'A character doing some simple body movements.',
    ]
    
    # Format quick prompts for dataset
    quick_prompts = [[x] for x in quick_prompts]
    
    # Create UI with CSS
    css = make_progress_bar_css() + """
    .tab-content {
        padding: 15px 0;
    }
    .action-buttons {
        display: flex;
        gap: 10px;
        margin: 10px 0;
    }
    .compact-slider .gradio-slider {
        margin-bottom: 10px;
    }
    .info-text {
        font-size: 0.9em;
        color: #666;
        margin: 5px 0;
    }
    .full-width-row {
        width: 100%;
    }
    .input-tab, .output-tab, .params-tab {
        padding: 10px;
    }
    .generation-info {
        background-color: rgba(0,0,0,0.03);
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    .seed-group {
        display: flex;
        flex-direction: column;
        gap: 5px;
    }
    .seed-group {
        display: flex;
        flex-direction: column;
        gap: 8px;
    }
    .seed-button {
        margin-top: 0;
        border-radius: 6px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
        width: auto !important;
    }
    .seed-button:hover {
        box-shadow: 0 2px 3px rgba(0,0,0,0.1) !important;
        transform: translateY(-1px) !important;
    }
    """
    
    block = gr.Blocks(css=css).queue()
    
    with block:
        gr.Markdown('# FramePack')
        
        # Hidden parameters (not shown in UI but needed for function calls)
        n_prompt = gr.Textbox(label="Negative Prompt", value="", visible=False)
        cfg = gr.Slider(
            label="CFG Scale", 
            minimum=1.0, maximum=32.0, 
            value=1.0, step=0.01, 
            visible=False
        )
        rs = gr.Slider(
            label="CFG Re-Scale", 
            minimum=0.0, maximum=1.0, 
            value=0.0, step=0.01, 
            visible=False
        )
        # Latent window size is now defined in the Parameters tab
        
        # Main tab interface        # Main tab interface
        with gr.Tabs():
            # Tab 1: Input and Output Combined
            with gr.TabItem("Generation", elem_classes="generation-tab"):
                with gr.Row():
                    # Left column for input
                    with gr.Column(scale=1):
                        # Input image and prompt
                        input_image = gr.Image(sources='upload', type="numpy", label="Input Image", height=320)
                        prompt = gr.Textbox(label="Prompt", value='', lines=4)
                        example_quick_prompts = gr.Dataset(
                            samples=quick_prompts, 
                            label='Quick List', 
                            samples_per_page=1000, 
                            components=[prompt]
                        )
                        example_quick_prompts.click(
                            lambda x: x[0], 
                            inputs=[example_quick_prompts], 
                            outputs=prompt, 
                            show_progress=False, 
                            queue=False
                        )
                        
                        # Basic parameters everyone needs
                        with gr.Row(equal_height=True):
                            with gr.Column(scale=1, elem_classes="seed-group"):
                                seed = gr.Number(label="Seed", value=31337, precision=0)
                                random_seed_btn = gr.Button("🎲 Random Seed", variant="secondary", size="sm", elem_classes="seed-button")
                            
                            with gr.Column(scale=1):
                                total_second_length = gr.Slider(
                                    label="Video Length (Seconds)", 
                                    minimum=1, maximum=120, 
                                    value=5, step=0.1
                                )
                        
                        # Resolution scale selection
                        resolution_scale = gr.Radio(
                            label="Resolution Scale",
                            choices=["Full (1x)", "Half (0.5x)"],
                            value="Full (1x)",
                            info="Half resolution provides much faster generation for previews."
                        )
                        
                        # Action buttons
                        with gr.Row(elem_classes="action-buttons"):
                            start_button = gr.Button(value="Start Generation", variant="primary", size="lg")
                            end_button = gr.Button(value="End Generation", interactive=False, size="lg")
                        
                        # TeaCache removed from here and moved to Parameters tab
                    
                    # Right column for output
                    with gr.Column(scale=1):
                        # Preview image and video output
                        preview_image = gr.Image(label="Next Latents", height=200, visible=False)
                        result_video = gr.Video(
                            label="Generated Video", 
                            autoplay=True, 
                            show_share_button=False, 
                            height=320, 
                            loop=True
                        )
                        
                        # Generation progress information
                        with gr.Group(elem_classes="generation-info"):
                            progress_desc = gr.Markdown('', elem_classes='no-generating-animation')
                            progress_bar = gr.HTML('', elem_classes='no-generating-animation')
                        
                        gr.Markdown(
                            'Note: Ending actions are generated before starting actions due to inverted sampling. If starting action is not visible yet, wait for more frames.',
                            elem_classes="info-text"
                        )
                
            # Tab 2: Parameters
            with gr.TabItem("Parameters", elem_classes="params-tab"):
                with gr.Column(elem_classes="compact-slider"):
                    # Add TeaCache at the top of parameters
                    use_teacache = gr.Checkbox(
                        label='Use TeaCache', 
                        value=True, 
                        info='Accelerates generation by reusing computation across steps. Faster speed, but may slightly affect quality of fine details like hands and fingers.'
                    )
                    
                    teacache_thresh = gr.Slider(
                        label="TeaCache Threshold", 
                        minimum=0.08, maximum=0.25, 
                        value=0.15, step=0.01, 
                        info='Controls cache reuse frequency. Higher values = faster generation but potential quality loss. Lower values = slower but higher quality.'
                    )
                    
                    # All other parameters directly visible
                    steps = gr.Slider(
                        label="Steps", 
                        minimum=1, maximum=100, 
                        value=25, step=1, 
                        info='Changing this value is not recommended.'
                    )
                    gs = gr.Slider(
                        label="Distilled CFG Scale", 
                        minimum=1.0, maximum=32.0, 
                        value=10.0, step=0.01, 
                        info='Changing this value is not recommended.'
                    )
                    gpu_memory_preservation = gr.Slider(
                        label="GPU Inference Preserved Memory (GB)", 
                        minimum=4, maximum=128, 
                        value=6, step=0.1, 
                        info="Set this number to a larger value if you encounter OOM. Larger value causes slower speed."
                    )
                    latent_window_size = gr.Slider(
                        label="Latent Window Size", 
                        minimum=1, maximum=33, 
                        value=9, step=1, 
                        info="Controls frames per section. For 24 FPS: 7=25 frames (≈1 sec), 9=33 frames (≈1.4 sec), 13=49 frames (≈2 sec). Higher values give better temporal coherence, lower values use less VRAM."
                    )
                    
                    enable_compile = gr.Checkbox(
                        label='Enable PyTorch Compile', 
                        value=False, 
                        info='Enables PyTorch 2.0+ compile optimization. Can improve performance but may cause instability on some systems.'
                    )
                    
                    mp4_crf = gr.Slider(
                        label="MP4 Compression", 
                        minimum=0, maximum=100, 
                        value=16, step=1, 
                        info="Lower means better quality. 0 is uncompressed. Change to 16 if you get black outputs."
                    )                    
        
        # Define process function
        def process(input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, 
                    steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, teacache_thresh, resolution_scale, mp4_crf, enable_compile):
            """
            Process the video generation request.
            
            Args:
                Multiple input parameters from UI
                
            Yields:
                Updated UI state
            """
            assert input_image is not None, 'No input image!'

            # Initial UI state - disable start button, enable end button
            yield None, None, '', '', gr.update(interactive=False), gr.update(interactive=True)

            # Start the worker in a separate thread
            async_run(
                worker, 
                input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, 
                steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, teacache_thresh, resolution_scale, mp4_crf, enable_compile,
                models, stream
            )

            output_filename = None

            # Monitor the stream for updates
            while True:
                flag, data = stream.output_queue.next()

                if flag == 'file':
                    output_filename = data
                    yield output_filename, gr.update(), gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=True)

                if flag == 'progress':
                    preview, desc, html = data
                    yield gr.update(), gr.update(visible=True, value=preview), desc, html, gr.update(interactive=False), gr.update(interactive=True)

                if flag == 'end':
                    # Check if we received frame stats
                    if data is not None and isinstance(data, tuple) and len(data) == 2:
                        final_frame_count, final_video_length = data
                        completion_desc = f'✅ Generation completed! Total frames: {final_frame_count}, Video length: {final_video_length:.2f} seconds (FPS-24)'
                    else:
                        completion_desc = '✅ Generation completed!'
                    
                    # Pass completion message to UI with completed progress bar
                    completed_progress_bar = make_progress_bar_html(100, "Generation completed!")
                    yield output_filename, gr.update(visible=False), completion_desc, completed_progress_bar, gr.update(interactive=True), gr.update(interactive=False)
                    break
        
        # Define end process function
        def end_process():
            """End the generation process."""
            stream.input_queue.push('end')
            
        # Connect callbacks
        start_button.click(
            fn=process,
            inputs=[
                input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, 
                steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, teacache_thresh, resolution_scale, mp4_crf, enable_compile
            ],
            outputs=[result_video, preview_image, progress_desc, progress_bar, start_button, end_button]
        )
        
        end_button.click(fn=end_process)
        
        # Connect random seed button
        def generate_random_seed():
            """Generate a random seed value."""
            import random
            import time
            
            # Use current time as part of the seed to ensure uniqueness
            random.seed(time.time())
            return int(random.randint(0, 2147483647))
        
        random_seed_btn.click(
            fn=generate_random_seed,
            inputs=[],
            outputs=[seed],
            show_progress=False
        )
    
    return block

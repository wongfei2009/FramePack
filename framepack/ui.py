"""
UI module for FramePack.
"""

import gradio as gr
from diffusers_helper.thread_utils import AsyncStream, async_run
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
    css = make_progress_bar_css()
    block = gr.Blocks(css=css).queue()
    
    with block:
        gr.Markdown('# FramePack')
        
        with gr.Row():
            with gr.Column():
                # Input components
                input_image = gr.Image(sources='upload', type="numpy", label="Image", height=320)
                prompt = gr.Textbox(label="Prompt", value='')
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

                # Action buttons
                with gr.Row():
                    start_button = gr.Button(value="Start Generation")
                    end_button = gr.Button(value="End Generation", interactive=False)

                # Generation parameters
                with gr.Group():
                    # TeaCache option
                    use_teacache = gr.Checkbox(
                        label='Use TeaCache', 
                        value=True, 
                        info='Accelerates generation by reusing computation across steps. Faster speed, but may slightly affect quality of fine details like hands and fingers.'
                    )

                    # Hidden parameters
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
                    latent_window_size = gr.Slider(
                        label="Latent Window Size", 
                        minimum=1, maximum=33, 
                        value=9, step=1, 
                        visible=False
                    )

                    # Visible parameters
                    seed = gr.Number(label="Seed", value=31337, precision=0)
                    total_second_length = gr.Slider(
                        label="Total Video Length (Seconds)", 
                        minimum=1, maximum=120, 
                        value=5, step=0.1
                    )
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
                        label="GPU Inference Preserved Memory (GB) (larger means slower)", 
                        minimum=4, maximum=128, 
                        value=6, step=0.1, 
                        info="Set this number to a larger value if you encounter OOM. Larger value causes slower speed."
                    )
                    mp4_crf = gr.Slider(
                        label="MP4 Compression", 
                        minimum=0, maximum=100, 
                        value=16, step=1, 
                        info="Lower means better quality. 0 is uncompressed. Change to 16 if you get black outputs."
                    )

            with gr.Column():
                # Output components
                preview_image = gr.Image(label="Next Latents", height=200, visible=False)
                result_video = gr.Video(
                    label="Finished Frames", 
                    autoplay=True, 
                    show_share_button=False, 
                    height=512, 
                    loop=True
                )
                gr.Markdown(
                    'Note that the ending actions will be generated before the starting actions due to the inverted sampling. ' 
                    'If the starting action is not in the video, you just need to wait, and it will be generated later.'
                )
                progress_desc = gr.Markdown('', elem_classes='no-generating-animation')
                progress_bar = gr.HTML('', elem_classes='no-generating-animation')
        
        # Define process function
        def process(input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, 
                    steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf):
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
                steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf, 
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
                        completion_desc = f'✅ Generation completed! Total frames: {final_frame_count}, Video length: {final_video_length:.2f} seconds (FPS-30)'
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
                steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf
            ],
            outputs=[result_video, preview_image, progress_desc, progress_bar, start_button, end_button]
        )
        
        end_button.click(fn=end_process)
    
    return block

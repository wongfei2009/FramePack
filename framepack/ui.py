"""
UI module for FramePack.
"""

import os
import time
import logging
import gradio as gr
from diffusers_helper.thread_utils import async_run
from diffusers_helper.gradio.progress_bar import make_progress_bar_css, make_progress_bar_html

from framepack.worker import worker
from framepack.config.parameter_config import param_config

# Set up logging
logger = logging.getLogger(__name__)

def create_ui(models, stream):
    """
    Create the gradio UI for FramePack.
    
    Args:
        models: FramePackModels instance
        stream: AsyncStream for communication
        
    Returns:
        Gradio Blocks instance
    """
    # Make sure parameters are loaded
    from framepack.config.parameter_config import param_config
    logger.info("Loading saved parameters for UI...")
    
    # Define sample prompts
    quick_prompts = [
        'The girl dances gracefully, with clear movements, full of charm.',
        'A character doing some simple body movements.',
    ]
    
    # Format quick prompts for dataset
    quick_prompts = [[x] for x in quick_prompts]
    
    # Create UI with CSS
    css = make_progress_bar_css() + """
    #generation-panel {
        position: sticky !important;
        bottom: 0 !important;
        background: white !important;
        border-top: 1px solid #ddd !important;
        padding: 10px !important;
        z-index: 100 !important;
        margin-top: 20px !important;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.05) !important;
    }
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
        margin: 6px 0;
        padding-left: 8px;
        border-left: 3px solid #0d6efd;
        line-height: 1.5;
        display: block;
        border-radius: 0;
        background: transparent !important; /* Force transparent background */
    }
    
    /* Specific alignment for latent window info */
    .compact-slider .gr-markdown.info-text {
        margin-top: 0;
        padding-top: 0;
        margin-left: 2px;
    }
    
    /* Override any potential default Gradio styling for info-text containers */
    .info-text-container, 
    .info-text-container > div,
    div:has(> .info-text) {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    
    /* Special styling for the latent window info */
    .latent-window-info {
        margin-top: 0 !important;
        margin-bottom: 15px !important;
        background: transparent !important;
        border-left-color: #0d6efd !important;
        padding-left: 8px !important;
    }
    
    /* Ensure the parent container of the latent window info has no background */
    div:has(> .latent-window-info),
    div:has(> div > .latent-window-info) {
        background: transparent !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    
    /* Style for total sections info */
    .total-sections-info {
        margin-top: 0 !important;
        margin-bottom: 10px !important;
        background: transparent !important;
        border-left-color: #fd7e14 !important; /* Different color to distinguish */
    }
    
    /* Ensure parent containers have no background */
    div:has(> .total-sections-info),
    div:has(> div > .total-sections-info) {
        background: transparent !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    
    /* Remove grey background from all markdown elements in the UI */
    .gr-markdown, 
    .gr-markdown-container,
    div:has(> .gr-markdown) {
        background: transparent !important;
        border-radius: 0 !important;
        box-shadow: none !important;
    }
    
    /* Ensure proper padding for all info-text elements */
    .gr-markdown.info-text {
        padding-left: 8px !important;
        padding-right: 0 !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }
    .full-width-row {
        width: 100%;
    }
    .input-tab, .output-tab, .params-tab, .section-tab {
        padding: 10px;
    }
    
    /* Section tab styling */
    .section-tab .highlighted-keyframe {
        border: 3px solid #ff3860 !important; 
        box-shadow: 0 0 10px rgba(255, 56, 96, 0.5) !important;
    }
    
    .section-tab .section-info-text {
        font-size: 0.9em;
        color: #666;
        margin: 5px 0;
    }
    
    /* New styles for improved section controls */
    .section-box {
        border: 1px solid #e6e6e6;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 15px;
        background-color: #f9f9f9;
    }
    
    .section-header {
        margin: 0 0 10px 0 !important;
        color: #0d6efd;
        font-weight: 600;
    }
    
    .section-separator {
        margin: 10px 0 !important;
        opacity: 0.5;
    }
    
    .section-tabs .tab-nav {
        flex-wrap: wrap !important;
    }
    
    .section-tabs .tab-nav button {
        margin: 2px !important;
    }
    
    .section-number input {
        background-color: #f0f0f0 !important;
        color: #666 !important;
        font-weight: 600 !important;
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
    .reset-button {
        margin: 10px 0 15px 0 !important;
        background-color: #f8f9fa !important;
        border: 1px solid #ddd !important;
        color: #555 !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }
    
    /* Two-column layout styling */
    .params-tab .gr-form {
        display: flex;
        flex-wrap: wrap;
    }
    
    .params-tab .gr-form > .gr-block {
        flex: 1 1 45%;
        margin-right: 15px;
    }
    .reset-button:hover {
        background-color: #e9ecef !important;
        border-color: #ced4da !important;
        color: #212529 !important;
    }
    .save-status {
        margin: 5px 0 15px 0 !important;
        padding: 5px 10px !important;
        color: #198754 !important;
        font-weight: 500 !important;
        line-height: 1.5 !important;
        text-align: right !important;
        font-size: 0.9em !important;
        opacity: 0.9 !important;
        transition: opacity 0.3s ease !important;
    }
    .settings-buttons-container {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        padding: 20px 0 10px 0 !important;
        margin-top: 20px !important;
        border-top: 1px solid #e9ecef !important;
    }
    
    .settings-button {
        margin: 0 10px !important;
        min-width: 150px !important;
        font-weight: 500 !important;
        padding: 10px 20px !important;
        font-size: 0.95em !important;
        transition: all 0.2s ease !important;
    }
    
    .save-button {
        background-color: #0d6efd !important;
        border-color: #0d6efd !important;
        color: white !important;
    }
    
    .save-button:hover {
        background-color: #0b5ed7 !important;
        border-color: #0a58ca !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.2) !important;
    }
    
    .reset-button {
        background-color: #f8f9fa !important;
        border: 1px solid #ced4da !important;
        border-radius: 4px !important;
        color: #495057 !important;
    }
    
    .reset-button:hover {
        background-color: #e9ecef !important;
        border-color: #adb5bd !important;
        color: #212529 !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
    }
    .params-section-title {
        font-size: 0.85em !important;
        color: #6c757d !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        font-weight: 600 !important;
        margin: 15px 0 10px 0 !important;
        padding-bottom: 5px !important;
        border-bottom: none !important; /* Remove the border that's causing double lines */
    }
    
    /* Target the Markdown headings specifically to ensure they don't have bottom borders */
    .params-section-title h4 {
        border-bottom: none !important;
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
    }
    /* Hide the individual reset buttons for each parameter */
    button[aria-label="Reset to default"],
    .gradio-slider button,
    .gradio-checkbox button,
    .gradio-radio button,
    .gradio-textbox button,
    .gradio-number button,
    button.gr-button.gr-button-lg,
    .gr-form button[class*="reset"],
    .refresh-button,
    .reset-value {
        display: none !important;
    }
    
    /* Additional selectors to catch all variations */
    [id*="reset"],
    [class*="reset-button"]:not(.reset-button),
    [data-testid*="reset"] {
        display: none !important;
    }
    """
    
    block = gr.Blocks(css=css, title="FramePack").queue() # Remove stateful=True
    
    with block:
        gr.Markdown('# FramePack')
        
        # Get saved parameters
        params = param_config.get_all_parameters()
        
        # Hidden parameters (not shown in UI but needed for function calls)
        # CFG Re-Scale moved to Settings tab
        # Latent window size is now defined in the Parameters tab
        
        # Main tab interface
        
        with gr.Tabs():
            # Tab 1: Generation with integrated section controls
            with gr.TabItem("Generation", elem_classes="generation-tab"):
                with gr.Row():
                    # Left column for input
                    with gr.Column(scale=1):
                        # Input image and end frame
                        with gr.Row():
                            input_image = gr.Image(sources='upload', type="numpy", label="Input Image", height=320)
                            end_frame = gr.Image(sources='upload', type="numpy", label="Final Frame (Optional)", height=320)
                        
                        # Section Controls right after the input images
                        with gr.Accordion("Section Controls", open=False):
                            # Top explanation and checkbox to enable section controls
                            enable_section_controls = gr.Checkbox(
                                label="Enable Section-Specific Controls", 
                                value=False,
                                info="When enabled, you can specify different images and prompts for different sections of your video."
                            )
                            
                            # Function to calculate full section info text
                            def calc_section_info_text(latent_window_size):
                                frames_per_section = latent_window_size * 4 - 3
                                seconds_per_section = frames_per_section / 24
                                return f"""
                                Define specific images and prompts for different sections of your video:
                                - Each section corresponds to {frames_per_section} frames ({seconds_per_section:.1f} seconds at 24fps)
                                - Section 0 starts at the beginning of the video
                                - If a section has no specific image/prompt, it will use the previous section's settings
                                """
                            
                            # Add a Markdown element with the full text
                            section_info_md = gr.Markdown(
                                calc_section_info_text(params.get("latent_window_size", 9))
                            )
                            
                            # Function to collect section settings
                            def collect_section_settings(*args):
                                # args contains all section inputs: [num1, img1, prompt1, num2, img2, prompt2, ...]
                                result = []
                                for i in range(0, len(args), 3):
                                    if i+2 < len(args):  # Ensure we have complete triplets
                                        num, img, prompt = args[i], args[i+1], args[i+2]
                                        if num is not None:  # Only include sections with numbers
                                            result.append([num, img, prompt])
                                return result
                                
                            # Function to update section settings when inputs change
                            def update_section_settings(enable_controls, *args):
                                if not enable_controls:
                                    return None
                                return collect_section_settings(*args)
                            
                            # Section settings container - shown/hidden based on checkbox
                            section_controls_group = gr.Group(visible=False)
                            with section_controls_group:
                                # Create section settings UI with improved layout
                                section_inputs = []
                                max_sections = 20  # Pre-define 20 sections as requested
                                
                                # Create a tabbed interface for sections
                                with gr.Tabs(elem_classes="section-tabs") as section_tabs:
                                    # Create tabs for sections with 5 sections per tab
                                    for tab_idx in range((max_sections + 4) // 5):  # Ceiling division to get number of tabs needed
                                        start_idx = tab_idx * 5
                                        end_idx = min(start_idx + 5, max_sections)
                                        tab_label = f"Sections {start_idx}-{end_idx - 1}"
                                        
                                        with gr.TabItem(tab_label):
                                            # Create sections for this tab
                                            for i in range(start_idx, end_idx):
                                                with gr.Group(elem_classes="section-box"):
                                                    with gr.Row():
                                                        with gr.Column(scale=1):
                                                            # Add header for each section
                                                            gr.Markdown(f"### Section {i}", elem_classes="section-header")
                                                    
                                                    with gr.Row():
                                                        with gr.Column(scale=1):
                                                            section_number = gr.Number(
                                                                label="Section Number", 
                                                                value=i,  # 0-based index for actual section
                                                                precision=0,
                                                                elem_classes="section-number",
                                                                interactive=False  # Make non-interactive since it's pre-defined
                                                            )
                                                            section_prompt = gr.Textbox(
                                                                label="Section Prompt", 
                                                                placeholder="Section-specific prompt (optional)",
                                                                lines=2,
                                                                elem_classes="section-prompt"
                                                            )
                                                        
                                                        with gr.Column(scale=2):
                                                            section_image = gr.Image(
                                                                label="Section Image", 
                                                                type="numpy", 
                                                                sources="upload",
                                                                elem_classes="section-image"
                                                            )
                                                    
                                                    # Add section inputs to the list
                                                    section_inputs.append([section_number, section_image, section_prompt])
                            
                            # Toggle visibility of section controls based on checkbox
                            # (No need for add/remove functions since all sections are predefined)
                            enable_section_controls.change(
                                fn=lambda x: gr.update(visible=x),
                                inputs=[enable_section_controls],
                                outputs=[section_controls_group]
                            )
                        
                        # Hidden state to store section settings
                        section_settings = gr.State(None)
                        
                        # Prompt inputs after section controls
                        prompt = gr.Textbox(label="Prompt", value=params.get("prompt", ''), lines=4)
                        n_prompt = gr.Textbox(label="Negative Prompt", value=params.get("n_prompt", ""), lines=2, 
                              info="Items to exclude from generation")
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
                                seed = gr.Number(label="Seed", value=params.get("seed", 31337), precision=0)
                                random_seed_btn = gr.Button("🎲 Random Seed", variant="secondary", size="sm", elem_classes="seed-button")
                            
                            with gr.Column(scale=1):
                                # Function to calculate total video length info
                                def calc_total_video_info(latent_window_size, total_sections):
                                    frames_per_section = latent_window_size * 4 - 3
                                    total_frames = total_sections * frames_per_section
                                    total_seconds = total_frames / 24
                                    return f"{total_sections} sections = {total_frames} frames (≈{total_seconds:.1f} seconds at 24fps)"
                                
                                # Create container for the slider and its info
                                with gr.Group():
                                    total_latent_sections = gr.Slider(
                                        label="Number of Sections", 
                                        minimum=1, maximum=20, 
                                        value=params.get("total_latent_sections", 3), step=1,
                                        info="Choose number of sections to generate"
                                    )
                                    
                                    total_sections_info = gr.Markdown(
                                        calc_total_video_info(
                                            params.get("latent_window_size", 9), 
                                            params.get("total_latent_sections", 3)
                                        ),
                                        elem_classes="info-text total-sections-info"
                                    )
                                
                                # Function to update the displayed info
                                def update_total_sections_info(latent_window_size, total_sections):
                                    return calc_total_video_info(latent_window_size, total_sections)
                        
                        # Resolution scale selection
                        resolution_scale = gr.Radio(
                            label="Resolution Scale",
                            choices=["Full (1x)", "Half (0.5x)"],
                            value=params.get("resolution_scale", "Full (1x)"),
                            info="Half resolution provides much faster generation for previews."
                        )
                        
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
                        
                        # Add custom JavaScript to handle autoplay when tab gets focus
                        gr.HTML(
                            """
                            <script>
                            // Function to ensure video autoplay works when tab becomes visible
                            document.addEventListener('visibilitychange', function() {
                                if (document.visibilityState === 'visible') {
                                    // Find all video elements in the Generated Video container
                                    setTimeout(function() {
                                        const videoContainers = document.querySelectorAll('.video-container');
                                        videoContainers.forEach(container => {
                                            const videos = container.querySelectorAll('video');
                                            videos.forEach(video => {
                                                if (video.paused) {
                                                    video.play().catch(e => console.log('Autoplay failed:', e));
                                                }
                                            });
                                        });
                                    }, 100); // Small delay to ensure DOM is ready
                                }
                            });
                            
                            // MutationObserver to detect when new videos are added
                            const observer = new MutationObserver(function(mutations) {
                                mutations.forEach(function(mutation) {
                                    if (mutation.addedNodes.length) {
                                        mutation.addedNodes.forEach(function(node) {
                                            if (node.nodeName === 'VIDEO') {
                                                // Try to play the video when added to DOM
                                                setTimeout(function() {
                                                    if (node.paused) {
                                                        node.play().catch(e => console.log('Autoplay failed:', e));
                                                    }
                                                }, 100);
                                            }
                                        });
                                    }
                                });
                            });
                            
                            // Start observing when the DOM is loaded
                            document.addEventListener('DOMContentLoaded', function() {
                                // Observe the entire document for video additions
                                observer.observe(document.body, { 
                                    childList: true, 
                                    subtree: true 
                                });
                            });
                            </script>
                            """,
                            visible=True
                        )
                        
                        # Generation progress information
                        with gr.Group(elem_classes="generation-info"):
                            progress_desc = gr.Markdown('', elem_classes='no-generating-animation')
                            progress_bar = gr.HTML('', elem_classes='no-generating-animation')
                        
                        gr.Markdown(
                            'Note: Ending actions are generated before starting actions due to inverted sampling. If starting action is not visible yet, wait for more frames.',
                            elem_classes="info-text"
                        )
                
                # Fixed generation panel at bottom
                with gr.Row(elem_id="generation-panel"):                                    
                    with gr.Column(scale=1):
                        # Generation controls
                        with gr.Row():
                            start_button = gr.Button("Generate Video", variant="primary", size="lg")
                            end_button = gr.Button("Stop Generation", interactive=False, size="lg")
                
            # Tab 2: Settings Tab (renamed from Performance)
            with gr.TabItem("Settings", elem_classes="params-tab"):
                with gr.Column(elem_classes="compact-slider"):
                    # Remove the reset container and status message
                    
                    # Two-column layout for settings
                    with gr.Row():
                        # Left column
                        with gr.Column():
                            # Performance settings in left column - use a span instead of markdown heading
                            gr.HTML("<div class='params-section-title'>PERFORMANCE</div>")
                            
                            # TeaCache parameters
                            use_teacache = gr.Checkbox(
                                label='Use TeaCache', 
                                value=params.get("use_teacache", True), 
                                info='Accelerates generation by reusing computation across steps. Faster speed, but may slightly affect quality of fine details like hands and fingers.'
                            )
                            
                            teacache_thresh = gr.Slider(
                                label="TeaCache Threshold", 
                                minimum=0.08, maximum=0.25, 
                                value=params.get("teacache_thresh", 0.15), step=0.01, 
                                info='Controls cache reuse frequency. Higher values = faster generation but potential quality loss. Lower values = slower but higher quality.'
                            )
                            

                            
                            gpu_memory_preservation = gr.Slider(
                                label="GPU Inference Preserved Memory (GB)", 
                                minimum=4, maximum=128, 
                                value=params.get("gpu_memory_preservation", 6), step=0.1, 
                                info="Set this number to a larger value if you encounter OOM. Larger value causes slower speed."
                            )
                            
                            # Output settings in left column - use HTML instead of Markdown
                            gr.HTML("<div class='params-section-title'>OUTPUT</div>")
                            
                            mp4_crf = gr.Slider(
                                label="MP4 Compression", 
                                minimum=0, maximum=100, 
                                value=params.get("mp4_crf", 16), step=1, 
                                info="Lower means better quality. 0 is uncompressed. Change to 16 if you get black outputs."
                            )
                        
                        # Right column
                        with gr.Column():
                            # Generation settings in right column - use HTML instead of Markdown
                            gr.HTML("<div class='params-section-title'>GENERATION</div>")
                            
                            # Generation parameters
                            steps = gr.Slider(
                                label="Steps", 
                                minimum=1, maximum=100, 
                                value=params.get("steps", 25), step=1, 
                                info='Changing this value is not recommended.'
                            )
                            
                            cfg = gr.Slider(
                                label="CFG Scale", 
                                minimum=1.0, maximum=15.0, 
                                value=params.get("cfg", 1.0), step=0.1, 
                                info='Higher values make generation adhere more closely to the prompt.'
                            )
                            
                            gs = gr.Slider(
                                label="Distilled CFG Scale", 
                                minimum=1.0, maximum=32.0, 
                                value=params.get("gs", 10.0), step=0.01, 
                                info='Works together with CFG Scale. Changing this value is typically not recommended.'
                            )
                            
                            rs = gr.Slider(
                                label="CFG Re-Scale", 
                                minimum=0.0, maximum=1.0, 
                                value=params.get("rs", 0.0), step=0.01, 
                                info='Controls the level of CFG re-scaling. Values above 0 can help reduce over-saturation and improve quality at higher CFG values.'
                            )
                            
                            # Calculate frames per section function
                            def calc_frames_per_section(latent_window_size):
                                frames_per_section = latent_window_size * 4 - 3
                                seconds_per_section = frames_per_section / 24
                                return f"Current size: {latent_window_size} = {frames_per_section} frames (≈{seconds_per_section:.1f} sec at 24fps) per section."
                            
                            # First create a State to store the current info text
                            latent_window_info_state = gr.State(calc_frames_per_section(params.get("latent_window_size", 9)))
                            
                            # Create a container for the info text
                            with gr.Group():
                                latent_window_size = gr.Slider(
                                    label="Latent Window Size", 
                                    minimum=1, maximum=33, 
                                    value=params.get("latent_window_size", 9), step=1, 
                                    info="Controls frames per section. Higher values give better temporal coherence but use more VRAM."
                                )
                                
                                latent_window_info = gr.Markdown(
                                    calc_frames_per_section(params.get("latent_window_size", 9)),
                                    elem_classes="info-text latent-window-info"
                                )
                            
                            # Function to update the displayed info
                            def update_latent_window_info(value):
                                info_text = calc_frames_per_section(value)
                                return info_text
                            
                            # This change event is now handled in the connections section
                            
                            end_frame_strength = gr.Slider(
                                label="End Frame Influence", 
                                minimum=0.01, maximum=1.00, 
                                value=params.get("end_frame_strength", 0.50), step=0.01, 
                                info="Controls how strongly the end frame influences the video. Lower values reduce impact."
                            )
                    
                    # Add buttons at the bottom
                    with gr.Row(elem_classes="settings-buttons-container"):
                        with gr.Column():
                            # Container for buttons
                            with gr.Row():
                                # Save button on the left
                                save_button = gr.Button(
                                    value="💾 Save Settings", 
                                    variant="primary",
                                    elem_classes="settings-button save-button"
                                )
                                
                                # Reset button on the right
                                reset_button = gr.Button(
                                    value="↺ Reset to Defaults", 
                                    variant="secondary",
                                    elem_classes="settings-button reset-button"
                                )
                    
                    # Hidden status for saving
                    save_status = gr.Markdown("", visible=False)                    
        
        # Define process function
        def process(input_image, end_frame, prompt, n_prompt, seed, total_latent_sections, latent_window_size, 
                    steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, teacache_thresh, resolution_scale, mp4_crf,
                    end_frame_strength, enable_section_controls, section_settings=None):
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

            # Process section settings if enabled
            processed_section_settings = None
            if enable_section_controls and section_settings is not None:
                processed_section_settings = section_settings
                print(f"Using section-specific settings: {len(processed_section_settings)} sections configured")
            
            # Start the worker in a separate thread
            async_run(
                worker, 
                input_image, end_frame, prompt, n_prompt, seed, total_latent_sections, latent_window_size, 
                steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, teacache_thresh, resolution_scale, mp4_crf,
                end_frame_strength, processed_section_settings,
                models, stream
            )

            output_filename = None

            # Initialize buffer for events
            events_buffer = []
            last_file_event_time = 0
            THROTTLE_VIDEO_UPDATES = 0.5  # Minimum seconds between video updates
            
            # Monitor the stream for updates
            while True:
                flag, data = stream.output_queue.next()
                current_time = time.time()
                
                # Handle for 'end' event - always process immediately
                if flag == 'end':
                    # Check if we received enhanced stats
                    if data is not None and isinstance(data, tuple):
                        if len(data) >= 4:  # Enhanced format with time and sections
                            final_frame_count, final_video_length, total_generation_time, total_sections = data
                            
                            # Format total time nicely
                            if total_generation_time < 60:
                                time_str = f"{total_generation_time:.1f} seconds"
                            elif total_generation_time < 3600:
                                minutes = int(total_generation_time // 60)
                                seconds = int(total_generation_time % 60)
                                time_str = f"{minutes}m {seconds}s"
                            else:
                                hours = int(total_generation_time // 3600)
                                minutes = int((total_generation_time % 3600) // 60)
                                seconds = int(total_generation_time % 60)
                                time_str = f"{hours}h {minutes}m {seconds}s"
                                
                            # Enhanced completion message with more organized display
                            completion_desc = f"""✅ Generation completed!

🎬 **Video Statistics**:
• Frames: {final_frame_count}
• Length: {final_video_length:.2f} seconds (24 FPS)

⏱️ **Generation Details**:
• Time: {time_str}
• Sections: {total_sections}"""
                    else:
                        completion_desc = """✅ Generation completed!

Video generation process has finished successfully."""
                    
                    # Pass completion message to UI with completed progress bar
                    completed_progress_bar = make_progress_bar_html(100, "Generation completed!")
                    yield output_filename, gr.update(visible=False), completion_desc, completed_progress_bar, gr.update(interactive=True), gr.update(interactive=False)
                    break
                
                # Hhandle for 'file' event
                elif flag == 'file':
                    # Always update the latest output filename
                    output_filename = data
                    print(f"UI: Received file update: {output_filename}")
                                            
                    # Sleep a tiny bit to ensure the file is fully written and UI can process it
                    time.sleep(1.0)
                        
                    # Check if file exists and has size > 0
                    if os.path.exists(output_filename) and os.path.getsize(output_filename) > 0:
                        yield output_filename, gr.update(), gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=True)
                    else:
                        print(f"UI: Warning - File not ready yet: {output_filename}")
                        # Don't yield if file isn't ready yet
                    
                # Handle progress events - can be throttled if needed
                elif flag == 'progress':
                    preview, desc, html = data
                    yield gr.update(), gr.update(visible=True, value=preview), desc, html, gr.update(interactive=False), gr.update(interactive=True)
        
        # Define end process function
        def end_process():
            """End the generation process."""
            stream.input_queue.push('end')
            
        # Connect callbacks

        # Connect section inputs to settings updates
        if 'section_inputs' in locals() and 'section_settings' in locals():
            section_input_list = []
            for inputs in section_inputs:
                section_input_list.extend(inputs)
            
            # Update section settings when any input changes
            for inp in section_input_list:
                inp.change(
                    fn=update_section_settings,
                    inputs=[enable_section_controls] + section_input_list,
                    outputs=[section_settings]
                )
        
        # Connect start button with all inputs
        start_button.click(
            fn=process,
            inputs=[
                input_image, end_frame, prompt, n_prompt, seed, total_latent_sections, latent_window_size, 
                steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, teacache_thresh, resolution_scale, mp4_crf,
                end_frame_strength, enable_section_controls, section_settings
            ],
            outputs=[result_video, preview_image, progress_desc, progress_bar, start_button, end_button]
        )
        
        end_button.click(fn=end_process)
        

        
        # Utility functions for parameter management
        # Define clear_status function first so it's available for all events
        def clear_status():
            """Clear the status message after a delay."""
            return ""
            
        # Function to generate a random seed
        def generate_random_seed():
            """Generate a random seed value."""
            import random
            import time
            
            # Use current time as part of the seed to ensure uniqueness
            random.seed(time.time())
            random_seed_val = int(random.randint(0, 2147483647))
            
            # Save the generated seed
            param_config.update_parameter("seed", random_seed_val)
            
            return random_seed_val
        
        # Function to save all parameters at once
        def save_all_parameters(seed_val, total_latent_sections_val, resolution_scale_val, 
                               use_teacache_val, teacache_thresh_val, steps_val, gs_val,
                               gpu_memory_val, latent_window_val, mp4_crf_val,
                               prompt_val, n_prompt_val, cfg_val, rs_val, end_frame_strength_val):
            """Save all current parameter values."""
            params_to_save = {
                "seed": seed_val,
                "total_latent_sections": total_latent_sections_val,
                "resolution_scale": resolution_scale_val,
                "use_teacache": use_teacache_val,
                "teacache_thresh": teacache_thresh_val,
                "steps": steps_val,
                "gs": gs_val,
                "gpu_memory_preservation": gpu_memory_val,
                "latent_window_size": latent_window_val,
                "mp4_crf": mp4_crf_val,
                "prompt": prompt_val,
                "n_prompt": n_prompt_val,
                "cfg": cfg_val,
                "rs": rs_val,
                "end_frame_strength": end_frame_strength_val
            }
            
            # Log the parameters we're saving
            logger.info(f"Saving parameters: {params_to_save}")
            
            # Update the parameters in the config
            param_config.update_parameters(params_to_save)
            
            # Log after saving
            logger.info("Parameters saved successfully")
            
            return gr.update(value="✓ Settings saved successfully", visible=True)
        
        # Function to reset all parameters to defaults
        def reset_with_message():
            """Reset parameters and show confirmation message."""
            defaults = param_config.reset_parameters()
            
            # Return values for UI components with status message
            return {
                seed: defaults["seed"],
                total_latent_sections: defaults["total_latent_sections"],
                resolution_scale: defaults["resolution_scale"],
                use_teacache: defaults["use_teacache"],
                teacache_thresh: defaults["teacache_thresh"],
                steps: defaults["steps"],
                gs: defaults["gs"],
                gpu_memory_preservation: defaults["gpu_memory_preservation"],
                latent_window_size: defaults["latent_window_size"],
                mp4_crf: defaults["mp4_crf"],
                prompt: defaults["prompt"],
                n_prompt: defaults["n_prompt"],
                cfg: defaults["cfg"],
                rs: defaults["rs"],
                end_frame_strength: defaults["end_frame_strength"],
                save_status: gr.update(value="✓ Parameters restored to default values", visible=True)
            }
            
        # Connect random seed button
        random_seed_btn.click(
            fn=generate_random_seed,
            inputs=[],
            outputs=[seed],
            show_progress=False
        )
        
        # Connect sliders to update the dynamic info displays
        if 'section_info_md' in locals():
            latent_window_size.change(
                fn=calc_section_info_text,
                inputs=[latent_window_size],
                outputs=[section_info_md],
                show_progress=False
            )
        
        # Update latent window size info when the slider changes
        latent_window_size.change(
            fn=update_latent_window_info,
            inputs=[latent_window_size],
            outputs=[latent_window_info],
            show_progress=False
        )
        
        # Update total sections info when either slider changes
        latent_window_size.change(
            fn=update_total_sections_info,
            inputs=[latent_window_size, total_latent_sections],
            outputs=[total_sections_info],
            show_progress=False
        )
        
        total_latent_sections.change(
            fn=update_total_sections_info,
            inputs=[latent_window_size, total_latent_sections],
            outputs=[total_sections_info],
            show_progress=False
        )
        
        # Connect reset button
        reset_button.click(
            fn=reset_with_message,
            inputs=[],
            outputs=[
                seed, total_latent_sections, resolution_scale,
                use_teacache, teacache_thresh, steps, gs,
                gpu_memory_preservation, latent_window_size,
                mp4_crf, prompt, n_prompt,
                cfg, rs, end_frame_strength, save_status
            ],
            show_progress=False
        )
        
        # Add a separate event to clear the status after reset
        # This uses a simpler approach that doesn't rely on .then()
        def delayed_clear():
            import time
            time.sleep(3)  # Wait 3 seconds
            return ""
            
        reset_button.click(
            fn=delayed_clear,
            inputs=[],
            outputs=[save_status],
            show_progress=False,
            queue=False  # Run in parallel
        )
        
        # Function to clear status message after delay
        def clear_status_after_delay():
            """Clear the save status message after a delay."""
            import time
            time.sleep(3)  # Wait for 3 seconds
            return gr.update(visible=False)
        
        # Connect Save button with all the parameter inputs
        save_button.click(
            fn=save_all_parameters,
            inputs=[
                seed, total_latent_sections, resolution_scale,
                use_teacache, teacache_thresh, steps, gs,
                gpu_memory_preservation, latent_window_size,
                mp4_crf, prompt, n_prompt,
                cfg, rs, end_frame_strength
            ],
            outputs=[save_status]
        )
        
        # Add auto-hide for save status
        save_button.click(
            fn=clear_status_after_delay,
            inputs=[],
            outputs=[save_status],
            queue=False,  # Run in parallel
            show_progress=False
        )
        
        # Connect reset button with the same behavior as before
        reset_button.click(
            fn=reset_with_message,
            inputs=[],
            outputs=[
                seed, total_latent_sections, resolution_scale,
                use_teacache, teacache_thresh, steps, gs,
                gpu_memory_preservation, latent_window_size,
                mp4_crf, prompt, n_prompt,
                cfg, rs, end_frame_strength, save_status
            ]
        )
        
        # Add auto-hide for reset status
        reset_button.click(
            fn=clear_status_after_delay,
            inputs=[],
            outputs=[save_status],
            queue=False,  # Run in parallel
            show_progress=False
        )
        
        # Connect random seed button
        random_seed_btn.click(
            fn=generate_random_seed,
            inputs=[],
            outputs=[seed],
            show_progress=False
        )

        # --- Add the .load() event handler ---
        # List ALL components whose values should be loaded from parameters.json on refresh
        components_to_load = [
            seed, total_latent_sections, resolution_scale, use_teacache,
            teacache_thresh, steps, gs, gpu_memory_preservation,
            latent_window_size, mp4_crf, prompt,
            n_prompt, cfg, rs, end_frame_strength
        ]

        def load_saved_params():
            """Loads parameters from config and returns them in the correct order."""
            logger.info("Reloading parameters for UI on page load...")
            params = param_config.get_all_parameters()
            # Return values in the *exact same order* as components_to_load list
            return [
                params.get("seed", 31337),
                params.get("total_latent_sections", 3),
                params.get("resolution_scale", "Full (1x)"),
                params.get("use_teacache", True),
                params.get("teacache_thresh", 0.15),
                params.get("steps", 25),
                params.get("gs", 10.0),
                params.get("gpu_memory_preservation", 6),
                params.get("latent_window_size", 9),
                params.get("mp4_crf", 16),
                params.get("prompt", ''),
                params.get("n_prompt", ''),
                params.get("cfg", 1.0),
                params.get("rs", 0.0),
                params.get("end_frame_strength", 1.0)
            ]

        block.load(
            fn=load_saved_params,
            inputs=[],
            outputs=components_to_load,
            show_progress=False # Optional: hide progress indicator for this load
        )
        # --- End of .load() event handler ---
        
        # No need for section status updates anymore

    return block

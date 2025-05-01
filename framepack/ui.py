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
    
    # Helper functions for time formatting
    def format_time(seconds):
        """Format seconds into a compact time string"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m{secs:.1f}s"
    
    # Function to calculate section timing information
    def calc_section_timing(section_index, latent_window_size):
        """Calculate the time range for a specific section"""
        frames_per_section = latent_window_size * 4 - 3
        start_time = (section_index * frames_per_section) / 24  # In seconds
        end_time = ((section_index + 1) * frames_per_section - 1) / 24  # In seconds
        return f"{format_time(start_time)} - {format_time(end_time)}"
    
    # Define sample prompts
    quick_prompts = [
        'The character dances gracefully, with clear movements, full of charm.',
        'The character does some simple body movements.',
        'The character walks gracefully, with clear movements, across the room.',
        'The character breathes calmly, with subtle body movements.',
		'The character walks forward, gestures with hands, with natural posture.',	
		'The character performs dynamic movements with energy and flowing motion.',	
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
        padding: 20px 0;
    }
    
    /* Main tabs styling */
    .gradio-tabs {
        margin-top: 10px !important;
    }
    
    .gradio-tabs .tab-nav {
        background-color: #f8fafc !important;
        border-radius: 8px 8px 0 0 !important;
        padding: 5px 5px 0 5px !important;
        border-bottom: 1px solid #e2e8f0 !important;
    }
    
    .gradio-tabs .tab-nav button {
        font-weight: 500 !important;
        border-radius: 8px 8px 0 0 !important;
        padding: 10px 20px !important;
        margin: 0 5px !important;
        transition: all 0.2s ease !important;
    }
    
    .gradio-tabs .tab-nav button.selected {
        background-color: white !important;
        border-bottom: 3px solid #3b82f6 !important;
        color: #1e40af !important;
        font-weight: 600 !important;
    }
    
    .gradio-tabs .tab-nav button:not(.selected):hover {
        background-color: #f1f5f9 !important;
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
    
    /* Custom styling for the section image upload area */
    .section-image .drop-region {
        border: 2px dashed #cbd5e1 !important;
        border-radius: 8px !important;
        background-color: #f8fafc !important;
        transition: all 0.2s ease-in-out !important;
    }
    
    .section-image .drop-region:hover {
        border-color: #93c5fd !important;
        background-color: #f1f5f9 !important;
    }
    
    .section-image .upload-button {
        background-color: #3b82f6 !important;
        color: white !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }
    
    .section-image .upload-button:hover {
        background-color: #2563eb !important;
        transform: translateY(-1px) !important;
    }
    
    /* Section info guide styling */
    .section-info-guide {
        background-color: #f0f9ff !important;
        border: 1px solid #bae6fd !important;
        border-radius: 8px !important;
        padding: 15px !important;
        margin: 0 0 20px 0 !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
        position: relative !important;
        z-index: 1 !important;
    }
    
    /* Fix for the double rectangle issue - ensure all child elements are transparent */
    .section-info-guide > div,
    .section-info-guide > p,
    .section-info-guide .gr-markdown,
    .section-info-guide .gr-markdown-container {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    
    /* Fix any container elements */
    div:has(> .section-info-guide),
    div:has(> div > .section-info-guide) {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
    }
    
    .section-info-guide h3 {
        color: #0369a1 !important;
        font-size: 1.1em !important;
        margin-top: 0 !important;
        margin-bottom: 10px !important;
    }
    
    .section-info-guide p {
        margin: 0 0 10px 0 !important;
    }
    
    .section-info-guide ul, .section-info-guide ol {
        padding-left: 20px !important;
        margin: 10px 0 !important;
    }
    
    .section-info-guide strong {
        color: #0369a1 !important;
        font-weight: 600 !important;
    }
    
    /* Section Controls Accordion Styling */
    .section-controls-accordion > .label-wrap {
        background-color: #f8fafc !important;
        border-radius: 8px !important;
        padding: 12px 15px !important;
        margin-bottom: 15px !important;
        border: 1px solid #e2e8f0 !important;
        transition: all 0.2s ease !important;
    }
    
    .section-controls-accordion > .label-wrap:hover {
        background-color: #f1f5f9 !important;
        border-color: #cbd5e1 !important;
    }
    
    .section-controls-accordion > .label-wrap > .icon-button {
        color: #3b82f6 !important;
        background-color: rgba(59, 130, 246, 0.1) !important;
        border-radius: 6px !important;
        padding: 5px !important;
    }
    
    .section-controls-accordion > .label-wrap > .label {
        font-weight: 600 !important;
        font-size: 1.05em !important;
        color: #1e40af !important;
    }
    
    /* Checkbox styling */
    .section-controls-accordion input[type="checkbox"] {
        transform: scale(1.1) !important;
        accent-color: #3b82f6 !important;
    }
    
    .section-controls-accordion label.block span {
        font-weight: 500 !important;
        color: #1f2937 !important;
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
    
    /* Fix for double rectangle issue in section timing */
    .section-timing-info {
        position: relative !important;
        z-index: 1 !important;
    }
    
    /* Target any potential nested elements within timing info */
    .section-timing-info > div,
    .section-timing-info > p,
    .section-timing-info .gr-markdown,
    div:has(> .section-timing-info) .gr-markdown-container {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* Ensure no nested backgrounds in markdown containers */
    .gr-markdown-container .gr-markdown {
        background: transparent !important;
        border: none !important;
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
    
    /* Additional styling for time range - ensure no double rectangles */
    .section-timing-info,
    div.gr-markdown.section-timing-info,
    div.section-timing-info > .gr-markdown,
    div:has(> .section-timing-info) > .gr-markdown,
    .gr-markdown.section-timing-info,
    .gr-markdown-container.section-timing-info,
    div:has(> .section-timing-info) .gr-markdown-container {
        margin: 0 0 8px 0 !important;
        padding: 4px 8px !important;
        font-size: 0.85em !important;
        color: #4b5563 !important;
        background-color: #f0f4f8 !important;
        border-radius: 4px !important;
        display: inline-block !important;
        font-weight: 500 !important;
        border: none !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
        position: relative !important;
        z-index: 1 !important;
        line-height: 1.4 !important;
        width: auto !important;
        overflow: visible !important;
    }
    
    /* Fix any container issues */
    div:has(> .section-timing-info),
    div.section-timing-info,
    div:has(> div > .section-timing-info) {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        margin-bottom: 8px !important;
        box-shadow: none !important;
        width: auto !important;
        display: block !important;
    }
    
    /* Improved styles for section controls */
    .section-box {
        border: 1px solid #eaeaea;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        background-color: #f9fafe;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        transition: box-shadow 0.3s ease-in-out, transform 0.2s ease;
    }
    
    .section-box:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
        transform: translateY(-2px);
    }
    
    .section-header-row {
        align-items: center !important;
        margin-bottom: 10px !important;
    }
    
    .section-number {
        background-color: #3b82f6;
        color: white;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 1.2em;
        box-shadow: 0 2px 5px rgba(59, 130, 246, 0.3);
    }
    
    .section-header {
        margin: 0 !important;
        padding: 0 !important;
        color: #0d6efd;
        font-weight: 600;
        font-size: 1.1em !important;
    }
    
    .section-header h3 {
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* Section prompt styling */
    .section-prompt textarea {
        min-height: 100px !important;
        font-size: 0.95em !important;
        line-height: 1.5 !important;
        border-color: #d1d5db !important;
        border-radius: 8px !important;
        resize: vertical !important;
    }
    
    .section-prompt textarea:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.3) !important;
    }
    
    /* Add some spacing between elements */
    .section-box > .gr-form > .gr-block, 
    .section-box > .gr-block {
        margin-top: 5px !important;
        margin-bottom: 5px !important;
    }
    
    .section-separator {
        margin: 10px 0 !important;
        opacity: 0.5;
    }
    
    /* Improved section tab navigation */
    .section-tabs .tab-nav {
        flex-wrap: wrap !important;
        background-color: #f5f7fb !important;
        border-radius: 8px !important;
        padding: 6px !important;
        border: 1px solid #e2e8f0 !important;
        margin-bottom: 15px !important;
    }
    
    .section-tabs .tab-nav button {
        margin: 3px !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
        padding: 8px 16px !important;
        transition: all 0.2s ease !important;
    }
    
    .section-tabs .tab-nav button.selected {
        background-color: #3b82f6 !important;
        color: white !important;
        box-shadow: 0 2px 5px rgba(59, 130, 246, 0.3) !important;
    }
    
    .section-tabs .tab-nav button:not(.selected):hover {
        background-color: #e2e8f0 !important;
    }
    
    .section-timing-info {
        margin: 0 0 8px 0 !important;
        padding: 4px 8px !important;
        font-size: 0.85em !important;
        color: #4b5563 !important;
        background-color: #f0f4f8 !important;
        border-radius: 4px !important;
        display: inline-block !important;
        font-weight: 500 !important;
        border: none !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
    }
    .generation-info {
        background-color: none;
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
                        
                        # Section Controls right after the input images with better styling
                        with gr.Accordion("Section Controls", open=False, elem_classes="section-controls-accordion"):
                            # Top explanation and checkbox to enable section controls
                            enable_section_controls = gr.Checkbox(
                                label="Enable", 
                                value=False,
                                info="When enabled, you can specify different images and prompts for different sections of your video."
                            )
                            
                            # Function to calculate full section info text with more details
                            def calc_section_info_text(latent_window_size):
                                frames_per_section = latent_window_size * 4 - 3
                                seconds_per_section = frames_per_section / 24
                                return f"""
                                <h3>Section Controls Guide</h3>
                                
                                <p>Each section represents approximately <strong>{frames_per_section} frames</strong> (≈{seconds_per_section:.2f} seconds at 24fps) of your generated video.</p>
                                
                                <p><strong>Key points:</strong></p>
                                <ul>
                                    <li>Customize each section with its own prompt and/or reference image</li>
                                    <li>Empty sections will use settings from the previous section</li>
                                    <li>First section uses the main prompt/image if not specified</li>
                                </ul>
                                """
                            
                            # Add a Markdown element with improved styling, using HTML to avoid double rectangle issue
                            section_info_md = gr.HTML(
                                f'<div class="section-info-guide">{calc_section_info_text(params.get("latent_window_size", 9))}</div>'
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
                                
                                section_data = collect_section_settings(*args)
                                
                                # Debug output to help track section settings
                                valid_sections = [s for s in section_data if s[0] is not None and (s[1] is not None or (s[2] is not None and s[2].strip()))]
                                if valid_sections:
                                    print(f"Section settings updated: {len(valid_sections)} valid sections")
                                    for sec in valid_sections:
                                        sec_num, img, prompt = sec
                                        has_img = img is not None
                                        has_prompt = prompt is not None and prompt.strip() != ""
                                        print(f"  Updated section {sec_num}: Image: {'✓' if has_img else '✗'}, Prompt: {'✓' if has_prompt else '✗'}")
                                else:
                                    print("No valid section settings after update")
                                    
                                return section_data
                            
                            # Section settings container - shown/hidden based on checkbox
                            section_controls_group = gr.Group(visible=False)
                            with section_controls_group:
                                # Create section settings UI with improved layout
                                section_inputs = []
                                max_sections = 20  # Pre-define 20 sections as requested
                                
                                # Create a tabbed interface for sections with better styling
                                with gr.Tabs(elem_classes="section-tabs") as section_tabs:
                                    # Create tabs for sections with 5 sections per tab
                                    for tab_idx in range((max_sections + 4) // 5):  # Ceiling division to get number of tabs needed
                                        start_idx = tab_idx * 5
                                        end_idx = min(start_idx + 5, max_sections)
                                        tab_label = f"Sections {start_idx + 1}-{end_idx}"
                                        
                                        with gr.TabItem(tab_label):
                                            # Create sections for this tab
                                            for i in range(start_idx, end_idx):
                                                with gr.Group(elem_classes="section-box"):                                                                                                        
                                                    # Time range as regular text at the top, using HTML instead of Markdown
                                                    current_latent_size = params.get("latent_window_size", 9)
                                                    timing_info = gr.HTML(
                                                        f'<div class="section-timing-info">{calc_section_timing(i, current_latent_size)}</div>'
                                                    )
                                                    
                                                    # Store timing_info references to update later
                                                    if 'section_timing_displays' not in locals():
                                                        section_timing_displays = []
                                                    section_timing_displays.append((i, timing_info))
                                                    
                                                    # Hidden field to store section number
                                                    section_number = gr.Number(
                                                        value=i,
                                                        visible=False
                                                    )
                                                
                                                    # Main content row with adjusted scales for better usability
                                                    with gr.Row(equal_height=True):
                                                        # Give more space to the prompt (scale 6) vs the image (scale 4)
                                                        with gr.Column(scale=6):
                                                            section_prompt = gr.Textbox(
                                                                label="Section Prompt", 
                                                                placeholder="Enter section-specific prompt here (optional).",
                                                                lines=6,  # Increased lines for better usability
                                                                max_lines=12,  # Allow expansion but limit it
                                                                elem_classes="section-prompt"
                                                            )
                                                        
                                                        with gr.Column(scale=4):
                                                            section_image = gr.Image(
                                                                label="Section Image", 
                                                                type="numpy", 
                                                                sources="upload",
                                                                elem_classes="section-image",
                                                                height=180,
                                                                interactive=True
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
                        # Debug State value for section settings - initialize with empty list
                        section_settings = gr.State([])
                        
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
                                    
                                    total_sections_info = gr.HTML(
                                        f'<div class="info-text total-sections-info">{calc_total_video_info(params.get("latent_window_size", 9), params.get("total_latent_sections", 3))}</div>'
                                    )
                                
                                # Function to update the displayed info
                                def update_total_sections_info(latent_window_size, total_sections):
                                    return f'<div class="info-text total-sections-info">{calc_total_video_info(latent_window_size, total_sections)}</div>'
                        
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
                            
                            fp8_optimization = gr.Checkbox(
                                label="FP8 Optimization",
                                value=params.get("fp8_optimization", False),
                                info="Quantize transformer weights to FP8, reducing GPU memory usage. May improve performance but could affect quality."
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
                            
                            # Option to keep section videos
                            keep_section_videos = gr.Checkbox(
                                label="Keep intermediate section videos",
                                value=False,
                                info="When checked, intermediate section videos will be kept. Otherwise, only the final video is saved."
                            )
                            
                            # No Model Customization section needed anymore
                        
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
                                info='Controls prompt adherence strength; amplifies pos/neg difference. Interacts with Distilled CFG.'
                            )
                            
                            gs = gr.Slider(
                                label="Distilled CFG Scale", 
                                minimum=1.0, maximum=32.0, 
                                value=params.get("gs", 10.0), step=0.01, 
                                info='Internal model conditioning guidance factor (default 10.0); significant changes not recommended.'
                            )
                            
                            rs = gr.Slider(
                                label="CFG Re-Scale", 
                                minimum=0.0, maximum=1.0, 
                                value=params.get("rs", 0.0), step=0.01, 
                                info='Rescales final guidance towards positive prediction to prevent artifacts (useful with high CFG); 0.0 disables.'
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
                                
                                latent_window_info = gr.HTML(
                                    f'<div class="info-text latent-window-info">{calc_frames_per_section(params.get("latent_window_size", 9))}</div>'
                                )
                            
                            # Function to update the displayed info
                            def update_latent_window_info(value):
                                info_text = calc_frames_per_section(value)
                                return f'<div class="info-text latent-window-info">{info_text}</div>'
                                                        
                            end_frame_strength = gr.Slider(
                                label="End Frame Influence", 
                                minimum=0.01, maximum=1.00, 
                                value=params.get("end_frame_strength", 0.50), step=0.01,
                                info="Controls how strongly the end frame influences the video. Lower values reduce impact."
                            )
                                                        
                            # Import the lora file utils
                            from utils.lora_file_utils import list_lora_files
                            
                            # Get available LoRA files
                            lora_file_options = list_lora_files()
                            lora_file_choices = ["None"] + [lora["name"] for lora in lora_file_options]
                            
                            # Store LoRA path mapping for lookup
                            lora_file_map = {lora["name"]: lora["path"] for lora in lora_file_options}
                            
                            # Simple LoRA selection dropdown (no refresh button)
                            lora_dropdown = gr.Dropdown(
                                label="Select LoRA",
                                choices=lora_file_choices,
                                value="None",
                                info="Select a LoRA from local_models/lora directory"
                            )

                            lora_multiplier = gr.Slider(
                                label="LoRA Multiplier",
                                minimum=0.0,
                                maximum=1.0,
                                value=0.8,
                                step=0.1,
                                info="Strength of the LoRA effect. Higher values make the LoRA more prominent."
                            )
                            
                            # Hidden state to store the current LoRA path
                            lora_path_state = gr.State(None)
                    
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
                    keep_section_videos, end_frame_strength, enable_section_controls, fp8_optimization, lora_path, 
                    lora_multiplier, section_settings=None):
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
                # Debug section settings
                processed_section_settings = section_settings
                print(f"Section controls enabled: Using {len(processed_section_settings)} section settings")
                
                # Print all sections with their details
                valid_sections = [s for s in processed_section_settings if s[0] is not None and (s[1] is not None or (s[2] is not None and s[2].strip()))]
                print(f"Valid section configurations: {len(valid_sections)}")
                
                for sec in valid_sections:
                    sec_num, img, prompt = sec
                    has_img = img is not None
                    has_prompt = prompt is not None and prompt.strip() != ""
                    print(f"  UI: Section {sec_num}: Image: {'✓' if has_img else '✗'}, Prompt: {'✓' if has_prompt else '✗'}")
                    
                # If no valid sections were found, log a warning
                if not valid_sections:
                    print("WARNING: No valid section configurations found!")
            else:
                print(f"Section controls disabled or no section_settings provided. enable_section_controls={enable_section_controls}, section_settings is None: {section_settings is None}")
            
            # Start the worker in a separate thread
            async_run(
                worker, 
                input_image, end_frame, prompt, n_prompt, seed, total_latent_sections, latent_window_size, 
                steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, teacache_thresh, resolution_scale, mp4_crf,
                keep_section_videos, end_frame_strength, processed_section_settings,
                lora_path, lora_multiplier, fp8_optimization,
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
        
        # Define function to handle LoRA dropdown changes
        def update_lora_path(dropdown_value):
            """
            Update the LoRA path based on dropdown selection
            
            Args:
                dropdown_value (str): Selected value from dropdown
                
            Returns:
                str or None: Path to the selected LoRA file or None
            """
            # If "None" is selected, clear the LoRA path
            if dropdown_value == "None":
                return None
                
            # Otherwise, use the selected LoRA
            selected_path = lora_file_map.get(dropdown_value, None)
            return selected_path
        
        # Connect LoRA dropdown to update event
        lora_dropdown.change(
            fn=update_lora_path,
            inputs=[lora_dropdown],
            outputs=[lora_path_state]
        )
        
        # Connect start button with all inputs (using lora_path_state directly)
        start_button.click(
            fn=process,
            inputs=[
                input_image, end_frame, prompt, n_prompt, seed, total_latent_sections, latent_window_size, 
                steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, teacache_thresh, resolution_scale, mp4_crf,
                keep_section_videos, end_frame_strength, enable_section_controls, fp8_optimization, lora_path_state, 
                lora_multiplier, section_settings
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
                               prompt_val, n_prompt_val, cfg_val, rs_val, end_frame_strength_val,
                               fp8_optimization_val, lora_multiplier_val, lora_dropdown_val):
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
                "end_frame_strength": end_frame_strength_val,
                "fp8_optimization": fp8_optimization_val,
                "lora_multiplier": lora_multiplier_val,
                "lora_dropdown": lora_dropdown_val
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
                fp8_optimization: defaults.get("fp8_optimization", False),
                lora_multiplier: defaults.get("lora_multiplier", 0.8),
                lora_dropdown: defaults.get("lora_dropdown", "None"),
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
            # Update the HTML content when latent window size changes
            def update_section_info_html(latent_size):
                return f'<div class="section-info-guide">{calc_section_info_text(latent_size)}</div>'
                
            latent_window_size.change(
                fn=update_section_info_html,
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
        
        # Function to update all section timing displays
        if 'section_timing_displays' in locals():
            def update_all_section_timings(latent_size):
                """Update all section timing displays with the new latent window size"""
                results = []
                for section_idx, _ in section_timing_displays:
                    results.append(f'<div class="section-timing-info">{calc_section_timing(section_idx, latent_size)}</div>')
                return results
            
            # Connect section timing update function
            latent_window_size.change(
                fn=update_all_section_timings,
                inputs=[latent_window_size],
                outputs=[timing_info for _, timing_info in section_timing_displays],
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
                cfg, rs, end_frame_strength, 
                fp8_optimization, lora_multiplier, lora_dropdown, save_status
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
                cfg, rs, end_frame_strength,
                fp8_optimization, lora_multiplier, lora_dropdown
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
                cfg, rs, end_frame_strength,
                fp8_optimization, lora_multiplier, lora_dropdown, save_status
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
            n_prompt, cfg, rs, end_frame_strength,
            fp8_optimization, lora_multiplier, lora_dropdown
        ]

        def load_saved_params():
            """Loads parameters from config and returns them in the correct order."""
            logger.info("Reloading parameters for UI on page load...")
            params = param_config.get_all_parameters()
            
            # Get the saved LoRA dropdown value
            lora_dropdown_value = params.get("lora_dropdown", "None")
            
            # Validate that the saved value is in the current dropdown options
            # This prevents errors if saved LoRA files have been removed
            if lora_dropdown_value not in ["None"] + [lora["name"] for lora in lora_file_options]:
                logger.warning(f"Saved LoRA selection '{lora_dropdown_value}' not found in available options. Using 'None'.")
                lora_dropdown_value = "None"
            
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
                params.get("end_frame_strength", 1.0),
                params.get("fp8_optimization", False),
                params.get("lora_multiplier", 0.8),
                lora_dropdown_value
            ]

        # When the UI loads, it will set the LoRA dropdown value based on the saved parameters
        # We need to also initialize the lora_path_state with the correct path
        def initialize_ui():
            """Initialize UI components and states after loading parameters"""
            # First, load all parameters
            params_values = load_saved_params()
            
            # Extract the LoRA dropdown value (which is the last item)
            lora_dropdown_value = params_values[-1]
            
            # Convert the dropdown value to a path
            path = None
            if lora_dropdown_value != "None":
                path = lora_file_map.get(lora_dropdown_value, None)
                
            # Return all parameter values plus the path for lora_path_state
            return params_values + [path]
        
        # Add lora_path_state to components that need initialization
        initialization_outputs = components_to_load + [lora_path_state]
        
        block.load(
            fn=initialize_ui,
            inputs=[],
            outputs=initialization_outputs,
            show_progress=False # Optional: hide progress indicator for this load
        )
        # --- End of .load() event handler ---
        
        # No need for section status updates anymore

    return block

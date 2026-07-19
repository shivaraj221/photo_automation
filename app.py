import gradio as gr
from PIL import Image
from pathlib import Path
import os
from config import PASSPORT_CONFIG, SUPPORTED_COPIES, FRAMING_PRESETS
import spaces
from pipeline import process_passport_photo

# Ensure required directories exist
os.makedirs("temp", exist_ok=True)
os.makedirs("output", exist_ok=True)

@spaces.GPU
def run_pipeline(image, copies, bg_color_hex, framing_preset, force_ai, progress=gr.Progress()):
    if image is None:
        raise gr.Error("⚠️ Please upload a portrait photo first.")

    progress(0.1, desc="Saving input...")
    temp_path = "temp/input_image.png"
    Image.fromarray(image).save(temp_path)

    # Build config from preset + user overrides
    p_cfg = PASSPORT_CONFIG.copy()
    if framing_preset and framing_preset in FRAMING_PRESETS:
        p_cfg.update(FRAMING_PRESETS[framing_preset])

    # Convert hex to BGR for OpenCV
    bg_rgb = tuple(int(bg_color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    p_cfg["background_color"] = (bg_rgb[2], bg_rgb[1], bg_rgb[0])

    progress(0.3, desc="Detecting face...")
    try:
        progress(0.5, desc="Processing portrait...")
        results = process_passport_photo(
            input_path=temp_path,
            copies=int(copies),
            force_ai=force_ai,
            passport_config=p_cfg
        )
        progress(0.9, desc="Rendering grid...")
        progress(1.0, desc="Done!")
        return results["passport_path"], results["sheet_path"]
    except Exception as e:
        raise gr.Error(f"Processing failed: {str(e)}")


# ═══════════════════════════════════════════════════════════════════════════
#  PREMIUM CSS — Studio Elite Pro Max
# ═══════════════════════════════════════════════════════════════════════════

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Global Reset ── */
*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background: #09090b !important;
    color: #fafafa !important;
}

.gradio-container {
    max-width: 1280px !important;
    padding: 0 2rem !important;
}

/* ── Animated Background Orbs ── */
.gradio-container::before {
    content: '';
    position: fixed;
    top: -200px;
    left: -200px;
    width: 600px;
    height: 600px;
    background: radial-gradient(circle, rgba(99,102,241,0.15) 0%, transparent 70%);
    animation: float-orb 20s ease-in-out infinite alternate;
    pointer-events: none;
    z-index: 0;
}

.gradio-container::after {
    content: '';
    position: fixed;
    bottom: -300px;
    right: -200px;
    width: 800px;
    height: 800px;
    background: radial-gradient(circle, rgba(168,85,247,0.12) 0%, transparent 70%);
    animation: float-orb 25s ease-in-out infinite alternate-reverse;
    pointer-events: none;
    z-index: 0;
}

@keyframes float-orb {
    0%   { transform: translate(0, 0) scale(1); }
    50%  { transform: translate(60px, -40px) scale(1.1); }
    100% { transform: translate(-30px, 50px) scale(0.95); }
}

/* ── Glass Panel ── */
.glass-panel {
    position: relative;
    background: rgba(255, 255, 255, 0.03) !important;
    backdrop-filter: blur(40px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(40px) saturate(180%) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 24px !important;
    padding: 2rem !important;
    transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1) !important;
    overflow: hidden;
}

.glass-panel::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.15), transparent);
}

.glass-panel:hover {
    border-color: rgba(129, 140, 248, 0.2) !important;
    box-shadow: 0 25px 80px -20px rgba(99, 102, 241, 0.15),
                inset 0 1px 0 rgba(255, 255, 255, 0.05) !important;
    transform: translateY(-2px) !important;
}

/* ── Section Labels ── */
.step-label span {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.7rem !important;
    font-weight: 700 !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    color: #818cf8 !important;
    padding-bottom: 1rem !important;
}

/* ── Form Controls ── */
.gradio-container label span {
    color: #a1a1aa !important;
    font-weight: 500 !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.5px !important;
}

.gradio-container input,
.gradio-container select,
.gradio-container textarea,
.gradio-container .wrap {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 12px !important;
    color: #fafafa !important;
    transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
}

.gradio-container input:focus,
.gradio-container select:focus {
    border-color: rgba(129, 140, 248, 0.5) !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1) !important;
    outline: none !important;
}

/* ── Generate Button ── */
#generate-btn {
    width: 100% !important;
    height: 3.5rem !important;
    border-radius: 16px !important;
    border: none !important;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    cursor: pointer !important;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 8px 30px -8px rgba(99, 102, 241, 0.5) !important;
    margin-top: 1rem !important;
}

#generate-btn:hover {
    transform: translateY(-3px) scale(1.01) !important;
    box-shadow: 0 16px 50px -10px rgba(99, 102, 241, 0.6) !important;
    filter: brightness(1.1) !important;
}

#generate-btn:active {
    transform: translateY(0) scale(0.99) !important;
}

/* ── Output Images ── */
.output-panel img {
    border-radius: 16px !important;
    object-fit: contain !important;
    max-height: 420px !important;
    width: 100% !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
}

/* ── Upload Area ── */
.gradio-image {
    border-radius: 16px !important;
    overflow: hidden !important;
}

.gradio-image .upload-container {
    border: 2px dashed rgba(255, 255, 255, 0.1) !important;
    border-radius: 16px !important;
    transition: border-color 0.3s ease !important;
    min-height: 220px !important;
}

.gradio-image .upload-container:hover {
    border-color: rgba(129, 140, 248, 0.4) !important;
}

/* ── Checkbox ── */
.gradio-container .gr-check-radio {
    accent-color: #818cf8 !important;
}

/* ── Hide Gradio Footer ── */
footer { display: none !important; }

/* ── Scroll Bar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.2); }

/* ── Dropdown Styling ── */
.gradio-container ul[role="listbox"] {
    background: #18181b !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
}

.gradio-container ul[role="listbox"] li {
    color: #fafafa !important;
}

.gradio-container ul[role="listbox"] li:hover {
    background: rgba(99, 102, 241, 0.2) !important;
}

/* ── Status Badge ── */
.status-bar p {
    text-align: center !important;
    color: #3f3f46 !important;
    font-size: 0.65rem !important;
    letter-spacing: 6px !important;
    text-transform: uppercase !important;
    padding-top: 3rem !important;
}
"""

custom_theme = gr.themes.Default(
    primary_hue="indigo",
    secondary_hue="purple",
    neutral_hue="zinc",
    font=gr.themes.GoogleFont("Inter"),
).set(
    body_background_fill="transparent",
    block_background_fill="transparent",
    block_border_width="0px",
    block_shadow="none",
    input_background_fill="rgba(255,255,255,0.04)",
    input_border_color="rgba(255,255,255,0.08)",
    input_border_width="1px",
    button_primary_background_fill="linear-gradient(135deg, #6366f1, #a855f7)",
    button_primary_text_color="white",
)


# ═══════════════════════════════════════════════════════════════════════════
#  UI LAYOUT
# ═══════════════════════════════════════════════════════════════════════════

with gr.Blocks(title="Studio Elite Pro Max") as demo:

    # ── Hero Header ──
    gr.HTML("""
    <div style="text-align:center; padding: 3rem 0 2.5rem;">
        <div style="display:inline-flex; align-items:center; gap:0.75rem; margin-bottom:1rem;">
            <div style="width:10px; height:10px; border-radius:50%; background:#818cf8; box-shadow: 0 0 12px #818cf8;"></div>
            <span style="font-size:0.7rem; font-weight:700; letter-spacing:4px; color:#818cf8; text-transform:uppercase;">AI-Powered</span>
        </div>
        <h1 style="
            font-family:'Inter',sans-serif;
            font-weight:900;
            font-size:clamp(2.5rem,6vw,4rem);
            background: linear-gradient(135deg, #fafafa 0%, #71717a 100%);
            -webkit-background-clip:text;
            -webkit-text-fill-color:transparent;
            margin:0;
            line-height:1.1;
            letter-spacing:-1px;
        ">Studio Elite Pro Max</h1>
        <p style="
            color:#52525b;
            font-size:0.85rem;
            font-weight:500;
            letter-spacing:3px;
            text-transform:uppercase;
            margin-top:0.75rem;
        ">Passport Photo Generator · Neural Engine v4.0</p>
    </div>
    """)

    with gr.Row(equal_height=False):

        # ════════════ LEFT: Controls ════════════
        with gr.Column(scale=1):

            # Upload Card
            with gr.Column(elem_classes="glass-panel"):
                gr.Markdown("01 · UPLOAD", elem_classes="step-label")
                img_input = gr.Image(
                    type="numpy",
                    label="Drop your portrait here",
                    height=240,
                )

            # Settings Card
            with gr.Column(elem_classes="glass-panel"):
                gr.Markdown("02 · CONFIGURE", elem_classes="step-label")

                copies_input = gr.Dropdown(
                    choices=[str(c) for c in SUPPORTED_COPIES],
                    value="8",
                    label="Copies per Sheet",
                )

                framing_input = gr.Dropdown(
                    choices=list(FRAMING_PRESETS.keys()),
                    value="Balanced",
                    label="Framing Preset",
                )

                bg_color_input = gr.ColorPicker(
                    value="#FFFFFF",
                    label="Background Color",
                )

                force_ai_input = gr.Checkbox(
                    label="✨ AI Enhancement (GFPGAN)",
                    value=False,
                )

                submit_btn = gr.Button(
                    "⚡ Generate Photos",
                    elem_id="generate-btn",
                )

        # ════════════ RIGHT: Output ════════════
        with gr.Column(scale=1):

            with gr.Column(elem_classes="glass-panel"):
                gr.Markdown("03 · RESULTS", elem_classes="step-label")

                passport_output = gr.Image(
                    label="Passport Portrait",
                    type="filepath",
                    elem_classes="output-panel",
                )

                sheet_output = gr.Image(
                    label="4×6 Print Sheet",
                    type="filepath",
                    elem_classes="output-panel",
                )

    # ── Connect ──
    submit_btn.click(
        fn=run_pipeline,
        inputs=[img_input, copies_input, bg_color_input, framing_input, force_ai_input],
        outputs=[passport_output, sheet_output],
    )

    # ── Footer ──
    gr.Markdown("Studio Elite Pro Max · Built with Neural Engine v4.0", elem_classes="status-bar")


# ═══════════════════════════════════════════════════════════════════════════
#  Launch
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            theme=custom_theme,
            css=custom_css,
        )
    except TypeError:
        demo.theme = custom_theme
        demo.css = custom_css
        demo.launch(server_name="0.0.0.0", server_port=7860)

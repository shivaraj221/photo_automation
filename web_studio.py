import streamlit as st
import os
import base64
import time
from pathlib import Path
from PIL import Image
from pipeline import process_passport_photo
from config import SUPPORTED_COPIES, OUTPUT_DIR
from print.printer import list_printers, print_image

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Studio Elite Pro Max",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- WORLD-CLASS ELITE MAX CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Outfit', sans-serif;
        background: #000000;
        color: #ffffff;
    }

    .block-container {
        max-width: 1400px;
        padding-top: 5rem;
        padding-bottom: 10rem;
        margin: auto;
    }

    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at 50% 0%, #1e1b4b 0%, #000000 80%);
    }

    .elite-card {
        background: rgba(255, 255, 255, 0.02);
        backdrop-filter: blur(30px) saturate(200%);
        border-radius: 40px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 4rem;
        margin-bottom: 4rem;
        box-shadow: 0 40px 100px -20px rgba(0, 0, 0, 0.8);
    }

    .hero-container {
        text-align: center;
        padding-bottom: 6rem;
    }
    .hero-container h1 {
        font-weight: 800;
        font-size: clamp(4rem, 12vw, 7rem);
        background: linear-gradient(135deg, #ffffff 30%, #475569 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -5px;
        line-height: 0.9;
        margin-bottom: 1.5rem;
    }

    .stButton>button {
        width: 100%;
        border-radius: 25px;
        height: 5rem;
        background: #ffffff;
        color: #000000;
        border: none;
        font-weight: 800;
        font-size: 1.4rem;
        transition: all 0.6s cubic-bezier(0.19, 1, 0.22, 1);
        text-transform: uppercase;
        letter-spacing: 4px;
    }
    .stButton>button:hover {
        transform: translateY(-10px) scale(1.01);
        background: #f1f5f9;
        box-shadow: 0 30px 60px rgba(255,255,255,0.15);
    }

    .section-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 2.5rem;
        background: linear-gradient(90deg, #6366f1, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: 1px;
    }
    </style>
""", unsafe_allow_html=True)

# --- HELPERS ---
def get_image_download_link(img_path, filename, text):
    with open(img_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    return f'<a href="data:image/png;base64,{b64}" download="{filename}" style="text-decoration:none;"><div style="background:rgba(255,255,255,0.05); color:white; padding:20px 40px; border-radius:20px; font-weight:600; text-align:center; border:1px solid rgba(255,255,255,0.1); transition: 0.3s; margin-top:10px; font-size:1.1rem;">📥 {text}</div></a>'

def trigger_browser_print(img_path):
    with open(img_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    # Custom JS to open a new window with ONLY the image and trigger print
    js = f"""
    <script>
    var printWindow = window.open('', '_blank');
    printWindow.document.write('<html><head><title>Print Photo</title>');
    printWindow.document.write('<style>@page {{ size: 4in 6in; margin: 0; }} body {{ margin: 0; padding: 0; display: flex; justify-content: center; align-items: center; background: white; }} img {{ width: 100%; height: auto; }}</style>');
    printWindow.document.write('</head><body>');
    printWindow.document.write('<img src="data:image/png;base64,{b64}" onload="window.print();window.close()">');
    printWindow.document.write('</body></html>');
    printWindow.document.close();
    </script>
    """
    st.components.v1.html(js, height=0)

# --- PAGE CONTENT ---

st.markdown('<div class="hero-container"><h1>STUDIO MAX</h1><p>ULTRA-WIDE AI ENGINE</p></div>', unsafe_allow_html=True)

# 1. INPUT SECTION
st.markdown('<div class="elite-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">01. SOURCE DEFINITION</div>', unsafe_allow_html=True)
file = st.file_uploader("Upload Portrait", type=['jpg','jpeg','png','webp'], label_visibility="collapsed")

st.markdown('<div class="section-title" style="margin-top:4rem;">02. ENGINE PARAMETERS</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3, gap="large")
with c1:
    copies = st.selectbox("Grid Multiplier", SUPPORTED_COPIES, index=1)
with c2:
    bg_color_hex = st.color_picker("Background Color", value="#FFFFFF")
with c3:
    st.markdown("<br>", unsafe_allow_html=True)
    quality = st.toggle("Neural Quality Matrix ✨", value=False)

printer_list = ["System default (Main PC)"] + list_printers()
printer = st.selectbox("Remote Epson Hardware", printer_list, help="These are printers connected to the MAIN computer.")

st.markdown("<br><br>", unsafe_allow_html=True)
process = st.button("INITIALIZE AI PIPELINE")
st.markdown('</div>', unsafe_allow_html=True)

# 2. STATUS
if process and file:
    st.markdown('<div class="elite-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">03. LIVE EXECUTION LOG</div>', unsafe_allow_html=True)
    steps = ["Booting...", "Scanning...", "Restoring...", "Matting...", "Exporting..."]
    progress_bar = st.progress(0)
    status_box = st.empty()
    temp_p = Path("temp") / file.name
    temp_p.parent.mkdir(exist_ok=True)
    with open(temp_p, "wb") as f: f.write(file.getbuffer())
    try:
        from config import PASSPORT_CONFIG
        p_cfg = PASSPORT_CONFIG.copy()
        bg_rgb = tuple(int(bg_color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        p_cfg["background_color"] = (bg_rgb[2], bg_rgb[1], bg_rgb[0])

        for i, step in enumerate(steps):
            status_box.markdown(f'<div style="font-size: 1.5rem; color: #6366f1; margin-bottom: 2rem;">⚡ <b>STATE:</b> `{step}`</div>', unsafe_allow_html=True)
            progress_bar.progress((i + 1) * 20)
            time.sleep(0.4)
            if i == 0:
                results = process_passport_photo(input_path=str(temp_p), copies=copies, force_ai=quality, passport_config=p_cfg)
        st.session_state['res'] = results
        st.rerun()
    except Exception as e:
        st.error(f"ENGINE FAILURE: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

# 3. RESULTS & DUAL PRINTING
if 'res' in st.session_state:
    res = st.session_state['res']
    st.markdown('<div class="elite-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">04. NEURAL OUTPUT MASTER</div>', unsafe_allow_html=True)
    
    pc1, pc2 = st.columns([1, 1.8], gap="large")
    with pc1: st.image(res['passport_path'], caption="MASTER PORTRAIT")
    with pc2: st.image(res['sheet_path'], caption="PRECISION 4x6 GRID")
    
    st.markdown('<div style="margin-top: 5rem; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 4rem;"></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-title">05. DISPATCH CENTER</div>', unsafe_allow_html=True)
    
    d1, d2 = st.columns(2, gap="medium")
    with d1: st.markdown(get_image_download_link(res['passport_path'], "passport.png", "DOWNLOAD MASTER"), unsafe_allow_html=True)
    with d2: st.markdown(get_image_download_link(res['sheet_path'], "sheet.png", "DOWNLOAD PRINT GRID"), unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # DUAL PRINTING OPTIONS
    pcol1, pcol2 = st.columns(2, gap="large")
    with pcol1:
        st.info("💻 **REMOTE PRINT**\n\nSends job to the Epson printer plugged into the **Main PC**.")
        if st.button("PRINT ON MAIN PC"):
            try:
                p_name = None if printer == "System default (Main PC)" else printer
                print_image(res['sheet_path'], printer_name=p_name)
                st.success("JOB DISPATCHED TO MAIN PC")
                st.balloons()
            except Exception as e: st.error(f"HARDWARE ERROR: {e}")
            
    with pcol2:
        st.info("📱 **LOCAL PRINT**\n\nOpens the print window on **THIS device** (Your phone or this PC).")
        if st.button("PRINT FROM THIS DEVICE"):
            trigger_browser_print(res['sheet_path'])

    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="height: 500px; display: flex; align-items: center; justify-content: center; border: 1px solid rgba(255,255,255,0.05); border-radius: 40px; color: #1e293b; text-align: center; background: rgba(255,255,255,0.01); margin-top: 4rem;">
        <div>
            <img src="https://img.icons8.com/ios-filled/100/ffffff/brain-puzzles.png" style="opacity: 0.03; margin-bottom: 3rem;"><br>
            <span style="letter-spacing: 20px; font-weight: 200; font-size: 1.2rem; text-transform: uppercase;">Awaiting System Signal</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<p style='text-align: center; color: #0f172a; margin-top: 10rem; font-size: 1rem; letter-spacing: 15px;'>STUDIO MAX • NEURAL v4.0</p>", unsafe_allow_html=True)

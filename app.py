"""
AI Football Analysis — Streamlit application.

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import os
import html
import textwrap
import uuid

import streamlit as st

import config as cfg
from services.detection import ModelBundle, run_pipeline
from services.event_engine import CoordinateTransformer
from services.match_statistics import build_match_report_text, run_full_analysis
from services.utils import AnalysisError, resolve_device
from services.visualization import plot_heatmap, plot_passing_network, render_annotated_video

st.set_page_config(page_title="AI Football Analysis", page_icon="⚽", layout="wide")

TEAM_A_COLOR = "#FF4D6D"
TEAM_B_COLOR = "#00BFFF"
NEON_GREEN = "#00FF9D"
NEON_CYAN = "#00BFFF"

# ────────────────────────────────────────────────────────────
#  Global styling — premium neon glassmorphism dashboard theme
# ────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=Poppins:wght@600;700;800&display=swap');

    #MainMenu, footer {{visibility: hidden;}}
    header[data-testid="stHeader"] {{
        background: transparent;
        box-shadow: none;
    }}

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}
    [data-testid="stIconMaterial"], span[data-testid="stIconMaterial"] {{
        font-family: 'Material Symbols Rounded' !important;
    }}

    .stApp {{
        background:
            radial-gradient(circle at 12% -10%, rgba(0,255,157,0.10) 0%, transparent 40%),
            radial-gradient(circle at 90% 10%, rgba(0,191,255,0.08) 0%, transparent 40%),
            #050505;
        color: #f5fff9;
    }}

    div[data-testid="stCaptionContainer"],
    div[data-testid="stCaptionContainer"] p,
    div[data-testid="stWidgetLabel"] p,
    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stRadio"] label p,
    div[data-testid="stCheckbox"] label p,
    div[data-testid="stExpander"] summary,
    div[data-testid="stExpander"] summary span,
    label {{
        color: #f5fff9 !important;
    }}

    .hero {{
        position: relative;
        background: linear-gradient(135deg, rgba(0,255,157,0.10) 0%, rgba(11,15,19,0.9) 55%, rgba(0,191,255,0.08) 100%);
        border: 1px solid rgba(0,255,157,0.18);
        border-radius: 20px;
        padding: 30px 34px;
        margin-bottom: 24px;
        backdrop-filter: blur(18px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.04);
        overflow: hidden;
    }}
    .hero::before {{
        content: "";
        position: absolute; inset: 0;
        background: linear-gradient(90deg, {NEON_GREEN}, {NEON_CYAN});
        height: 3px; top: 0; left: 0; right: 0;
        opacity: 0.8;
    }}
    .hero h1 {{
        color: #f5fff9;
        font-family: 'Poppins', sans-serif;
        font-size: 2.1rem;
        font-weight: 800;
        margin: 0 0 6px 0;
        letter-spacing: -0.5px;
        text-shadow: 0 0 22px rgba(0,255,157,0.25);
    }}
    .hero p {{
        color: #9db3ab;
        margin: 0;
        font-size: 0.95rem;
        letter-spacing: 0.01em;
    }}

    .kpi-card {{
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 18px 20px;
        height: 100%;
        backdrop-filter: blur(14px);
        transition: all 0.25s ease;
        box-shadow: 0 4px 18px rgba(0,0,0,0.3);
    }}
    .kpi-card:hover {{
        border-color: rgba(0,255,157,0.4);
        box-shadow: 0 4px 24px rgba(0,255,157,0.15), 0 0 0 1px rgba(0,255,157,0.15);
        transform: translateY(-2px);
    }}
    .kpi-label {{
        color: #7fa895;
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 8px;
    }}
    .kpi-value {{
        color: #f5fff9;
        font-family: 'Poppins', sans-serif;
        font-size: 1.6rem;
        font-weight: 800;
        line-height: 1.15;
    }}
    .kpi-sub {{
        color: #5f8977;
        font-size: 0.76rem;
        margin-top: 5px;
    }}

    .poss-wrap {{ margin: 8px 0 4px 0; }}
    .poss-bar {{
        display: flex; width: 100%; height: 36px; border-radius: 10px;
        overflow: hidden; font-weight: 700; font-size: 0.85rem;
        box-shadow: inset 0 0 0 1px rgba(255,255,255,0.1);
    }}
    .poss-a {{ background: linear-gradient(90deg, {TEAM_A_COLOR}, #ff7a94); display:flex; align-items:center; justify-content:center; color:white; }}
    .poss-b {{ background: linear-gradient(90deg, #0088cc, {TEAM_B_COLOR}); display:flex; align-items:center; justify-content:center; color:white; }}
    .poss-loose {{ background: rgba(255,255,255,0.08); display:flex; align-items:center; justify-content:center; color:#c9d6d0; font-size:0.72rem; }}

    .team-chip {{
        display:inline-block; padding: 4px 12px; border-radius: 999px;
        font-size: 0.75rem; font-weight: 700; color: white; margin-right: 8px;
        box-shadow: 0 0 12px rgba(0,0,0,0.4);
    }}

    .section-title {{
        color: #f5fff9; font-family: 'Poppins', sans-serif;
        font-size: 1.05rem; font-weight: 700;
        margin: 6px 0 14px 0; padding-left: 12px;
        border-left: 3px solid {NEON_GREEN};
        text-shadow: 0 0 12px rgba(0,255,157,0.2);
    }}

    div[data-testid="stMetric"] {{
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 14px 18px 8px 18px;
        backdrop-filter: blur(14px);
    }}

    .stTabs [data-baseweb="tab-list"] {{ gap: 6px; }}
    .stTabs [data-baseweb="tab"] {{
        background-color: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px 12px 0 0;
        padding: 9px 20px;
        color: #8fada0;
        font-weight: 500;
    }}
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, rgba(0,255,157,0.18), rgba(0,191,255,0.10)) !important;
        color: #f5fff9 !important;
        border-color: rgba(0,255,157,0.35) !important;
    }}

    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #0B0F13 0%, #050505 100%);
        border-right: 1px solid rgba(0,255,157,0.08);
    }}
    section[data-testid="stSidebar"] * {{ font-family: 'Inter', sans-serif; }}

    .stButton > button {{
        background: linear-gradient(135deg, {NEON_GREEN} 0%, {NEON_CYAN} 100%);
        color: #06110b;
        font-weight: 700;
        border: none;
        border-radius: 12px;
        padding: 10px 18px;
        box-shadow: 0 4px 20px rgba(0,255,157,0.25);
        transition: all 0.2s ease;
    }}
    .stButton > button:hover {{
        box-shadow: 0 6px 28px rgba(0,255,157,0.4);
        transform: translateY(-1px);
    }}
    .stButton > button:disabled {{
        background: rgba(255,255,255,0.06);
        color: #5c6b64;
        box-shadow: none;
    }}

    section[data-testid="stFileUploaderDropzone"] {{
        background: rgba(255,255,255,0.03) !important;
        border: 1.5px dashed rgba(0,255,157,0.35) !important;
        border-radius: 16px !important;
        transition: all 0.25s ease;
    }}
    section[data-testid="stFileUploaderDropzone"]:hover {{
        border-color: rgba(0,255,157,0.7) !important;
        box-shadow: 0 0 24px rgba(0,255,157,0.12);
    }}

    div[data-testid="stDataFrame"] {{
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.08);
    }}

    div[data-testid="stProgress"] > div > div {{
        background: linear-gradient(90deg, {NEON_GREEN}, {NEON_CYAN}) !important;
    }}

    details {{
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
    }}

    .app-footer {{
        margin-top: 40px;
        padding: 18px 8px 6px 8px;
        border-top: 1px solid rgba(255,255,255,0.06);
        text-align: center;
        color: #4d6459;
        font-size: 0.78rem;
        letter-spacing: 0.03em;
    }}
    .app-footer span {{
        display: inline-block; margin: 0 10px; color: #6f9483;
    }}

    .report-box {{
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 26px 30px;
        font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace;
        font-size: 0.86rem;
        line-height: 1.7;
        color: #cfe6dc;
        white-space: pre-wrap;
        word-break: break-word;
        max-height: 640px;
        overflow-y: auto;
        backdrop-filter: blur(14px);
        box-shadow: 0 4px 18px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.04);
    }}
    .report-box::-webkit-scrollbar {{ width: 8px; }}
    .report-box::-webkit-scrollbar-thumb {{
        background: rgba(0,255,157,0.25); border-radius: 8px;
    }}
    .report-header {{
        color: {NEON_GREEN};
        font-weight: 700;
        letter-spacing: 0.04em;
    }}
    .report-rule {{
        color: rgba(255,255,255,0.14);
    }}
</style>
""", unsafe_allow_html=True)


def kpi_card(col, label, value, sub=""):
    with col:
        st.markdown(textwrap.dedent(f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """).strip(), unsafe_allow_html=True)


def team_chip(label, color):
    return f'<span class="team-chip" style="background:{color}">{label}</span>'


def render_report_html(report_text: str) -> str:
    out_lines = []
    for line in report_text.split("\n"):
        stripped = line.strip()
        if stripped and set(stripped) <= {"="}:
            out_lines.append(f'<span class="report-rule">{html.escape(line)}</span>')
        elif stripped and stripped.isupper() and not any(ch.isdigit() for ch in stripped):
            out_lines.append(f'<span class="report-header">{html.escape(line)}</span>')
        else:
            out_lines.append(html.escape(line))
    return "\n".join(out_lines)


def render_footer():
    st.markdown(textwrap.dedent(f"""
    <div class="app-footer">
        <span>⚡ YOLO11</span>
        <span>🎯 ByteTrack</span>
        <span>👁️ OpenCV</span>
        <span>🔥 PyTorch</span>
        <span>🚀 Streamlit</span>
    </div>
    """).strip(), unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_models(device: str) -> ModelBundle:
    return ModelBundle(device=device)


def init_session_state():
    defaults = {
        "results": None,
        "pipeline_result": None,
        "output_video_path": None,
        "processing": False,
        "error": None,
        "app_started": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session_state()


def render_landing_page():
    st.markdown(textwrap.dedent(f"""
    <style>
        #MainMenu, footer {{visibility: hidden;}}
        header[data-testid="stHeader"] {{ background: transparent; box-shadow: none; }}
        .block-container {{ padding-top: 2rem; max-width: 1100px; }}

        @keyframes floatUp {{
            0%   {{ transform: translateY(0px) translateX(0px); opacity: 0; }}
            10%  {{ opacity: 0.8; }}
            100% {{ transform: translateY(-620px) translateX(30px); opacity: 0; }}
        }}
        @keyframes pulseGlow {{
            0%, 100% {{ text-shadow: 0 0 24px rgba(0,255,157,0.35), 0 0 60px rgba(0,191,255,0.15); }}
            50%      {{ text-shadow: 0 0 44px rgba(0,255,157,0.65), 0 0 90px rgba(0,191,255,0.35); }}
        }}
        @keyframes fadeInUp {{
            from {{ opacity: 0; transform: translateY(16px); }}
            to   {{ opacity: 1; transform: translateY(0); }}
        }}

        .landing-wrap {{
            position: relative;
            min-height: 640px;
            border-radius: 24px;
            overflow: hidden;
            background:
                radial-gradient(circle at 20% 20%, rgba(0,255,157,0.12) 0%, transparent 45%),
                radial-gradient(circle at 80% 30%, rgba(0,191,255,0.10) 0%, transparent 45%),
                linear-gradient(180deg, #0B0F13 0%, #050505 100%);
            border: 1px solid rgba(0,255,157,0.15);
            padding: 70px 40px 50px 40px;
            text-align: center;
            box-shadow: 0 20px 60px rgba(0,0,0,0.6);
        }}
        .pitch-lines {{
            position: absolute; inset: 0; opacity: 0.10; pointer-events: none;
            background-image:
                repeating-linear-gradient(0deg, transparent, transparent 78px, rgba(0,255,157,0.5) 79px, transparent 80px),
                linear-gradient(rgba(0,255,157,0.35) 1px, transparent 1px),
                linear-gradient(90deg, rgba(0,255,157,0.35) 1px, transparent 1px);
            background-size: 100% 100%, 100% 100px, 100px 100%;
        }}
        .particle {{
            position: absolute; bottom: -20px; border-radius: 50%;
            background: radial-gradient(circle, rgba(0,255,157,0.9) 0%, rgba(0,255,157,0) 70%);
            animation: floatUp linear infinite;
            pointer-events: none;
        }}
        .landing-logo {{
            font-size: 4rem; margin-bottom: 6px;
            animation: pulseGlow 2.6s ease-in-out infinite;
        }}
        .landing-title {{
            font-family: 'Poppins', sans-serif;
            font-size: 3rem; font-weight: 800; letter-spacing: -1px;
            background: linear-gradient(90deg, #ffffff 20%, {NEON_GREEN} 60%, {NEON_CYAN} 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            background-clip: text;
            margin: 6px 0 14px 0;
            animation: fadeInUp 0.9s ease both;
            position: relative; z-index: 2;
        }}
        .landing-sub {{
            color: #9db3ab; font-size: 1.15rem; font-weight: 500;
            letter-spacing: 0.02em; margin-bottom: 26px;
            animation: fadeInUp 0.9s ease 0.15s both;
            position: relative; z-index: 2;
        }}
        .landing-sub b {{ color: {NEON_GREEN}; font-weight: 700; }}
        .landing-badges {{
            display: flex; justify-content: center; gap: 12px; flex-wrap: wrap;
            margin-bottom: 34px; position: relative; z-index: 2;
            animation: fadeInUp 0.9s ease 0.3s both;
        }}
        .landing-badge {{
            padding: 7px 16px; border-radius: 999px; font-size: 0.82rem; font-weight: 600;
            background: rgba(255,255,255,0.045); border: 1px solid rgba(255,255,255,0.12);
            color: #cfe6dc; backdrop-filter: blur(10px);
        }}
        .landing-powered {{
            color: #567a6a; font-size: 0.85rem; letter-spacing: 0.06em;
            text-transform: uppercase; margin-bottom: 40px; position: relative; z-index: 2;
            animation: fadeInUp 0.9s ease 0.4s both;
        }}
        .landing-powered b {{ color: #9fe6c8; }}
    </style>

    <div class="landing-wrap">
        <div class="pitch-lines"></div>
        <div class="particle" style="left:8%;  width:5px; height:5px; animation-duration:7s;  animation-delay:0s;"></div>
        <div class="particle" style="left:18%; width:3px; height:3px; animation-duration:9s;  animation-delay:1.2s;"></div>
        <div class="particle" style="left:32%; width:6px; height:6px; animation-duration:6.5s;animation-delay:0.4s;"></div>
        <div class="particle" style="left:48%; width:4px; height:4px; animation-duration:8s;  animation-delay:2s;"></div>
        <div class="particle" style="left:63%; width:3px; height:3px; animation-duration:7.5s;animation-delay:0.8s;"></div>
        <div class="particle" style="left:77%; width:5px; height:5px; animation-duration:9.5s;animation-delay:1.6s;"></div>
        <div class="particle" style="left:90%; width:4px; height:4px; animation-duration:6.8s;animation-delay:0.3s;"></div>

        <div class="landing-logo">⚽</div>
        <div class="landing-title">AI Football Analysis Platform</div>
        <div class="landing-sub">
            <b>Computer Vision</b> &nbsp;·&nbsp; <b>Machine Learning</b> &nbsp;·&nbsp; <b>Tactical Intelligence</b>
        </div>
        <div class="landing-badges">
            <span class="landing-badge">🎯 Player Detection</span>
            <span class="landing-badge">🏃 Multi-Object Tracking</span>
            <span class="landing-badge">🔥 Heatmaps</span>
            <span class="landing-badge">🎯 Passing Networks</span>
            <span class="landing-badge">📊 Match Statistics</span>
            <span class="landing-badge">🤖 AI Tactical Reports</span>
        </div>
        <div class="landing-powered">Powered by &nbsp;<b>YOLO11</b>&nbsp; + &nbsp;<b>ByteTrack</b></div>
    </div>
    """).strip(), unsafe_allow_html=True)

    _, mid, _ = st.columns([1, 1, 1])
    with mid:
        if st.button("▶  Start Analysis", type="primary", use_container_width=True, key="landing_start"):
            st.session_state["app_started"] = True
            st.rerun()


if not st.session_state["app_started"]:
    render_landing_page()
    st.stop()


with st.sidebar:
    st.markdown(textwrap.dedent(f"""
    <div style="text-align:center; padding: 6px 0 14px 0;">
        <div style="font-size:2.1rem; line-height:1;">⚽</div>
        <div style="font-family:'Poppins',sans-serif; font-weight:800; font-size:1.05rem;
                    color:#f5fff9; margin-top:6px;">AI FOOTBALL ANALYSIS</div>
        <div style="font-size:0.72rem; color:{NEON_GREEN}; letter-spacing:0.08em; margin-top:2px;">
            YOLO11 · ByteTrack
        </div>
    </div>
    """).strip(), unsafe_allow_html=True)
    st.caption("Upload match footage to run detection, tracking, and analytics.")
    st.markdown("---")

    st.markdown(textwrap.dedent("""
    <div style="font-size:0.8rem; font-weight:700; color:#cfe6dc; margin-bottom:4px;
                text-transform:uppercase; letter-spacing:0.05em;">
        📤 Upload Match Footage
    </div>
    """).strip(), unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Video Upload", type=list(cfg.SUPPORTED_VIDEO_FORMATS),
        help="Supported formats: " + ", ".join(cfg.SUPPORTED_VIDEO_FORMATS),
        label_visibility="collapsed",
    )
    st.caption("Supported: " + ", ".join(cfg.SUPPORTED_VIDEO_FORMATS).upper())

    with st.expander("⚙️ Analysis Settings", expanded=True):
        confidence = st.slider(
            "Confidence Threshold", min_value=0.05, max_value=0.90,
            value=cfg.DEFAULT_CONFIDENCE, step=0.05,
            help="Minimum detection confidence for the player/referee model.",
        )
        device_choice = st.radio("Device", options=["cpu", "cuda"], index=0,
                                  help="CUDA requires an available GPU.", horizontal=True)
        compute_homography = st.checkbox(
            "Compute pitch homography", value=True,
            help="Enables real-world pitch-metre coordinates for more accurate "
                 "directional statistics and heatmaps. Slower to process.",
        )

    start_clicked = st.button("▶  Start Analysis", type="primary",
                               use_container_width=True, disabled=uploaded_file is None)

    st.markdown("---")
    st.markdown(textwrap.dedent(f"""
    <div style="display:flex; gap:8px;">
        <span class="team-chip" style="background:{TEAM_A_COLOR};">Team A</span>
        <span class="team-chip" style="background:{TEAM_B_COLOR};">Team B</span>
    </div>
    """).strip(), unsafe_allow_html=True)


st.markdown("""
<div class="hero">
    <h1>AI Football Match Analysis</h1>
    <p>Computer-vision powered detection, tracking, and professional-grade match analytics.</p>
</div>
""", unsafe_allow_html=True)

col_preview, col_status = st.columns([2, 1])

video_path = None
if uploaded_file is not None:
    ext = uploaded_file.name.split(".")[-1].lower()
    session_id = st.session_state.get("session_id") or str(uuid.uuid4())[:8]
    st.session_state["session_id"] = session_id
    video_path = os.path.join(cfg.TEMP_DIR, f"input_{session_id}.{ext}")
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with col_preview:
        st.markdown('<div class="section-title">Uploaded Video</div>', unsafe_allow_html=True)
        st.video(video_path)
    with col_status:
        st.markdown('<div class="section-title">Status</div>', unsafe_allow_html=True)
        st.info("Ready. Configure settings in the sidebar and click **Start Analysis**.")

progress_placeholder = st.empty()

STAGES = [
    ("Uploading Video", 0.02),
    ("Loading Models", 0.08),
    ("Detecting Players", 0.30),
    ("Tracking Ball", 0.50),
    ("Classifying Teams", 0.65),
    ("Calculating Statistics", 0.80),
    ("Generating Heatmaps", 0.88),
    ("Building Passing Network", 0.92),
    ("Generating AI Report", 0.95),
    ("Rendering Video", 0.99),
    ("Analysis Complete", 1.0),
]


def render_stage_loader(frac: float, msg: str):
    pct = int(frac * 100)
    rows = []
    for label, threshold in STAGES:
        if frac >= threshold:
            icon, cls = "✅", "stage-done"
        elif frac >= threshold - 0.12:
            icon, cls = "⏳", "stage-active"
        else:
            icon, cls = "◌", "stage-pending"
        rows.append(f'<div class="stage-row {cls}"><span class="stage-icon">{icon}</span>{label}</div>')

    progress_placeholder.markdown(textwrap.dedent(f"""
    <style>
        .loader-card {{
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(0,255,157,0.18);
            border-radius: 18px;
            padding: 22px 26px;
            margin: 10px 0 20px 0;
            backdrop-filter: blur(16px);
            box-shadow: 0 8px 30px rgba(0,0,0,0.4);
        }}
        .loader-top {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; }}
        .loader-title {{ font-family:'Poppins',sans-serif; font-weight:700; color:#f5fff9; font-size:1rem; }}
        .loader-pct {{ color:{NEON_GREEN}; font-weight:800; font-size:1rem; }}
        .loader-track {{ width:100%; height:8px; border-radius:6px; background:rgba(255,255,255,0.06); overflow:hidden; margin-bottom:16px; }}
        .loader-fill {{ height:100%; border-radius:6px; background:linear-gradient(90deg,{NEON_GREEN},{NEON_CYAN});
                         width:{pct}%; transition: width 0.4s ease; box-shadow: 0 0 12px rgba(0,255,157,0.6); }}
        .stage-grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap:8px 16px; }}
        .stage-row {{ font-size:0.82rem; padding:4px 0; display:flex; align-items:center; gap:8px; transition: color 0.3s ease; }}
        .stage-icon {{ font-size:0.85rem; }}
        .stage-done {{ color:#9fe6c8; }}
        .stage-active {{ color:{NEON_CYAN}; font-weight:700; }}
        .stage-pending {{ color:#4d6459; }}
    </style>
    <div class="loader-card">
        <div class="loader-top">
            <span class="loader-title">⚙️ {msg}</span>
            <span class="loader-pct">{pct}%</span>
        </div>
        <div class="loader-track"><div class="loader-fill"></div></div>
        <div class="stage-grid">{''.join(rows)}</div>
    </div>
    """).strip(), unsafe_allow_html=True)


status_placeholder = st.empty()

if start_clicked and video_path is not None:
    st.session_state["processing"] = True
    st.session_state["error"] = None

    def progress_cb(frac: float, msg: str):
        render_stage_loader(frac, msg)

    try:
        device = resolve_device(device_choice)
    except AnalysisError as e:
        device = "cpu"
        status_placeholder.warning(str(e))

    try:
        render_stage_loader(0.05, "Loading models (first run only)...")
        models = get_models(device)

        pipeline_result = run_pipeline(
            video_path, models, confidence=confidence,
            compute_homography=compute_homography, progress_cb=progress_cb,
        )

        progress_cb(0.92, "Computing match statistics...")
        stats_cfg = cfg.StatsConfig(
            fps=pipeline_result.output_fps,
            frame_w=pipeline_result.video_info.width,
            frame_h=pipeline_result.video_info.height,
            pitch_visible_m=cfg.PITCH_VISIBLE_M,
        )
        frame_homographies = pipeline_result.frame_homographies if compute_homography else None
        results = run_full_analysis(pipeline_result.tracking_data, stats_cfg,
                                     frame_homographies=frame_homographies)

        progress_cb(0.95, "Rendering annotated output video...")
        output_path = os.path.join(cfg.OUTPUT_DIR, f"output_{st.session_state['session_id']}.mp4")
        render_annotated_video(pipeline_result, output_path,
                                progress_cb=lambda f, m: progress_cb(0.95 + 0.05 * f, m))

        st.session_state["results"] = results
        st.session_state["pipeline_result"] = pipeline_result
        st.session_state["stats_cfg"] = stats_cfg
        st.session_state["output_video_path"] = output_path

        progress_cb(1.0, "Analysis complete.")
        status_placeholder.success("Analysis complete!")

    except AnalysisError as e:
        st.session_state["error"] = str(e)
    except Exception as e:  # noqa: BLE001 — surface any unexpected error cleanly
        st.session_state["error"] = f"Unexpected error during processing: {e}"
    finally:
        st.session_state["processing"] = False

if st.session_state["error"]:
    st.error(st.session_state["error"])


results = st.session_state.get("results")
pipeline_result = st.session_state.get("pipeline_result")
stats_cfg = st.session_state.get("stats_cfg")
output_video_path = st.session_state.get("output_video_path")

if results and output_video_path and os.path.exists(output_video_path):
    st.markdown('<div class="section-title">Output Video</div>', unsafe_allow_html=True)
    v_col, dl_col = st.columns([3, 1])
    with v_col:
        st.video(output_video_path)
    with dl_col:
        st.write("")
        with open(output_video_path, "rb") as f:
            st.download_button("⬇ Download Processed Video", f, file_name="analyzed_match.mp4",
                                mime="video/mp4", use_container_width=True)
        report_text = build_match_report_text(results)
        st.download_button("⬇ Download Report (.txt)", report_text,
                            file_name="match_report.txt", mime="text/plain",
                            use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-title">Match Analytics Dashboard</div>', unsafe_allow_html=True)

    tab_overview, tab_passing, tab_players, tab_heatmaps, tab_report = st.tabs(
        ["📊 Overview", "🎯 Passing", "🏃 Player Stats", "🔥 Heatmaps", "📄 Full Report"]
    )

    with tab_overview:
        pos = results["possession_pct"]
        loose = results["possession_uncontrolled_pct"]
        pos_a, pos_b = pos.get("A", 0), pos.get("B", 0)

        st.markdown(
            f"{team_chip('Team A', TEAM_A_COLOR)}{team_chip('Team B', TEAM_B_COLOR)}",
            unsafe_allow_html=True
        )

        st.markdown(textwrap.dedent(f"""
        <div class="poss-wrap">
            <div class="poss-bar">
                <div class="poss-a" style="width:{max(pos_a, 2)}%">{pos_a}%</div>
                <div class="poss-b" style="width:{max(pos_b, 2)}%">{pos_b}%</div>
                {f'<div class="poss-loose" style="width:{loose}%">{loose}% loose</div>' if loose > 1 else ''}
            </div>
        </div>
        """).strip(), unsafe_allow_html=True)

        st.write("")

        c1, c2, c3 = st.columns(3)
        kpi_card(c1, "Match Time", f"{results['match_time_s']}s")
        bs = results["ball_speed"]
        kpi_card(c2, "Avg Ball Speed", f"{bs['avg_kmh']} km/h", f"Peak {bs['max_kmh']} km/h")
        attacks = results["attacks"]
        total_attacks = attacks.get("A", 0) + attacks.get("B", 0)
        kpi_card(c3, "Total Attacks", total_attacks)

        c1, c2, c3 = st.columns(3)
        kpi_card(c1, "Attacks — Team A", attacks.get("A", 0))
        kpi_card(c2, "Attacks — Team B", attacks.get("B", 0))
        sprints = results["sprints_per_team"]
        total_sprints = sprints.get("A", 0) + sprints.get("B", 0)
        kpi_card(c3, "Total Sprints", total_sprints)

        c1, c2 = st.columns(2)
        kpi_card(c1, "Sprints — Team A", sprints.get("A", 0))
        kpi_card(c2, "Sprints — Team B", sprints.get("B", 0))

        if not results.get("homography_calibrated", False):
            st.caption(
                "⚠️ No pitch homography computed — directional stats "
                "(progressive passes, through balls, crosses, attacks, shot angle) are approximate."
            )

    with tab_passing:
        p = results["passes"]

        def passing_block(col, team, color):
            with col:
                st.markdown(team_chip(f"Team {team}", color), unsafe_allow_html=True)
                att = p["attempts"].get(team, 0)
                comp = p["completions"].get(team, 0)
                acc = p["accuracy_pct"].get(team, 0)
                cc1, cc2, cc3 = st.columns(3)
                kpi_card(cc1, "Attempts", att)
                kpi_card(cc2, "Completed", comp)
                kpi_card(cc3, "Accuracy", f"{acc}%")
                st.progress(min(acc / 100, 1.0))
                cc1, cc2, cc3 = st.columns(3)
                kpi_card(cc1, "Progressive", p["progressive_count"].get(team, 0))
                kpi_card(cc2, "Through Balls", p["through_ball_count"].get(team, 0))
                kpi_card(cc3, "Crosses", p["cross_count"].get(team, 0))

        c1, c2 = st.columns(2)
        passing_block(c1, "A", TEAM_A_COLOR)
        passing_block(c2, "B", TEAM_B_COLOR)

        st.write("")
        st.markdown('<div class="section-title">Passing Network</div>', unsafe_allow_html=True)
        net = results["passing_network"]
        fig = plot_passing_network(net, stats_cfg.frame_w, stats_cfg.frame_h)
        if fig is not None:
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("Not enough completed passes yet to draw a passing network.")

    with tab_players:
        per_player = results["per_player"]
        touches = results["touches"]["per_player"]
        total_touches = sum(touches.values()) or 1

        rows = []
        for tid, d in sorted(per_player.items(), key=lambda x: x[1]["distance_m"], reverse=True):
            t = touches.get(int(tid), 0)
            rows.append({
                "Player": tid, "Team": d.get("team"), "Distance (m)": d["distance_m"],
                "Avg Speed (m/s)": d["avg_speed_ms"], "Max Speed (m/s)": d["max_speed_ms"],
                "Sprints": d["sprints"], "Touches": t,
                "Possession Share (%)": round(100 * t / total_touches, 1),
            })

        if rows:
            top = rows[0]
            c1, c2, c3 = st.columns(3)
            kpi_card(c1, "Most Distance Covered", f"{top['Player']} (Team {top['Team']})",
                     f"{top['Distance (m)']} m")
            fastest = max(rows, key=lambda r: r["Max Speed (m/s)"])
            kpi_card(c2, "Top Speed", f"{fastest['Player']} (Team {fastest['Team']})",
                     f"{fastest['Max Speed (m/s)']} m/s")
            most_touches = max(rows, key=lambda r: r["Touches"])
            kpi_card(c3, "Most Involved", f"{most_touches['Player']} (Team {most_touches['Team']})",
                     f"{most_touches['Touches']} touches")
            st.write("")

        team_filter = st.radio("Filter", ["All", "Team A", "Team B"], horizontal=True, label_visibility="collapsed")
        if team_filter == "Team A":
            rows = [r for r in rows if r["Team"] == "A"]
        elif team_filter == "Team B":
            rows = [r for r in rows if r["Team"] == "B"]

        st.dataframe(
            rows, use_container_width=True, hide_index=True,
            column_config={
                "Distance (m)": st.column_config.ProgressColumn(
                    "Distance (m)", min_value=0,
                    max_value=max((r["Distance (m)"] for r in rows), default=1) or 1, format="%.1f",
                ),
                "Possession Share (%)": st.column_config.ProgressColumn(
                    "Possession Share (%)", min_value=0, max_value=100, format="%.1f%%",
                ),
            },
        )

    with tab_heatmaps:
        transformer: CoordinateTransformer = results["transformer"]
        tracking_data = pipeline_result.tracking_data

        st.markdown('<div class="section-title">Team Heatmaps (top-down, homography-corrected)</div>',
                    unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(team_chip("Team A", TEAM_A_COLOR), unsafe_allow_html=True)
            fig_a = plot_heatmap(tracking_data, transformer, stats_cfg, team="A", title="Team A Heatmap")
            st.pyplot(fig_a, use_container_width=True)
        with c2:
            st.markdown(team_chip("Team B", TEAM_B_COLOR), unsafe_allow_html=True)
            fig_b = plot_heatmap(tracking_data, transformer, stats_cfg, team="B", title="Team B Heatmap")
            st.pyplot(fig_b, use_container_width=True)

    with tab_report:
        st.markdown(f'<div class="report-box">{render_report_html(report_text)}</div>',
                    unsafe_allow_html=True)
        st.write("")
        st.download_button("⬇ Download Report (.txt)", report_text,
                            file_name="match_report.txt", mime="text/plain",
                            key="report_tab_dl")

elif not uploaded_file:
    st.info("Upload a video and click **Start Analysis** in the sidebar to begin.")

render_footer()
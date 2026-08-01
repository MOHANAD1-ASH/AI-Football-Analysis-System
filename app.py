"""
AI Football Analysis — Streamlit application.

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import os
import uuid

import streamlit as st
import pandas as pd

import config as cfg
from services.detection import ModelBundle, run_pipeline
from services.event_engine import CoordinateTransformer
from services.match_statistics import UNAVAILABLE, build_match_report_text, run_full_analysis
from services.utils import AnalysisError, resolve_device
from services.visualization import plot_heatmap, plot_passing_network, render_annotated_video

st.set_page_config(page_title="AI Football Analysis", page_icon="", layout="wide")

TEAM_A_COLOR = "#e74c3c"
TEAM_B_COLOR = "#3498db"

# Dark theme palette — used both in CSS and passed to matplotlib figures
BG_MAIN = "#071a6b"
BG_PANEL = "#08286d"
BG_PANEL_2 = "#0D588A"
BORDER = "rgba(255,255,255,0.1)"
TEXT_PRIMARY = "#f4fff8"
TEXT_SECONDARY = "#050505"   # brighter than before for readability
TEXT_MUTED = "#D30819"
ACCENT = "#f8f8fa"

# ────────────────────────────────────────────────────────────
#  Global styling — dark analytics-dashboard theme
# ────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
    #MainMenu, footer {{visibility: hidden;}}

    html, body, .stApp, [class*="css"] {{
        color: {TEXT_SECONDARY};
    }}

    .stApp {{
        background: radial-gradient(circle at 15% 0%, {BG_PANEL_2} 0%, {BG_PANEL} 45%, {BG_MAIN} 100%);
    }}

    .hero {{
        background: linear-gradient(120deg, {BG_PANEL} 0%, {BG_PANEL_2} 55%, {BG_MAIN} 100%);
        border: 1px solid {BORDER};
        border-radius: 18px;
        padding: 28px 32px;
        margin-bottom: 22px;
        box-shadow: 0 8px 28px rgba(0,0,0,0.35);
    }}
    .hero h1 {{
        color: {TEXT_PRIMARY};
        font-size: 2.0rem;
        font-weight: 800;
        margin: 0 0 4px 0;
        letter-spacing: -0.5px;
    }}
    .hero p {{
        color: {TEXT_PRIMARY};
        margin: 0;
        font-size: 0.95rem;
    }}

    .kpi-card {{
        background: linear-gradient(160deg, {BG_PANEL_2} 0%, {BG_PANEL} 100%);
        border: 1px solid {BORDER};
        border-radius: 14px;
        padding: 16px 18px;
        height: 100%;
    }}
    .kpi-label {{
        color: {TEXT_MUTED};
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
    }}
    .kpi-value {{
        color: {TEXT_PRIMARY};
        font-size: 1.65rem;
        font-weight: 800;
        line-height: 1.1;
    }}
    .kpi-sub {{
        color: {TEXT_PRIMARY};
        font-size: 0.78rem;
        margin-top: 4px;
    }}

    .poss-wrap {{ margin: 6px 0 2px 0; }}
    .poss-bar {{
        display: flex; width: 100%; height: 34px; border-radius: 8px;
        overflow: hidden; font-weight: 700; font-size: 0.85rem;
        box-shadow: inset 0 0 0 1px rgba(255,255,255,0.08);
    }}
    .poss-a {{ background: {TEAM_A_COLOR}; display:flex; align-items:center; justify-content:center; color:white; }}
    .poss-b {{ background: {TEAM_B_COLOR}; display:flex; align-items:center; justify-content:center; color:white; }}
    .poss-loose {{ background: {BG_PANEL_2}; display:flex; align-items:center; justify-content:center; color:{TEXT_PRIMARY}; font-size:0.72rem; }}

    .team-chip {{
        display:inline-block; padding: 3px 10px; border-radius: 999px;
        font-size: 0.75rem; font-weight: 700; color: white; margin-right: 6px;
    }}

    .section-title {{
        color: {TEXT_PRIMARY}; font-size: 1.05rem; font-weight: 700;
        margin: 4px 0 12px 0; padding-left: 10px;
        border-left: 4px solid {ACCENT};
    }}

    /* Metrics */
    div[data-testid="stMetric"] {{
        background: linear-gradient(160deg, {BG_PANEL_2} 0%, {BG_PANEL} 100%);
        border: 1px solid {BORDER};
        border-radius: 14px;
        padding: 12px 16px 6px 16px;
    }}
    div[data-testid="stMetricLabel"] {{ color: {TEXT_MUTED} !important; }}
    div[data-testid="stMetricValue"] {{ color: {TEXT_PRIMARY} !important; }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}
    .stTabs [data-baseweb="tab"] {{
        background-color: rgba(255,255,255,0.04);
        border-radius: 10px 10px 0 0;
        padding: 8px 18px;
        color: {TEXT_SECONDARY};
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {BG_PANEL_2} !important;
        color: {TEXT_PRIMARY} !important;
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {BG_PANEL} 0%, {BG_MAIN} 100%);
        border-right: 1px solid {BORDER};
    }}
    section[data-testid="stSidebar"] * {{
        color: {TEXT_PRIMARY} !important;
    }}
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {{
        color: {TEXT_PRIMARY} !important;
    }}

    /* General text elements — fix low-contrast grey text */
    p, span, label, .stCaption, .stMarkdown, .stText {{
        color: {TEXT_SECONDARY};
    }}
    h1, h2, h3, h4, h5, h6 {{
        color: {TEXT_PRIMARY} !important;
    }}

    /* Info / warning / success boxes */
    div[data-testid="stAlert"] {{
        background: {BG_PANEL_2};
        border: 1px solid {BORDER};
        border-radius: 12px;
        color: {TEXT_SECONDARY};
    }}

    /* ── Dataframe / table styling ── */
    div[data-testid="stDataFrame"] {{
        background: {BG_PANEL_2};
        border: 1px solid {BORDER};
        border-radius: 14px;
        padding: 6px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.25);
    }}
    div[data-testid="stDataFrame"] * {{
        color: {TEXT_SECONDARY} !important;
    }}

    /* Buttons */
    .stButton > button, .stDownloadButton > button {{
        border-radius: 10px !important;
        border: 1px solid {BORDER} !important;
        background: {BG_PANEL_2} !important;
        color: {TEXT_PRIMARY} !important;
    }}
    .stButton > button:hover, .stDownloadButton > button:hover {{
        background: {BG_PANEL} !important;
        border: 1px solid {ACCENT} !important;
        color: {TEXT_PRIMARY} !important;
    }}
    .stButton > button p, .stDownloadButton > button p {{
        color: {TEXT_PRIMARY} !important;
    }}

    /* File uploader dropzone (Video Upload box) — covers every nesting level */
    [data-testid="stFileUploader"],
    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploaderDropzone"],
    [data-testid="stFileUploaderDropzoneInstructions"] {{
        background: {BG_PANEL_2} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 12px !important;
    }}
    [data-testid="stFileUploader"] * ,
    [data-testid="stFileUploaderDropzone"] *,
    [data-testid="stFileUploaderDropzoneInstructions"] *,
    [data-testid="stFileUploaderDropzoneInstructions"] span,
    [data-testid="stFileUploaderDropzoneInstructions"] small,
    [data-testid="stFileUploaderDropzoneInstructions"] div {{
        color: {TEXT_PRIMARY} !important;
        fill: {TEXT_PRIMARY} !important;
    }}
    [data-testid="stFileUploader"] button,
    [data-testid="stBaseButton-secondary"] {{
        background: {BG_PANEL} !important;
        color: {TEXT_PRIMARY} !important;
        border: 1px solid {BORDER} !important;
    }}

    /* Expander header (e.g. Team Names section) — covers every nesting level */
    [data-testid="stExpander"],
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] details,
    [data-testid="stExpanderDetails"],
    .streamlit-expanderHeader,
    .streamlit-expanderContent {{
        background: {BG_PANEL} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 10px !important;
    }}
    [data-testid="stExpander"] summary {{
        background: {BG_PANEL_2} !important;
    }}
    [data-testid="stExpander"] * {{
        color: {TEXT_PRIMARY} !important;
    }}

    /* Text inputs (Team A name / Team B name, etc.) — covers every nesting level */
    [data-testid="stTextInput"] > div,
    [data-testid="stTextInput"] div[data-baseweb="base-input"],
    div[data-baseweb="input"],
    div[data-baseweb="base-input"] {{
        background: {BG_PANEL_2} !important;
        border-radius: 8px !important;
        border: 1px solid {BORDER} !important;
    }}
    [data-testid="stTextInput"] input,
    div[data-baseweb="input"] input,
    div[data-baseweb="base-input"] input {{
        background: {BG_PANEL_2} !important;
        background-color: {BG_PANEL_2} !important;
        color: {TEXT_PRIMARY} !important;
        -webkit-text-fill-color: {TEXT_PRIMARY} !important;
        caret-color: {TEXT_PRIMARY} !important;
    }}
    [data-testid="stTextInput"] input::placeholder,
    div[data-baseweb="input"] input::placeholder {{
        color: {TEXT_PRIMARY} !important;
        -webkit-text-fill-color: {TEXT_PRIMARY} !important;
        opacity: 0.55 !important;
    }}

    /* Matplotlib figure containers — give them a dark frame so any
       white figure background reads as an inset panel, not a stray box */
    div[data-testid="stImage"], .stPyplot {{
        background: {BG_PANEL_2};
        border: 1px solid {BORDER};
        border-radius: 14px;
        padding: 10px;
    }}
</style>
""", unsafe_allow_html=True)


def kpi_card(col, label, value, sub=""):
    with col:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)


def team_chip(label, color):
    return f'<span class="team-chip" style="background:{color}">{label}</span>'


def _build_player_rows(results: dict) -> list[dict]:
    """Builds the per-player table rows (distance, speed, sprints, touches,
    possession share) shared by the Player Stats tab and the Full Report
    dashboard, sorted by distance covered (descending)."""
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
    return rows


# ────────────────────────────────────────────────────────────
#  Cached model loading — loaded once per (device) session
# ────────────────────────────────────────────────────────────
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
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session_state()


# ────────────────────────────────────────────────────────────
#  Sidebar
# ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Football Analysis")
    st.caption("Upload tactical camera match footage to run detection, tracking, and analytics.")
    st.markdown("---")

    uploaded_file = st.file_uploader(
        " Video Upload", type=list(cfg.SUPPORTED_VIDEO_FORMATS),
        help="Supported formats: " + ", ".join(cfg.SUPPORTED_VIDEO_FORMATS),
    )

    with st.expander("Team Names", expanded=True):
        team_a_name = st.text_input("Team A name", value="", placeholder="Team A",
                                     help="Leave blank to keep the default 'Team A'.")
        team_b_name = st.text_input("Team B name", value="", placeholder="Team B",
                                     help="Leave blank to keep the default 'Team B'.")
    team_names = {
        "A": team_a_name.strip() or "Team A",
        "B": team_b_name.strip() or "Team B",
    }
    st.session_state["team_names"] = team_names

    with st.expander("Analysis Settings", expanded=True):
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
        show_minimap = st.checkbox(
            "Show mini-map overlay on output video", value=True,
            help="Burns a small radar-style top-down pitch (player + ball "
                 "positions) into the bottom-center of the output video.",
        )

    start_clicked = st.button("▶  Start Analysis", type="primary",
                               use_container_width=True, disabled=uploaded_file is None)

    st.markdown("---")
    st.caption(f"{team_names['A']}  VS {team_names['B']} ")


# ────────────────────────────────────────────────────────────
#  Header
# ────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>AI Football Match Analysis</h1>
    <p>Computer-vision powered detection, tracking, homography and professional-grade match analytics.</p>
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

progress_bar_placeholder = st.empty()
status_placeholder = st.empty()

if start_clicked and video_path is not None:
    st.session_state["processing"] = True
    st.session_state["error"] = None
    progress_bar = progress_bar_placeholder.progress(0.0)

    def progress_cb(frac: float, msg: str):
        progress_bar.progress(frac)
        status_placeholder.info(msg)

    try:
        device = resolve_device(device_choice)
    except AnalysisError as e:
        device = "cpu"
        status_placeholder.warning(str(e))

    try:
        with st.spinner("Loading models (first run only)..."):
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
                                progress_cb=lambda f, m: progress_cb(0.95 + 0.05 * f, m),
                                team_names=team_names,
                                transformer=results["transformer"],
                                stats_cfg=stats_cfg,
                                show_minimap=show_minimap)

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


# ────────────────────────────────────────────────────────────
#  Results dashboard
# ────────────────────────────────────────────────────────────
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
        report_text = build_match_report_text(results, team_names=team_names)
        st.download_button("⬇ Download Report (.txt)", report_text,
                            file_name="match_report.txt", mime="text/plain",
                            use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-title">Match Analytics Dashboard</div>', unsafe_allow_html=True)

    tab_overview, tab_passing, tab_players, tab_heatmaps, tab_report = st.tabs(
        ["Overview", "Passing", "Player Stats", "Heatmaps", "Full Report"]
    )

    # ── Overview ────────────────────────────────────────────
    with tab_overview:
        pos = results["possession_pct"]
        loose = results["possession_uncontrolled_pct"]
        pos_a, pos_b = pos.get("A", 0), pos.get("B", 0)

        st.markdown(f"{team_chip(team_names['A'], TEAM_A_COLOR)}{team_chip(team_names['B'], TEAM_B_COLOR)}",
                    unsafe_allow_html=True)

        st.markdown(f"""
        <div class="poss-wrap">
        <div class="poss-bar">
            <div class="poss-a" style="width:{max(pos_a, 2)}%">{pos_a}%</div>
            <div class="poss-b" style="width:{max(pos_b, 2)}%">{pos_b}%</div>
            {f'<div class="poss-loose" style="width:{loose}%">{loose}% loose</div>' if loose > 1 else ''}
        </div>
        </div>
        """, unsafe_allow_html=True)

        st.write("")

        # Row 1 → Match Time + Ball Speed
        col1, col2 = st.columns(2)
        kpi_card(col1, "Match Time", f"{results['match_time_s']}s")
        bs = results["ball_speed"]
        kpi_card(col2, "Avg Ball Speed", f"{bs['avg_kmh']} km/h", f"peak {bs['max_kmh']} km/h")

        # Row 2 → Attacks
        col1, col2 = st.columns(2)
        attacks = results["attacks"]
        kpi_card(col1, f"Attacks — {team_names['A']}", attacks.get("A", 0))
        kpi_card(col2, f"Attacks — {team_names['B']}", attacks.get("B", 0))

        # Row 3 → Sprints
        col1, col2 = st.columns(2)
        sprints = results["sprints_per_team"]
        kpi_card(col1, f"Sprints — {team_names['A']}", sprints.get("A", 0))
        kpi_card(col2, f"Sprints — {team_names['B']}", sprints.get("B", 0))

        if not results.get("homography_calibrated", False):
            st.caption("No pitch homography computed — directional stats (progressive "
                       "passes, through balls, crosses, attacks, shot angle) are approximate.")

    # ── Passing ─────────────────────────────────────────────
    with tab_passing:
        p = results["passes"]

        def passing_block(col, team, color):
            with col:
                st.markdown(team_chip(team_names[team], color), unsafe_allow_html=True)
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

    # ── Player Stats ────────────────────────────────────────
    with tab_players:
        rows = _build_player_rows(results)

        if rows:
            top = rows[0]
            c1, c2, c3 = st.columns(3)
            kpi_card(c1, "Most Distance Covered", f"{top['Player']} ({team_names[top['Team']]})",
                     f"{top['Distance (m)']} m")
            fastest = max(rows, key=lambda r: r["Max Speed (m/s)"])
            kpi_card(c2, "Top Speed", f"{fastest['Player']} ({team_names[fastest['Team']]})",
                     f"{fastest['Max Speed (m/s)']} m/s")
            most_touches = max(rows, key=lambda r: r["Touches"])
            kpi_card(c3, "Most Involved", f"{most_touches['Player']} ({team_names[most_touches['Team']]})",
                     f"{most_touches['Touches']} touches")
            st.write("")

        team_filter = st.radio("Filter", ["All", team_names["A"], team_names["B"]],
                                horizontal=True, label_visibility="collapsed")
        if team_filter == team_names["A"]:
            rows = [r for r in rows if r["Team"] == "A"]
        elif team_filter == team_names["B"]:
            rows = [r for r in rows if r["Team"] == "B"]

        st.markdown('<div class="section-title">Player Table</div>', unsafe_allow_html=True)
        st.dataframe(
            rows, use_container_width=True, hide_index=True,
            column_config={
                "Player": st.column_config.NumberColumn("Player", format="%d"),
                "Distance (m)": st.column_config.ProgressColumn(
                    "Distance (m)", min_value=0,
                    max_value=max((r["Distance (m)"] for r in rows), default=1) or 1, format="%.1f",
                ),
                "Avg Speed (m/s)": st.column_config.NumberColumn("Avg Speed (m/s)", format="%.2f"),
                "Max Speed (m/s)": st.column_config.NumberColumn("Max Speed (m/s)", format="%.2f"),
                "Possession Share (%)": st.column_config.ProgressColumn(
                    "Possession Share (%)", min_value=0, max_value=100, format="%.1f%%",
                ),
            },
        )

        # ── Distance comparison line chart (dark-theme native) ──
        if rows:
            st.write("")
            st.markdown('<div class="section-title">Distance Covered — by Player</div>',
                        unsafe_allow_html=True)
            chart_df = {str(r["Player"]): r["Distance (m)"] for r in rows}
            st.line_chart(chart_df, height=280, use_container_width=True)

    # ── Heatmaps ────────────────────────────────────────────
    with tab_heatmaps:
        transformer: CoordinateTransformer = results["transformer"]
        tracking_data = pipeline_result.tracking_data

        st.markdown('<div class="section-title">Team Heatmaps (top-down, homography-corrected)</div>',
                    unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(team_chip(team_names["A"], TEAM_A_COLOR), unsafe_allow_html=True)
            fig_a = plot_heatmap(tracking_data, transformer, stats_cfg, team="A",
                                  title=f"{team_names['A']} Heatmap")
            st.pyplot(fig_a, use_container_width=True)
        with c2:
            st.markdown(team_chip(team_names["B"], TEAM_B_COLOR), unsafe_allow_html=True)
            fig_b = plot_heatmap(tracking_data, transformer, stats_cfg, team="B",
                                  title=f"{team_names['B']} Heatmap")
            st.pyplot(fig_b, use_container_width=True)

    # ── Full Report (dashboard) ──────────────────────────────
    with tab_report:
        pos = results["possession_pct"]
        loose = results["possession_uncontrolled_pct"]
        pos_a, pos_b = pos.get("A", 0), pos.get("B", 0)
        p = results["passes"]
        shots = results["shots"]
        attacks = results["attacks"]
        sprints = results["sprints_per_team"]
        bs = results["ball_speed"]

        st.markdown('<div class="section-title">Match Summary</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        kpi_card(col1, "Match Time", f"{results['match_time_s']}s")
        kpi_card(col2, "Avg Ball Speed", f"{bs['avg_kmh']} km/h", f"peak {bs['max_kmh']} km/h")
        kpi_card(col3, "Loose / Uncontrolled", f"{loose}%")

        st.markdown(f"""
        <div class="poss-wrap">
        <div class="poss-bar">
            <div class="poss-a" style="width:{max(pos_a, 2)}%">{team_names['A']} {pos_a}%</div>
            <div class="poss-b" style="width:{max(pos_b, 2)}%">{team_names['B']} {pos_b}%</div>
            {f'<div class="poss-loose" style="width:{loose}%">{loose}% loose</div>' if loose > 1 else ''}
        </div>
        </div>
        """, unsafe_allow_html=True)

        if not results.get("homography_calibrated", False):
            st.caption("No pitch homography computed — directional stats (progressive "
                       "passes, through balls, crosses, attacks, shot angle) are approximate.")

        st.write("")
        st.markdown('<div class="section-title">Team Comparison</div>', unsafe_allow_html=True)

        def team_summary_block(col, team, color):
            with col:
                st.markdown(team_chip(team_names[team], color), unsafe_allow_html=True)
                cc1, cc2 = st.columns(2)
                kpi_card(cc1, "Possession", f"{pos.get(team, 0)}%")
                kpi_card(cc2, "Attacks", attacks.get(team, 0))
                cc1, cc2 = st.columns(2)
                att, comp = p["attempts"].get(team, 0), p["completions"].get(team, 0)
                kpi_card(cc1, "Passes", f"{comp}/{att}", f"{p['accuracy_pct'].get(team, 0)}% accuracy")
                kpi_card(cc2, "Sprints", sprints.get(team, 0))
                cc1, cc2 = st.columns(2)
                kpi_card(cc1, "Shots", shots["per_team"].get(team, 0),
                         f"on target: {shots['on_target_per_team'].get(team, UNAVAILABLE)}")
                kpi_card(cc2, "Progressive / Through / Crosses",
                         f"{p['progressive_count'].get(team, 0)} / "
                         f"{p['through_ball_count'].get(team, 0)} / "
                         f"{p['cross_count'].get(team, 0)}")

        c1, c2 = st.columns(2)
        team_summary_block(c1, "A", TEAM_A_COLOR)
        team_summary_block(c2, "B", TEAM_B_COLOR)

        st.write("")
        st.markdown('<div class="section-title">Team Comparison — Chart</div>', unsafe_allow_html=True)
        compare_df = pd.DataFrame({
            "Shots": [shots["per_team"].get("A", 0), shots["per_team"].get("B", 0)],
            "Attacks": [attacks.get("A", 0), attacks.get("B", 0)],
            "Sprints": [sprints.get("A", 0), sprints.get("B", 0)],
            "Completed Passes": [p["completions"].get("A", 0), p["completions"].get("B", 0)],
        }, index=[team_names["A"], team_names["B"]])
        st.bar_chart(compare_df, height=300, use_container_width=True)

        st.write("")
        st.markdown('<div class="section-title">All Players</div>', unsafe_allow_html=True)
        report_rows = _build_player_rows(results)
        st.dataframe(
            report_rows, use_container_width=True, hide_index=True,
            column_config={
                "Player": st.column_config.NumberColumn("Player", format="%d"),
                "Distance (m)": st.column_config.ProgressColumn(
                    "Distance (m)", min_value=0,
                    max_value=max((r["Distance (m)"] for r in report_rows), default=1) or 1, format="%.1f",
                ),
                "Avg Speed (m/s)": st.column_config.NumberColumn("Avg Speed (m/s)", format="%.2f"),
                "Max Speed (m/s)": st.column_config.NumberColumn("Max Speed (m/s)", format="%.2f"),
                "Possession Share (%)": st.column_config.ProgressColumn(
                    "Possession Share (%)", min_value=0, max_value=100, format="%.1f%%",
                ),
            },
        )

        st.write("")
        st.download_button("⬇ Download Report (.txt)", report_text,
                            file_name="match_report.txt", mime="text/plain",
                            key="report_tab_dl")

elif not uploaded_file:
    st.info("Upload a video and click **Start Analysis** in the sidebar to begin.")
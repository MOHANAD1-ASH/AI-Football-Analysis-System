"""
AI Football Analysis — Streamlit application.

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import os
import shutil
import time
import uuid

import streamlit as st

import config as cfg
from services.detection import ModelBundle, run_pipeline
from services.event_engine import CoordinateTransformer
from services.match_statistics import build_match_report_text, run_full_analysis
from services.pitch_detection import HomographyStore
from services.utils import AnalysisError, resolve_device
from services.visualization import plot_heatmap, plot_passing_network, render_annotated_video

st.set_page_config(page_title="AI Football Analysis", page_icon="⚽", layout="wide")


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
    st.title("⚽ Football Analysis")
    st.caption("Upload match footage to run detection, tracking, and analytics.")

    uploaded_file = st.file_uploader(
        "Video Upload", type=list(cfg.SUPPORTED_VIDEO_FORMATS),
        help="Supported formats: " + ", ".join(cfg.SUPPORTED_VIDEO_FORMATS),
    )

    confidence = st.slider(
        "Confidence Threshold", min_value=0.05, max_value=0.90,
        value=cfg.DEFAULT_CONFIDENCE, step=0.05,
        help="Minimum detection confidence for the player/referee model.",
    )

    device_choice = st.radio("Device", options=["cpu", "cuda"], index=0,
                              help="CUDA requires an available GPU.")

    compute_homography = st.checkbox(
        "Compute pitch homography", value=True,
        help="Enables real-world pitch-metre coordinates for more accurate "
             "directional statistics and heatmaps. Slower to process.",
    )

    start_clicked = st.button("▶ Start Analysis", type="primary",
                               use_container_width=True, disabled=uploaded_file is None)


# ────────────────────────────────────────────────────────────
#  Main page
# ────────────────────────────────────────────────────────────
st.title("AI Football Match Analysis")

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
        st.subheader("Uploaded Video")
        st.video(video_path)

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


# ────────────────────────────────────────────────────────────
#  Results dashboard
# ────────────────────────────────────────────────────────────
results = st.session_state.get("results")
pipeline_result = st.session_state.get("pipeline_result")
stats_cfg = st.session_state.get("stats_cfg")
output_video_path = st.session_state.get("output_video_path")

if results and output_video_path and os.path.exists(output_video_path):
    st.divider()
    st.subheader("Output Video")
    st.video(output_video_path)
    with open(output_video_path, "rb") as f:
        st.download_button("⬇ Download Processed Video", f, file_name="analyzed_match.mp4",
                            mime="video/mp4", use_container_width=False)

    st.divider()
    st.subheader("Match Analytics Dashboard")

    tab_overview, tab_passing, tab_players, tab_heatmaps, tab_report = st.tabs(
        ["Overview", "Passing", "Player Stats", "Heatmaps", "Full Report"]
    )

    with tab_overview:
        c1, c2, c3, c4 = st.columns(4)
        pos = results["possession_pct"]
        c1.metric("Possession — Team A", f"{pos.get('A', 0)}%")
        c2.metric("Possession — Team B", f"{pos.get('B', 0)}%")
        c3.metric("Uncontrolled", f"{results['possession_uncontrolled_pct']}%")
        c4.metric("Match Time", f"{results['match_time_s']} s")

        c1, c2, c3, c4 = st.columns(4)
        shots = results["shots"]["per_team"]
        c1.metric("Shots — Team A", shots.get("A", 0))
        c2.metric("Shots — Team B", shots.get("B", 0))
        c3.metric("Shots on Target — A", results["shots"]["on_target_per_team"].get("A"))
        c4.metric("Shots on Target — B", results["shots"]["on_target_per_team"].get("B"))

        c1, c2, c3, c4 = st.columns(4)
        rt = results["recovery_and_turnover"]
        c1.metric("Recoveries — A", rt["recoveries"].get("A", 0))
        c2.metric("Recoveries — B", rt["recoveries"].get("B", 0))
        c3.metric("Turnovers — A", rt["turnovers"].get("A", 0))
        c4.metric("Turnovers — B", rt["turnovers"].get("B", 0))

        c1, c2 = st.columns(2)
        bs = results["ball_speed"]
        c1.metric("Avg Ball Speed", f"{bs['avg_kmh']} km/h")
        c2.metric("Max Ball Speed", f"{bs['max_kmh']} km/h")

        if not results.get("homography_calibrated", False):
            st.caption("⚠️ No pitch homography computed — directional stats (progressive "
                       "passes, through balls, crosses, attacks, shot angle) are approximate.")

    with tab_passing:
        p = results["passes"]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Team A**")
            st.write(f"Attempts: {p['attempts'].get('A', 0)}")
            st.write(f"Completions: {p['completions'].get('A', 0)}")
            st.write(f"Accuracy: {p['accuracy_pct'].get('A', 0)}%")
            st.write(f"Progressive: {p['progressive_count'].get('A', 0)}")
            st.write(f"Through balls: {p['through_ball_count'].get('A', 0)}")
            st.write(f"Crosses: {p['cross_count'].get('A', 0)}")
        with c2:
            st.markdown("**Team B**")
            st.write(f"Attempts: {p['attempts'].get('B', 0)}")
            st.write(f"Completions: {p['completions'].get('B', 0)}")
            st.write(f"Accuracy: {p['accuracy_pct'].get('B', 0)}%")
            st.write(f"Progressive: {p['progressive_count'].get('B', 0)}")
            st.write(f"Through balls: {p['through_ball_count'].get('B', 0)}")
            st.write(f"Crosses: {p['cross_count'].get('B', 0)}")

        net = results["passing_network"]
        fig = plot_passing_network(net, stats_cfg.frame_w, stats_cfg.frame_h)
        if fig is not None:
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("Not enough completed passes yet to draw a passing network.")

    with tab_players:
        per_player = results["per_player"]
        rows = []
        for tid, d in sorted(per_player.items(), key=lambda x: x[1]["distance_m"], reverse=True):
            rows.append({
                "Player ID": tid, "Team": d.get("team"), "Distance (m)": d["distance_m"],
                "Avg Speed (m/s)": d["avg_speed_ms"], "Max Speed (m/s)": d["max_speed_ms"],
                "Sprints": d["sprints"], "Touches": results["touches"]["per_player"].get(int(tid), 0),
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)

    with tab_heatmaps:
        transformer: CoordinateTransformer = results["transformer"]
        tracking_data = pipeline_result.tracking_data

        c1, c2 = st.columns(2)
        with c1:
            fig_a = plot_heatmap(tracking_data, transformer, stats_cfg, team="A", title="Team A Heatmap")
            st.pyplot(fig_a, use_container_width=True)
        with c2:
            fig_b = plot_heatmap(tracking_data, transformer, stats_cfg, team="B", title="Team B Heatmap")
            st.pyplot(fig_b, use_container_width=True)

        st.markdown("**Individual Player Heatmap**")
        player_ids = sorted({e["track_id"] for e in tracking_data if e["class"] == "player"})
        if player_ids:
            selected_pid = st.selectbox("Select a player ID", player_ids)
            fig_p = plot_heatmap(tracking_data, transformer, stats_cfg, track_id=selected_pid,
                                  title=f"Player #{selected_pid} Heatmap")
            st.pyplot(fig_p, use_container_width=True)

    with tab_report:
        report_text = build_match_report_text(results)
        st.text(report_text)
        st.download_button("⬇ Download Report (.txt)", report_text,
                            file_name="match_report.txt", mime="text/plain")

elif not uploaded_file:
    st.info("Upload a video and click **Start Analysis** in the sidebar to begin.")

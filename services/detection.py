"""
Main detection + tracking pipeline.

Two-model architecture (preserved from the notebook):
  - Model 1 (player/goalkeeper/referee): `.track()` + ByteTrack.
  - Model 2 (ball-only): `.predict()`, multi-candidate collection per
    frame, later fed into the Kalman tracker in a second lightweight
    pass (no re-inference needed).

Also performs:
  - Team classification via KMeans on jersey-region HSV colour.
  - Optional pitch-keypoint homography (sampled every N frames, held/
    interpolated in between via HomographyStore).
  - Player-ID stitching + noise-track removal.
  - Construction of the `tracking_data` schema consumed by the Event
    Engine: {"frame", "track_id", "class", "team", "center"}.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from sklearn.cluster import KMeans
from ultralytics import YOLO

import config as cfg
from services.ball_tracker import BallKalmanTracker, find_static_false_positive_zones, interpolate_ball
from services.id_stitching import apply_id_remap, drop_noise_tracks, normalize_team_per_track, stitch_player_ids
from services.pitch_detection import HomographyStore, ViewTransformer, compute_frame_homography
from services.utils import AnalysisError, VideoInfo, check_model_weights, probe_video

ProgressCallback = Optional[Callable[[float, str], None]]


class ModelBundle:
    """Loads and caches the three YOLO models. Streamlit's own
    `st.cache_resource` wraps a factory that returns this, so weights
    are loaded exactly once per session."""

    def __init__(self, device: str = "cpu"):
        check_model_weights(cfg.PLAYER_MODEL_PATH, cfg.BALL_MODEL_PATH, cfg.PITCH_MODEL_PATH)
        self.device = device
        self.player_model = YOLO(cfg.PLAYER_MODEL_PATH)
        self.ball_model = YOLO(cfg.BALL_MODEL_PATH)
        self.pitch_model = YOLO(cfg.PITCH_MODEL_PATH)
        self.player_model.to(device)
        self.ball_model.to(device)
        self.pitch_model.to(device)
        self.player_keep_ids = [i for i, n in self.player_model.names.items() if n != "ball"]


class PipelineResult:
    """Everything downstream stages (statistics, heatmaps, video export)
    need."""

    def __init__(self):
        self.frame_data: List[tuple] = []          # (frame, player_list, referee_list, ball_box)
        self.tracking_data: List[dict] = []
        self.ball_smoothed: Dict[int, Tuple[float, float]] = {}
        self.ball_smoothed_raw: Dict[int, Tuple[float, float]] = {}
        self.frame_homographies: Dict[int, Optional[ViewTransformer]] = {}
        self.video_info: Optional[VideoInfo] = None
        self.output_fps: float = 25.0
        self.n_processed_frames: int = 0


def run_pipeline(
    video_path: str,
    models: ModelBundle,
    confidence: float = cfg.DEFAULT_CONFIDENCE,
    compute_homography: bool = True,
    progress_cb: ProgressCallback = None,
) -> PipelineResult:
    """Runs the full two-pass detection + tracking pipeline on a video
    file and returns a PipelineResult ready for statistics / rendering."""

    def report(frac: float, msg: str):
        if progress_cb:
            progress_cb(min(max(frac, 0.0), 1.0), msg)

    video_info = probe_video(video_path)
    result = PipelineResult()
    result.video_info = video_info

    video = cv2.VideoCapture(video_path)
    frame_w, frame_h = video_info.width, video_info.height
    total_frames = video_info.total_frames
    output_fps = (video_info.fps or 25.0) / cfg.FRAME_SKIP
    result.output_fps = output_fps

    frame_data = []
    all_colors = []
    frame_ball_candidates: Dict[int, list] = {}
    n_ball_det = 0
    n_referee_det = 0
    frame_homographies: Dict[int, Optional[ViewTransformer]] = {}
    last_homography: Optional[ViewTransformer] = None

    report(0.0, "Starting detection pass...")

    raw_idx = -1
    for raw_idx in range(total_frames):
        ret, frame = video.read()
        if not ret:
            break

        if raw_idx % cfg.FRAME_SKIP != 0:
            continue

        frame_idx = raw_idx // cfg.FRAME_SKIP

        # ── Model 1: players / goalkeeper / referee ─────────────────────
        results_p = models.player_model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
            imgsz=cfg.PLAYER_INFER_SIZE,
            conf=confidence,
            iou=cfg.PLAYER_IOU,
            max_det=cfg.PLAYER_MAX_DET,
            classes=models.player_keep_ids,
        )

        player_list = []
        referee_list = []

        for box in results_p[0].boxes:
            cls = int(box.cls[0])
            name = models.player_model.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            tid = int(box.id[0]) if (box.id is not None and len(box.id) > 0) else -1

            if name in ("player", "goalkeeper"):
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                h, w = crop.shape[:2]
                jersey = crop[int(h * 0.15):int(h * 0.55), int(w * 0.15):int(w * 0.85)]
                if jersey.size == 0:
                    jersey = crop
                hsv = cv2.cvtColor(jersey, cv2.COLOR_BGR2HSV)
                all_colors.append(hsv.mean(axis=(0, 1)))
                player_list.append((x1, y1, x2, y2, tid, name))

            elif name == "referee":
                n_referee_det += 1
                referee_list.append((x1, y1, x2, y2, tid, name))

        # ── Model 2: ball-only — collect candidates ─────────────────────
        results_b = models.ball_model.predict(
            frame,
            verbose=False,
            imgsz=cfg.BALL_INFER_SIZE,
            conf=cfg.BALL_CONF,
            iou=cfg.BALL_IOU,
            max_det=cfg.BALL_MAX_DET,
        )

        candidates_this_frame = []
        ball_box = None
        for box in results_b[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf_b = float(box.conf[0])
            w, h = (x2 - x1), (y2 - y1)
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            size = float(np.hypot(w, h))

            aspect = max(w, h) / max(min(w, h), 1e-6)
            if aspect > cfg.MAX_BALL_ASPECT_RATIO:
                continue

            near_referee = any(
                (rx1 - cfg.REFEREE_EXCLUSION_PAD_PX <= cx <= rx2 + cfg.REFEREE_EXCLUSION_PAD_PX and
                 ry1 - cfg.REFEREE_EXCLUSION_PAD_PX <= cy <= ry2 + cfg.REFEREE_EXCLUSION_PAD_PX)
                for (rx1, ry1, rx2, ry2, rtid, rname) in referee_list
            )
            if near_referee:
                continue

            candidates_this_frame.append((cx, cy, conf_b, size))
            n_ball_det += 1
            if ball_box is None:
                ball_box = (x1, y1, x2, y2)

        frame_ball_candidates[frame_idx] = candidates_this_frame
        frame_data.append((frame.copy(), player_list, referee_list, ball_box))

        # ── Optional pitch homography (sampled every N frames) ──────────
        if compute_homography:
            if frame_idx % cfg.PITCH_SAMPLE_EVERY == 0:
                vt = compute_frame_homography(
                    models.pitch_model, frame,
                    infer_size=cfg.PITCH_INFER_SIZE,
                    detect_conf=cfg.PITCH_DETECT_CONF,
                    conf_threshold=cfg.PITCH_CONF_THRESHOLD,
                )
                if vt is not None:
                    last_homography = vt
                frame_homographies[frame_idx] = vt
            else:
                frame_homographies[frame_idx] = None

        if frame_idx % 25 == 0:
            report(0.05 + 0.55 * (raw_idx / max(total_frames, 1)),
                   f"Detecting players & ball — frame {raw_idx}/{total_frames}")

    video.release()
    n_processed_frames = len(frame_data)
    result.n_processed_frames = n_processed_frames

    if n_processed_frames == 0:
        raise AnalysisError("No frames could be read from the video — it may be corrupted or empty.")

    report(0.62, "Filtering static false-positive ball detections...")

    # ── Static false-positive scan (fixes e.g. penalty-spot misdetection) ──
    all_candidates_flat = [
        (fi, cx, cy, conf_b)
        for fi, cands in frame_ball_candidates.items()
        for (cx, cy, conf_b, size) in cands
    ]
    blacklist = find_static_false_positive_zones(
        all_candidates_flat,
        frame_count=n_processed_frames,
        cluster_radius_px=cfg.BLACKLIST_CLUSTER_RADIUS_PX,
        min_presence_ratio=cfg.BLACKLIST_MIN_PRESENCE,
    )

    report(0.65, "Tracking ball trajectory (Kalman filter)...")

    ball_tracker = BallKalmanTracker(
        fps=output_fps,
        base_gate_px=220.0,
        max_lost_frames=int(output_fps * 1.5),
        blacklist_zones=blacklist,
        confirm_frames=2,
        confirm_speed_budget_px=90.0,
        reinit_min_conf=0.25,
        confirm_min_conf=0.30,
        frame_w=frame_w,
        pitch_visible_m=cfg.PITCH_VISIBLE_M,
        max_speed_kmh=130.0,
        coast_visible_frames=0,
    )
    for fi in sorted(frame_ball_candidates.keys()):
        ball_tracker.update(fi, frame_ball_candidates[fi])

    # ── KMeans team classification ──────────────────────────────────────
    report(0.72, "Classifying teams by jersey colour...")
    if len(all_colors) >= 2:
        colors_arr = np.array(all_colors, dtype=float)
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = kmeans.fit_predict(colors_arr)
    else:
        labels = np.zeros(len(all_colors), dtype=int)

    expected_label_count = sum(len(player_list) for (_, player_list, _, _) in frame_data)
    if expected_label_count != len(labels):
        raise AnalysisError(
            "Internal team-label alignment mismatch during processing — please retry the analysis."
        )

    # ── Ball smoothing + interpolation ──────────────────────────────────
    ball_smoothed_raw = ball_tracker.smoothed_positions(window=5)
    ball_smoothed = interpolate_ball(ball_smoothed_raw, max_gap=cfg.BALL_INTERP_MAX_GAP)
    result.ball_smoothed = ball_smoothed
    result.ball_smoothed_raw = ball_smoothed_raw
    result.frame_data = frame_data
    result.frame_homographies = frame_homographies

    # ── Build tracking_data (Event-Engine input format) ─────────────────
    report(0.78, "Building tracking data...")
    tracking_data = []
    label_idx = 0
    for fi, (_, player_list, referee_list, _) in enumerate(frame_data):
        for (x1, y1, x2, y2, tid, cls_name) in player_list:
            team = "A" if int(labels[label_idx]) == 0 else "B"
            label_idx += 1
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            tracking_data.append({"frame": fi, "track_id": tid, "class": "player", "team": team, "center": [cx, cy]})

        for (x1, y1, x2, y2, tid, cls_name) in referee_list:
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            tracking_data.append({"frame": fi, "track_id": tid, "class": "referee", "team": None, "center": [cx, cy]})

        if fi in ball_smoothed:
            bcx, bcy = ball_smoothed[fi]
            tracking_data.append({"frame": fi, "track_id": -1, "class": "ball", "team": None,
                                   "center": [float(bcx), float(bcy)]})

    # ── ID stitching + team normalization + noise removal ──────────────
    report(0.85, "Stitching player track IDs...")
    remap = stitch_player_ids(
        tracking_data, fps=output_fps,
        pixel_to_meter=cfg.PITCH_VISIBLE_M / frame_w,
        max_gap_seconds=cfg.ID_STITCH_MAX_GAP_SECONDS,
        position_gate_meters=cfg.ID_STITCH_POSITION_GATE_M,
    )
    tracking_data = apply_id_remap(tracking_data, remap)
    tracking_data = normalize_team_per_track(tracking_data)
    tracking_data = drop_noise_tracks(tracking_data, min_frames=cfg.ID_STITCH_MIN_SEGMENT_FRAMES)

    result.tracking_data = tracking_data
    report(0.90, "Detection and tracking complete.")
    return result

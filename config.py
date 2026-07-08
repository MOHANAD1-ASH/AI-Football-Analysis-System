"""
Central configuration for the Football Analysis application.

Every tunable knob used across the pipeline lives here so nothing is
hardcoded deep inside the services. Update model paths after placing
your trained weights in the corresponding `models/` sub-folders.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Model weight paths (relative, configurable) ─────────────────────────────
PLAYER_MODEL_PATH = os.path.join(BASE_DIR, "models", "player_detector", "best.pt")
BALL_MODEL_PATH = os.path.join(BASE_DIR, "models", "ball_detector", "best.pt")
PITCH_MODEL_PATH = os.path.join(BASE_DIR, "models", "pitch_detector", "best.pt")

# ── Output / temp directories ───────────────────────────────────────────────
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
TEMP_DIR = os.path.join(BASE_DIR, "temp")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

# ── Performance knobs (mirrors the notebook's tuned pipeline) ───────────────
FRAME_SKIP = 2
PLAYER_INFER_SIZE = 960
PLAYER_IOU = 0.45
PLAYER_MAX_DET = 50

BALL_INFER_SIZE = 1280
BALL_CONF = 0.05
BALL_IOU = 0.45
BALL_MAX_DET = 5

PITCH_INFER_SIZE = 1280
PITCH_DETECT_CONF = 0.1
PITCH_CONF_THRESHOLD = 0.3
# Only run the (heavier) pitch-keypoint model every N processed frames and
# hold/interpolate homography in between (see services.pitch_detection).
PITCH_SAMPLE_EVERY = 5

PITCH_VISIBLE_M = 50.0
BALL_INTERP_MAX_GAP = 25

REFEREE_EXCLUSION_PAD_PX = 45.0
MAX_BALL_ASPECT_RATIO = 1.8

BLACKLIST_CLUSTER_RADIUS_PX = 15.0
BLACKLIST_MIN_PRESENCE = 0.35

# ── ID stitching ─────────────────────────────────────────────────────────────
ID_STITCH_MAX_GAP_SECONDS = 3.6
ID_STITCH_POSITION_GATE_M = 5.5
ID_STITCH_MIN_SEGMENT_FRAMES = 8

# ── Default confidence for player/ball detection (overridable from the UI) ──
DEFAULT_CONFIDENCE = 0.30

# ── Supported video formats ──────────────────────────────────────────────────
SUPPORTED_VIDEO_FORMATS = ("mp4", "avi", "mov", "mkv")

# ── Team colors (BGR) used for on-video annotation ──────────────────────────
TEAM_COLORS = {0: (220, 60, 60), 1: (60, 60, 220)}
BALL_COLOR = (0, 220, 220)
REF_COLOR = (0, 255, 255)


@dataclass
class StatsConfig:
    """Configuration for the event-based match-statistics engine.

    Kept as a dataclass (mirroring the notebook) so it can be constructed
    per-video with the actual measured fps / resolution.
    """

    fps: float = 25.0
    frame_w: int = 1280
    frame_h: int = 720
    pitch_visible_m: float = 50.0
    pitch_length_m: float = 105.0
    pitch_width_m: float = 68.0
    attack_direction: Dict[str, int] = field(default_factory=lambda: {"A": 1, "B": -1})

    possession_gain_radius_px: float = 120.0
    possession_lose_radius_px: float = 180.0
    possession_min_frames: int = 3
    possession_max_ball_gap_frames: int = 10
    possession_owner_occlusion_grace_frames: int = 5

    pass_max_duration_s: float = 3.0
    pass_min_travel_px: float = 30.0

    max_ball_speed_ms: float = 55.0
    max_player_speed_ms: float = 11.0
    motion_max_frame_gap: int = 10
    position_smoothing_window: int = 5
    ball_smoothing_window: int = 5

    sprint_speed_threshold_ms: float = 7.0
    sprint_min_duration_s: float = 1.0

    attacking_third_fraction: float = 1.0 / 3.0
    attack_min_possession_frames: int = 5
    attack_cooldown_s: float = 3.0

    progressive_pass_min_reduction_fraction: float = 0.25
    progressive_pass_min_length_m: float = 8.0

    through_ball_requires_forward: bool = True

    cross_wide_zone_fraction: float = 0.20
    penalty_area_length_fraction: float = 0.17
    penalty_area_width_fraction: float = 0.62

    shot_min_speed_ms: float = 15.0
    shot_goal_cone_deg: float = 20.0

    @property
    def pixel_to_meter(self) -> float:
        return self.pitch_visible_m / self.frame_w

    @property
    def pixel_to_meter_y(self) -> float:
        return self.pitch_width_m / self.frame_h

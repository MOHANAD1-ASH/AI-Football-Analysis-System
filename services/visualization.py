"""
Visualization: annotated output video rendering, and professional
football-analytics-style heatmaps / passing network plots.

Heatmaps use a Gaussian-smoothed density surface (instead of scattered
raw points) rendered over a properly-scaled, correctly-proportioned
pitch outline, matching the look of professional analytics platforms.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

import config as cfg
from services.detection import PipelineResult
from services.event_engine import CoordinateTransformer
from services.match_statistics import compute_heatmap

PITCH_GREEN = "#0b3d24"
PITCH_LINE = "#011A03"


_MAX_LABEL_CHARS = 4  # keeps the "{name}#{id}" label from crowding small boxes

# ── Mini-map (radar) overlay ─────────────────────────────────────────────────
# Pure-cv2 drawing (no matplotlib) so it stays fast enough to run once per
# video frame. Placed bottom-center, semi-transparent, over the main feed.
_MINIMAP_WIDTH_FRAC = 0.20     # width of the mini-map as a fraction of frame width
_MINIMAP_MAX_WIDTH_PX = 260    # hard cap so it stays small even on large/4K frames
_MINIMAP_MARGIN_PX = 14        # gap from the bottom edge of the frame
_MINIMAP_PAD_M = 2.5           # padding (in pitch metres) around the pitch outline
_MINIMAP_BG_COLOR = (20, 60, 30)     # BGR dark pitch-green background
_MINIMAP_LINE_COLOR = (200, 230, 200)  # BGR light line color
_MINIMAP_BALL_COLOR = (0, 220, 220)  # BGR — matches cfg.BALL_COLOR
_MINIMAP_ALPHA = 0.85           # opacity when compositing onto the frame


def _build_minimap_canvas(pitch_length_m: float, pitch_width_m: float,
                           canvas_w: int) -> Tuple[np.ndarray, callable]:
    """Builds a blank BGR mini-pitch canvas sized to `canvas_w`, and returns
    (canvas, to_canvas_xy) where to_canvas_xy(x_m, y_m) -> (px, py) maps a
    pitch-metre coordinate to a pixel position on that canvas."""
    span_x = pitch_length_m + 2 * _MINIMAP_PAD_M
    span_y = pitch_width_m + 2 * _MINIMAP_PAD_M
    scale = canvas_w / span_x
    canvas_h = max(int(round(span_y * scale)), 1)

    canvas = np.full((canvas_h, canvas_w, 3), _MINIMAP_BG_COLOR, dtype=np.uint8)

    def to_canvas_xy(x_m: float, y_m: float) -> Tuple[int, int]:
        px = int(round((x_m + _MINIMAP_PAD_M) * scale))
        py = int(round((y_m + _MINIMAP_PAD_M) * scale))
        return px, py

    # Pitch outline
    x0, y0 = to_canvas_xy(0, 0)
    x1, y1 = to_canvas_xy(pitch_length_m, pitch_width_m)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), _MINIMAP_LINE_COLOR, 1)
    # Halfway line
    xm0, _ = to_canvas_xy(pitch_length_m / 2, 0)
    xm1, _ = to_canvas_xy(pitch_length_m / 2, pitch_width_m)
    cv2.line(canvas, (xm0, y0), (xm1, y1), _MINIMAP_LINE_COLOR, 1)
    # Centre circle
    cx, cy = to_canvas_xy(pitch_length_m / 2, pitch_width_m / 2)
    cv2.circle(canvas, (cx, cy), max(int(9.15 * scale), 2), _MINIMAP_LINE_COLOR, 1)

    return canvas, to_canvas_xy


def _composite_minimap(frame: np.ndarray, minimap: np.ndarray) -> None:
    """Alpha-blends `minimap` onto the bottom-center of `frame`, in place."""
    fh, fw = frame.shape[:2]
    mh, mw = minimap.shape[:2]
    x0 = (fw - mw) // 2
    y0 = fh - mh - _MINIMAP_MARGIN_PX
    if x0 < 0 or y0 < 0 or x0 + mw > fw or y0 + mh > fh:
        return  # frame too small for the configured mini-map size — skip safely
    roi = frame[y0:y0 + mh, x0:x0 + mw]
    blended = cv2.addWeighted(minimap, _MINIMAP_ALPHA, roi, 1 - _MINIMAP_ALPHA, 0)
    frame[y0:y0 + mh, x0:x0 + mw] = blended


def _video_team_label(team_names: Optional[Dict[str, str]], team_code: Optional[str]) -> str:
    """Short label used on the video overlay: team display name truncated
    to a few characters (e.g. "Real Madrid" -> "Real"). Falls back to the
    raw "A"/"B" code when no team_names mapping is supplied, so existing
    callers keep working unchanged."""
    if not team_code:
        return "?"
    if not team_names:
        return team_code
    name = team_names.get(team_code) or team_code
    return name[:_MAX_LABEL_CHARS]


_LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
_LABEL_FONT_SCALE = 0.5
_LABEL_THICKNESS = 2            # bold-looking stroke for the colored text
_LABEL_OUTLINE_COLOR = (0, 0, 0)
_LABEL_OUTLINE_THICKNESS = 4    # thicker black pass drawn underneath, for contrast


def _draw_bold_label(frame: np.ndarray, text: str, origin: Tuple[int, int],
                      color: Tuple[int, int, int]) -> None:
    """Draws `text` twice — a thick black outline pass first, then the
    colored text on top with a heavier stroke — so the label stays
    readable regardless of what's behind it (dark kit, grass, etc.)."""
    cv2.putText(frame, text, origin, _LABEL_FONT, _LABEL_FONT_SCALE,
                _LABEL_OUTLINE_COLOR, _LABEL_OUTLINE_THICKNESS, cv2.LINE_AA)
    cv2.putText(frame, text, origin, _LABEL_FONT, _LABEL_FONT_SCALE,
                color, _LABEL_THICKNESS, cv2.LINE_AA)


def render_annotated_video(result: PipelineResult, output_path: str,
                            progress_cb=None,
                            team_names: Optional[Dict[str, str]] = None,
                            transformer: Optional[CoordinateTransformer] = None,
                            stats_cfg=None,
                            show_minimap: bool = True) -> str:
    """Draws team-colored boxes, referee boxes, and the tracked ball onto
    every processed frame, and writes the annotated output video.

    team_names: optional {"A": "...", "B": "..."} display-name override
    for the on-video label. Truncated to a few characters per team so the
    label doesn't crowd the bounding box. Defaults to the raw "A"/"B"
    code when not supplied.

    transformer / stats_cfg: when both are supplied AND show_minimap is
    True, a bottom-center radar-style mini-map is burned into every frame,
    showing each player's and the ball's position projected onto a
    top-down pitch outline (pixel -> pitch metres via `transformer`).
    When either is missing, the mini-map is silently skipped so existing
    callers keep working unchanged.
    """
    frame_data = result.frame_data
    ball_smoothed = result.ball_smoothed
    ball_smoothed_raw = result.ball_smoothed_raw

    if not frame_data:
        raise RuntimeError("No frames available to render.")

    height, width = frame_data[0][0].shape[:2]
    out = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), result.output_fps, (width, height)
    )

    draw_minimap = bool(show_minimap and transformer is not None and stats_cfg is not None)
    base_minimap = None
    if draw_minimap:
        minimap_w = min(max(int(width * _MINIMAP_WIDTH_FRAC), 40), _MINIMAP_MAX_WIDTH_PX)
        base_minimap, minimap_xy = _build_minimap_canvas(
            pitch_length_m=stats_cfg.pitch_length_m,
            pitch_width_m=stats_cfg.pitch_width_m,
            canvas_w=minimap_w,
        )

    # Recover each player's assigned team from tracking_data (already
    # stitched + normalized) for consistent coloring across frames.
    # tracking_data's track_id is the CANONICAL (post-stitching) id, while
    # player_list below still holds the RAW (pre-stitching) id, so lookups
    # must go through id_remap first or they'll miss for every player whose
    # id got merged/renamed during stitching.
    team_by_frame_track: Dict[Tuple[int, int], str] = {}
    for e in result.tracking_data:
        if e["class"] == "player":
            team_by_frame_track[(e["frame"], e["track_id"])] = e["team"]

    id_remap = result.id_remap
    valid_track_ids = result.valid_track_ids
    UNKNOWN_COLOR = (160, 160, 160)  # neutral gray — do NOT default to Team B

    total = len(frame_data)
    for fi, (frame, player_list, referee_list, ball_box) in enumerate(frame_data):
        minimap = base_minimap.copy() if draw_minimap else None

        for (x1, y1, x2, y2, tid, cls_name) in player_list:
            canonical_tid = id_remap.get(tid, tid)
            # tid == -1 (no ByteTrack id this frame) or a track that was
            # dropped as noise: we genuinely don't know who this is, so
            # mark it unknown instead of guessing.
            if tid == -1 or canonical_tid not in valid_track_ids:
                cv2.rectangle(frame, (x1, y1), (x2, y2), UNKNOWN_COLOR, 1)
                cv2.putText(frame, "?", (x1, max(y1 - 5, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, UNKNOWN_COLOR, 1)
                continue

            team = team_by_frame_track.get((fi, canonical_tid))
            team_idx = 0 if team == "A" else 1
            color = cfg.TEAM_COLORS[team_idx]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = _video_team_label(team_names, team)
            _draw_bold_label(frame, f"{label}#{canonical_tid}", (x1, max(y1 - 5, 10)), color)

            if draw_minimap:
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                x_m, y_m = transformer.to_pitch(cx, cy, frame=fi)
                px, py = minimap_xy(x_m, y_m)
                if 0 <= px < minimap.shape[1] and 0 <= py < minimap.shape[0]:
                    cv2.circle(minimap, (px, py), 3, color, -1)

        for (x1, y1, x2, y2, tid, cls_name) in referee_list:
            cv2.rectangle(frame, (x1, y1), (x2, y2), cfg.REF_COLOR, 2)
            cv2.putText(frame, f"REF#{tid}", (x1, max(y1 - 5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, cfg.REF_COLOR, 1)

        if fi in ball_smoothed:
            bcx, bcy = ball_smoothed[fi]
            is_interp = fi not in ball_smoothed_raw
            if is_interp:
                cv2.circle(frame, (int(bcx), int(bcy)), 6, cfg.BALL_COLOR, -1)
            else:
                cv2.circle(frame, (int(bcx), int(bcy)), 10, cfg.BALL_COLOR, 2)
            cv2.putText(frame, "ball", (int(bcx) + 12, int(bcy) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, cfg.BALL_COLOR, 1)
            if draw_minimap:
                bx_m, by_m = transformer.to_pitch(bcx, bcy, frame=fi)
                bpx, bpy = minimap_xy(bx_m, by_m)
                if 0 <= bpx < minimap.shape[1] and 0 <= bpy < minimap.shape[0]:
                    cv2.circle(minimap, (bpx, bpy), 3, _MINIMAP_BALL_COLOR, -1)
        elif ball_box is not None:
            bx1, by1, bx2, by2 = ball_box
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), cfg.BALL_COLOR, 2)

        if draw_minimap:
            _composite_minimap(frame, minimap)

        out.write(frame)
        if progress_cb and fi % 25 == 0:
            progress_cb(fi / max(total, 1), f"Rendering annotated video — frame {fi}/{total}")

    out.release()
    return output_path


def _draw_pitch(ax, pitch_length_m: float, pitch_width_m: float) -> None:
    """Draws a clean, proportionally-correct pitch outline."""
    ax.set_facecolor(PITCH_GREEN)
    lw = 1.6

    # Outer boundary
    ax.plot([0, pitch_length_m, pitch_length_m, 0, 0],
             [0, 0, pitch_width_m, pitch_width_m, 0], color=PITCH_LINE, linewidth=lw)
    # Halfway line
    ax.plot([pitch_length_m / 2, pitch_length_m / 2], [0, pitch_width_m], color=PITCH_LINE, linewidth=lw)
    # Centre circle
    circle = plt.Circle((pitch_length_m / 2, pitch_width_m / 2), 9.15,
                         color=PITCH_LINE, fill=False, linewidth=lw)
    ax.add_patch(circle)
    ax.scatter([pitch_length_m / 2], [pitch_width_m / 2], color=PITCH_LINE, s=8)

    # Penalty boxes (18-yard, ~16.5m x 40.3m) + 6-yard boxes, both ends
    box_len, box_w = 16.5, 40.32
    goal_box_len, goal_box_w = 5.5, 18.32
    for x0, direction in ((0, 1), (pitch_length_m, -1)):
        bx = x0 + direction * box_len
        by0 = (pitch_width_m - box_w) / 2
        by1 = (pitch_width_m + box_w) / 2
        ax.plot([x0, bx, bx, x0], [by0, by0, by1, by1], color=PITCH_LINE, linewidth=lw)

        gx = x0 + direction * goal_box_len
        gy0 = (pitch_width_m - goal_box_w) / 2
        gy1 = (pitch_width_m + goal_box_w) / 2
        ax.plot([x0, gx, gx, x0], [gy0, gy0, gy1, gy1], color=PITCH_LINE, linewidth=lw)

    ax.set_xlim(-2, pitch_length_m + 2)
    ax.set_ylim(-2, pitch_width_m + 2)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


_HEAT_CMAP = LinearSegmentedColormap.from_list(
    "football_heat",
    ["#0b3d2400", "#1b5e3c", "#2e8b57", "#f1c40f", "#e67e22", "#e74c3c"],
)


def plot_heatmap(tracking_data: List[dict], transformer: CoordinateTransformer, stats_cfg,
                  team: Optional[str] = None, track_id: Optional[int] = None,
                  title: str = "Heatmap", sigma: float = 1.6) -> plt.Figure:
    """Renders a smooth, professional football-style heatmap: a Gaussian-
    blurred density surface over a proportionally-correct pitch, instead
    of scattered raw points."""
    hm = compute_heatmap(tracking_data, transformer, stats_cfg, track_id=track_id, team=team,
                          bins=(48, 32))
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    pitch_length_m = stats_cfg.pitch_length_m if stats_cfg.pitch_visible_m >= stats_cfg.pitch_length_m else stats_cfg.pitch_visible_m
    pitch_width_m = stats_cfg.frame_h * stats_cfg.pixel_to_meter_y

    _draw_pitch(ax, pitch_length_m=stats_cfg.pitch_visible_m, pitch_width_m=pitch_width_m)

    if hm["hist"]:
        density = gaussian_filter(np.array(hm["hist"]), sigma=sigma)
        if density.max() > 0:
            density = density / density.max()
        ax.imshow(
            density, origin="lower", cmap=_HEAT_CMAP, aspect="auto", alpha=0.85,
            extent=[0, stats_cfg.pitch_visible_m, 0, pitch_width_m],
            interpolation="bicubic", zorder=2,
        )

    ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=12)
    fig.patch.set_facecolor("#0b1f14")
    plt.tight_layout()
    return fig


def plot_passing_network(net: dict, frame_w: int, frame_h: int, title: str = "Passing Network") -> Optional[plt.Figure]:
    """Node = mean pass-involvement position, edge width proportional to
    pass count."""
    node_positions = net.get("node_positions", {})
    edges = net.get("edges", [])
    if not node_positions or not edges:
        return None

    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    max_w = max(e["weight"] for e in edges)
    for e in edges:
        src, dst, w = e["from"], e["to"], e["weight"]
        if src not in node_positions or dst not in node_positions:
            continue
        x1, y1 = node_positions[src]
        x2, y2 = node_positions[dst]
        ax.plot([x1, x2], [y1, y2], color="#3498db", linewidth=0.6 + 4.0 * (w / max_w),
                alpha=0.65, zorder=1, solid_capstyle="round")

    for node, (x, y) in node_positions.items():
        ax.scatter([x], [y], s=520, color="white", edgecolors="#2980b9", linewidths=2, zorder=2)
        ax.annotate(f"P{node}", (x, y), ha="center", va="center", fontsize=9,
                    fontweight="bold", zorder=3, color="#1a1a1a")

    ax.set_xlim(0, frame_w)
    ax.set_ylim(frame_h, 0)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_facecolor("#f4f6f7")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    return fig

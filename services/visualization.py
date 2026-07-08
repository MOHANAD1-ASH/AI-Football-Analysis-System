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
PITCH_LINE = "#e8f5e9"


def render_annotated_video(result: PipelineResult, output_path: str,
                            progress_cb=None) -> str:
    """Draws team-colored boxes, referee boxes, and the tracked ball onto
    every processed frame, and writes the annotated output video."""
    frame_data = result.frame_data
    ball_smoothed = result.ball_smoothed
    ball_smoothed_raw = result.ball_smoothed_raw

    if not frame_data:
        raise RuntimeError("No frames available to render.")

    height, width = frame_data[0][0].shape[:2]
    out = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), result.output_fps, (width, height)
    )

    # Recover each player's assigned team from tracking_data (already
    # stitched + normalized) for consistent coloring across frames.
    team_by_frame_track: Dict[Tuple[int, int], str] = {}
    for e in result.tracking_data:
        if e["class"] == "player":
            team_by_frame_track[(e["frame"], e["track_id"])] = e["team"]

    total = len(frame_data)
    for fi, (frame, player_list, referee_list, ball_box) in enumerate(frame_data):
        for (x1, y1, x2, y2, tid, cls_name) in player_list:
            team = team_by_frame_track.get((fi, tid))
            team_idx = 0 if team == "A" else 1
            color = cfg.TEAM_COLORS[team_idx]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{team or '?'}#{tid}", (x1, max(y1 - 5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

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
        elif ball_box is not None:
            bx1, by1, bx2, by2 = ball_box
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), cfg.BALL_COLOR, 2)

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

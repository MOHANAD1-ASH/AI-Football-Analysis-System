"""
Pitch keypoint detection, homography estimation, and coordinate
transformation from camera pixels to real pitch metres.

The vertex layout and ViewTransformer are the battle-tested homography
code used in the original notebook (adapted from the `roboflow/sports`
project) — logic unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class SoccerPitchConfiguration:
    width: int = 7000    # cm
    length: int = 12000   # cm
    penalty_box_width: int = 4100
    penalty_box_length: int = 2015
    goal_box_width: int = 1832
    goal_box_length: int = 550
    centre_circle_radius: int = 915
    penalty_spot_distance: int = 1100

    @property
    def vertices(self) -> List[Tuple[float, float]]:
        return [
            (0, 0), (0, (self.width - self.penalty_box_width) / 2),
            (0, (self.width - self.goal_box_width) / 2),
            (0, (self.width + self.goal_box_width) / 2),
            (0, (self.width + self.penalty_box_width) / 2), (0, self.width),
            (self.goal_box_length, (self.width - self.goal_box_width) / 2),
            (self.goal_box_length, (self.width + self.goal_box_width) / 2),
            (self.penalty_spot_distance, self.width / 2),
            (self.penalty_box_length, (self.width - self.penalty_box_width) / 2),
            (self.penalty_box_length, (self.width - self.goal_box_width) / 2),
            (self.penalty_box_length, (self.width + self.goal_box_width) / 2),
            (self.penalty_box_length, (self.width + self.penalty_box_width) / 2),
            (self.length / 2, 0),
            (self.length / 2, self.width / 2 - self.centre_circle_radius),
            (self.length / 2, self.width / 2 + self.centre_circle_radius),
            (self.length / 2, self.width),
            (self.length - self.penalty_box_length, (self.width - self.penalty_box_width) / 2),
            (self.length - self.penalty_box_length, (self.width - self.goal_box_width) / 2),
            (self.length - self.penalty_box_length, (self.width + self.goal_box_width) / 2),
            (self.length - self.penalty_box_length, (self.width + self.penalty_box_width) / 2),
            (self.length - self.penalty_spot_distance, self.width / 2),
            (self.length - self.goal_box_length, (self.width - self.goal_box_width) / 2),
            (self.length - self.goal_box_length, (self.width + self.goal_box_width) / 2),
            (self.length, 0), (self.length, (self.width - self.penalty_box_width) / 2),
            (self.length, (self.width - self.goal_box_width) / 2),
            (self.length, (self.width + self.goal_box_width) / 2),
            (self.length, (self.width + self.penalty_box_width) / 2), (self.length, self.width),
            (self.length / 2 - self.centre_circle_radius, self.width / 2),
            (self.length / 2 + self.centre_circle_radius, self.width / 2),
        ]


CONFIG = SoccerPitchConfiguration()


class ViewTransformer:
    """Homography-based transform from source (pitch keypoints in the
    camera frame) to target (real-world pitch coordinates)."""

    def __init__(self, source: np.ndarray, target: np.ndarray):
        if source.shape != target.shape or source.shape[1] != 2:
            raise ValueError("source/target must have the same shape and be 2D points")
        self.m, _ = cv2.findHomography(source.astype(np.float32), target.astype(np.float32))

    def transform_points(self, points: np.ndarray) -> Optional[np.ndarray]:
        if self.m is None or points.size == 0:
            return None
        reshaped = points.reshape(-1, 1, 2).astype(np.float32)
        out = cv2.perspectiveTransform(reshaped, self.m)
        return out.reshape(-1, 2)


def get_best_pitch_instance(result, conf_threshold: float = 0.3):
    """If the pitch-keypoint model returns more than one instance, pick the
    one with the highest mean keypoint confidence."""
    kpts = result.keypoints
    if kpts is None or kpts.xy.shape[0] == 0:
        return None, None

    if kpts.xy.shape[0] == 1:
        return kpts.xy[0].cpu().numpy(), kpts.conf[0].cpu().numpy()

    best_idx = 0
    best_mean_conf = -1.0
    for i in range(kpts.xy.shape[0]):
        mean_conf = float(kpts.conf[i].mean())
        if mean_conf > best_mean_conf:
            best_mean_conf = mean_conf
            best_idx = i

    return kpts.xy[best_idx].cpu().numpy(), kpts.conf[best_idx].cpu().numpy()


def compute_frame_homography(model_pitch, frame: np.ndarray, infer_size: int = 1280,
                              detect_conf: float = 0.1,
                              conf_threshold: float = 0.3) -> Optional[ViewTransformer]:
    """Runs the pitch-keypoint model on a single frame and returns a
    ViewTransformer, or None if not enough confident keypoints were found."""
    result = model_pitch.predict(frame, verbose=False, imgsz=infer_size, conf=detect_conf)[0]
    xy, conf = get_best_pitch_instance(result, conf_threshold=conf_threshold)

    if xy is None:
        return None

    mask = conf >= conf_threshold
    if mask.sum() < 4:
        return None

    target_vertices = np.array(CONFIG.vertices, dtype=np.float32)
    return ViewTransformer(source=xy[mask], target=target_vertices[mask])


class HomographyStore:
    """Manages per-frame homographies with temporal smoothing (reduces
    jitter from natural frame-to-frame variation in visible pitch
    keypoints) and gap-holding (reuses the nearest valid homography for
    frames where none was computed, instead of falling back abruptly to
    the linear approximation)."""

    def __init__(self, frame_homographies: Dict[int, Optional[ViewTransformer]],
                 smoothing_window: int = 5, max_hold_frames: int = 15):
        self.max_hold_frames = max_hold_frames
        valid_frames = sorted(
            fi for fi, vt in frame_homographies.items()
            if vt is not None and getattr(vt, "m", None) is not None
        )
        self._matrices: Dict[int, np.ndarray] = {}
        if not valid_frames:
            return
        raw = {fi: frame_homographies[fi].m for fi in valid_frames}
        half = smoothing_window // 2
        for idx, fi in enumerate(valid_frames):
            s, e = max(0, idx - half), min(len(valid_frames), idx + half + 1)
            near = [f for f in valid_frames[s:e] if abs(f - fi) <= max_hold_frames]
            self._matrices[fi] = np.mean([raw[f] for f in near], axis=0)

    def get(self, frame: int) -> Optional[np.ndarray]:
        if not self._matrices:
            return None
        if frame in self._matrices:
            return self._matrices[frame]
        nearby = [fi for fi in self._matrices if abs(fi - frame) <= self.max_hold_frames]
        if not nearby:
            return None
        nearest = min(nearby, key=lambda fi: abs(fi - frame))
        return self._matrices[nearest]

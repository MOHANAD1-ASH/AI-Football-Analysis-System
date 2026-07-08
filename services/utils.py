"""Small shared utilities: device selection, video probing, error helpers."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import cv2


class AnalysisError(Exception):
    """Raised for any user-facing, recoverable pipeline error (missing
    weights, corrupted video, unsupported format, etc.) so the Streamlit
    layer can show a clean message instead of a traceback."""


@dataclass
class VideoInfo:
    fps: float
    width: int
    height: int
    total_frames: int


def probe_video(video_path: str) -> VideoInfo:
    if not os.path.exists(video_path):
        raise AnalysisError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise AnalysisError(
            "Could not open the uploaded video. The file may be corrupted "
            "or in an unsupported codec/container."
        )

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if width <= 0 or height <= 0:
        raise AnalysisError("The video has invalid dimensions — it may be corrupted.")
    if total_frames <= 0:
        raise AnalysisError("Could not determine the video's frame count — it may be corrupted.")

    return VideoInfo(fps=fps, width=width, height=height, total_frames=total_frames)


def resolve_device(requested: str) -> str:
    """Validates the requested device ('cpu' or 'cuda'), falling back to
    CPU with a clear message if CUDA was requested but unavailable."""
    if requested == "cpu":
        return "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        raise AnalysisError(
            "CUDA was selected but no GPU is available in this environment. "
            "Falling back to CPU (processing will be slower)."
        )
    except ImportError:
        raise AnalysisError("PyTorch is not installed correctly; cannot use CUDA. Falling back to CPU.")


def check_model_weights(*paths: str) -> None:
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        names = "\n".join(f"  - {p}" for p in missing)
        raise AnalysisError(
            "Missing model weight file(s):\n" + names +
            "\n\nPlace your trained .pt files at these paths (see README) before running analysis."
        )

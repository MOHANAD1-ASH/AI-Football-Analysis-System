# AI Football Analysis

A production-ready Streamlit application for football (soccer) match video
analysis: player/ball/referee detection, multi-object tracking, team
classification, pitch-keypoint homography, an event-based match-statistics
engine, and professional-style heatmaps.

This app is a modular refactor of an original research notebook — the
detection, Kalman ball-tracking, ID-stitching, and event-engine logic are
preserved, cleaned up, and wired into a maintainable Streamlit UI.

## Features

- **Two-model detection pipeline**: a player/goalkeeper/referee model
  (tracked with ByteTrack) and a dedicated ball-only model.
- **Kalman-filter ball tracking** (v9) with multi-candidate gating, static
  false-positive blacklisting (e.g. penalty-spot misdetections), and
  gap-aware interpolation.
- **Team classification** via KMeans clustering on jersey HSV colour.
- **Pitch-keypoint homography** for real-world pitch-metre coordinates
  (optional, improves directional-stat accuracy).
- **Player-ID stitching** to heal ByteTrack ID discontinuities using the
  Hungarian algorithm.
- **Event-based match statistics engine**: possession, passes (with
  progressive/through-ball/cross classification), shots, recoveries,
  turnovers, attacks, distance/speed/sprints, ball speed. Any statistic
  that can't be reliably computed from the available data (e.g. shots on
  target) is reported as `Unavailable` rather than estimated.
- **Professional heatmaps**: Gaussian-smoothed density surfaces on a
  correctly-scaled pitch outline, for teams and individual players.
- **Passing network** visualization.
- Cached model loading, clear error handling, and a downloadable annotated
  output video + text report.

## Installation

```bash
git clone <this-repo>
cd football_analysis_app
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

If you have a CUDA-capable GPU, install the matching CUDA build of PyTorch
from https://pytorch.org/get-started/locally/ before installing the rest of
`requirements.txt`.

## Folder Structure

```text
football_analysis_app/
│
├── app.py                     # Streamlit entry point
├── config.py                  # All configurable paths & thresholds
├── requirements.txt
├── README.md
│
├── models/
│   ├── player_detector/best.pt   # player/goalkeeper/referee YOLO model
│   ├── ball_detector/best.pt     # dedicated ball-only YOLO model
│   └── pitch_detector/best.pt    # pitch-keypoint YOLO-pose model
│
├── services/
│   ├── detection.py            # main two-model detection/tracking pipeline
│   ├── ball_tracker.py         # Kalman ball tracker
│   ├── id_stitching.py         # player-ID stitching / noise removal
│   ├── pitch_detection.py      # pitch config, homography, ViewTransformer
│   ├── event_engine.py         # possession FSM + event generation
│   ├── match_statistics.py     # all statistics + orchestrator
│   ├── visualization.py        # annotated video + heatmaps + passing network
│   └── utils.py                # device selection, video probing, errors
│
├── outputs/                    # processed videos land here
├── assets/
└── temp/                       # uploaded videos (scratch space)
```

## Where to Place Model Weights

Place your trained `.pt` files exactly here (filenames matter — `config.py`
points to these paths):

```
models/player_detector/best.pt   # classes: ball(unused), goalkeeper, player, referee
models/ball_detector/best.pt     # single class: ball
models/pitch_detector/best.pt    # YOLO-pose pitch-keypoint model
```

To use different filenames or locations, edit `PLAYER_MODEL_PATH`,
`BALL_MODEL_PATH`, and `PITCH_MODEL_PATH` in `config.py`.

## How to Run

```bash
streamlit run app.py
```

Then, in the browser UI:
1. Upload a match video (mp4/avi/mov/mkv) in the sidebar.
2. Adjust the confidence threshold and device (CPU/CUDA) if needed.
3. Click **Start Analysis**.
4. Watch the progress bar; once complete, view the annotated output video,
   download it, and explore the analytics dashboard (possession, passing,
   per-player stats, heatmaps, and a full text report).

## Notes on Accuracy

- Directional statistics (progressive passes, through balls, crosses,
  attacks, shot angle) are most accurate when **pitch homography** is
  enabled, since it maps camera pixels to real pitch metres. Without it,
  a linear pixel-to-metre approximation is used and these stats are
  labeled as approximate.
- "Shots on target" cannot be reliably inferred without goal-frame/save
  detection and is reported as `Unavailable` rather than guessed.
- Processing a full match on CPU is slow; a CUDA GPU is strongly
  recommended for anything beyond short clips.

## Example Usage

```bash
streamlit run app.py
# In the browser:
#   1. Upload sample_match.mp4
#   2. Confidence: 0.30, Device: cuda
#   3. Click "Start Analysis"
#   4. Download analyzed_match.mp4 and match_report.txt from the dashboard
```

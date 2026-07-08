"""
Player-ID stitching.

Heals ByteTrack ID discontinuities BEFORE any distance/speed/pass
statistic is computed, by matching a track's end-state (position +
extrapolated velocity) against candidate tracks that start shortly
after, within a plausible running distance, using the Hungarian
algorithm for a globally-optimal (not greedy) assignment.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import numpy as np
from scipy.optimize import linear_sum_assignment


def build_player_track_segments(tracking_data: List[dict]) -> Dict[int, List[dict]]:
    """track_id -> its points, sorted by frame."""
    tracks = defaultdict(list)
    for e in tracking_data:
        if e.get("class") == "player":
            tracks[e["track_id"]].append(e)
    for tid in tracks:
        tracks[tid].sort(key=lambda e: e["frame"])
    return dict(tracks)


def _segment_end_state(pts: List[dict], n_pts_for_velocity: int = 5):
    """Last position + approximate velocity (px/frame) from the last n points."""
    tail = pts[-n_pts_for_velocity:]
    if len(tail) < 2:
        return np.asarray(pts[-1]["center"], dtype=float), np.zeros(2), pts[-1]["frame"]
    p0, p1 = tail[0], tail[-1]
    dt = p1["frame"] - p0["frame"]
    vel = (np.asarray(p1["center"], dtype=float) - np.asarray(p0["center"], dtype=float)) / max(dt, 1)
    return np.asarray(p1["center"], dtype=float), vel, p1["frame"]


def _segment_start_state(pts: List[dict]):
    p0 = pts[0]
    return np.asarray(p0["center"], dtype=float), p0["frame"]


def _majority_team(pts: List[dict]):
    teams = [p.get("team") for p in pts if p.get("team")]
    if not teams:
        return None
    vals, counts = np.unique(teams, return_counts=True)
    return vals[int(np.argmax(counts))]


def stitch_player_ids(
    tracking_data: List[dict],
    fps: float,
    pixel_to_meter: float,
    max_gap_seconds: float = 1.6,
    position_gate_meters: float = 3.5,
    require_same_team: bool = True,
    min_segment_len: int = 2,
) -> Dict[int, int]:
    """Returns a dict old_track_id -> canonical_track_id (identity if no
    stitching applies).

    Algorithm:
      1. Compute end-state (position + extrapolated velocity) and
         start-state for every track.
      2. Build a cost matrix between (ended track) x (track starting
         shortly after), keeping only pairs within `max_gap_seconds`.
      3. Cost = distance between the extrapolated position and the new
         track's start position; rejected (inf) if teams differ or the
         distance exceeds `position_gate_meters`.
      4. Solve with the Hungarian algorithm for a globally-optimal
         assignment (rather than greedy first-match).
      5. Union-find merges chains (A->B->C) into one canonical ID.
    """
    max_gap_frames = int(max_gap_seconds * fps)
    position_gate_px = position_gate_meters / pixel_to_meter

    segments = build_player_track_segments(tracking_data)
    segments = {tid: pts for tid, pts in segments.items() if len(pts) >= min_segment_len}

    tids = list(segments.keys())
    n = len(tids)
    if n < 2:
        return {tid: tid for tid in tids}

    end_states = {}
    start_states = {}
    team_of = {}
    for tid, pts in segments.items():
        end_states[tid] = _segment_end_state(pts)
        start_states[tid] = _segment_start_state(pts)
        team_of[tid] = _majority_team(pts)

    ordered = sorted(tids, key=lambda t: start_states[t][1])

    INF = 1e9
    cost = np.full((n, n), INF)
    for i, tid_a in enumerate(ordered):
        end_pos, vel, end_frame = end_states[tid_a]
        for j, tid_b in enumerate(ordered):
            if tid_a == tid_b:
                continue
            start_pos, start_frame = start_states[tid_b]
            gap = start_frame - end_frame
            if gap <= 0 or gap > max_gap_frames:
                continue
            if require_same_team and team_of[tid_a] is not None and team_of[tid_b] is not None:
                if team_of[tid_a] != team_of[tid_b]:
                    continue
            predicted_pos = end_pos + vel * gap
            dist = float(np.hypot(*(predicted_pos - start_pos)))
            if dist > position_gate_px:
                continue
            cost[i, j] = dist

    row_ind, col_ind = linear_sum_assignment(cost)

    parent = {tid: tid for tid in tids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    n_stitched = 0
    for i, j in zip(row_ind, col_ind):
        if cost[i, j] < INF:
            union(ordered[i], ordered[j])
            n_stitched += 1

    remap = {tid: find(tid) for tid in tids}
    return remap


def apply_id_remap(tracking_data: List[dict], remap: Dict[int, int]) -> List[dict]:
    """Applies the remap, returning a new list (does not mutate the input)."""
    out = []
    for e in tracking_data:
        if e.get("class") == "player" and e["track_id"] in remap:
            e = {**e, "track_id": remap[e["track_id"]]}
        out.append(e)
    return out


def normalize_team_per_track(tracking_data: List[dict]) -> List[dict]:
    """Every point of a given track_id takes that track's majority team,
    instead of being decided frame-by-frame — prevents momentary
    flickering during player contact/occlusion."""
    segments = build_player_track_segments(tracking_data)
    majority = {tid: _majority_team(pts) for tid, pts in segments.items()}
    out = []
    for e in tracking_data:
        if e.get("class") == "player" and e["track_id"] in majority and majority[e["track_id"]]:
            e = {**e, "team": majority[e["track_id"]]}
        out.append(e)
    return out


def drop_noise_tracks(tracking_data: List[dict], min_frames: int = 8) -> List[dict]:
    """Removes any player track_id that appears in fewer than `min_frames`
    frames — almost certainly a momentary false positive, not a real player."""
    segments = build_player_track_segments(tracking_data)
    valid_ids = {tid for tid, pts in segments.items() if len(pts) >= min_frames}
    return [e for e in tracking_data
            if e.get("class") != "player" or e["track_id"] in valid_ids]

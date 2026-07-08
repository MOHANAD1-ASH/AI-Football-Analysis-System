"""
Event Engine — walks the entire match once and produces a temporally
consistent event stream (possession_gain/loss, touch, pass/pass_failed/
pass_intercepted, ball_recovery/turnover, shot). Every statistic in
`match_statistics.py` is derived purely from this event list (plus
smoothed trajectories for the inherently-continuous distance/speed/
sprint metrics).

Ported from the notebook's Event Engine redesign — logic unchanged.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import StatsConfig
from services.pitch_detection import HomographyStore


# ────────────────────────────────────────────────────────────
#  COORDINATE TRANSFORM
# ────────────────────────────────────────────────────────────
class CoordinateTransformer:
    """Converts camera-pixel coordinates to real pitch metres. Uses a
    per-frame homography when available (via HomographyStore); otherwise
    falls back to a linear approximation based on visible pitch metres."""

    def __init__(self, config: StatsConfig, homography_store: Optional[HomographyStore] = None):
        self.config = config
        self.homography_store = homography_store

    def to_pitch(self, px: float, py: float, frame: Optional[int] = None) -> Tuple[float, float]:
        if self.homography_store is not None and frame is not None:
            H = self.homography_store.get(frame)
            if H is not None:
                vec = np.array([px, py, 1.0])
                out = H @ vec
                out = out / out[2]
                # CONFIG.vertices are in centimetres -> convert to metres to
                # stay consistent with the rest of the engine (metres everywhere).
                return float(out[0] / 100.0), float(out[1] / 100.0)
        return float(px * self.config.pixel_to_meter), float(py * self.config.pixel_to_meter_y)

    def distance_m(self, p1: Tuple[float, float], p2: Tuple[float, float],
                   frame1: Optional[int] = None, frame2: Optional[int] = None) -> float:
        f2 = frame2 if frame2 is not None else frame1
        a = self.to_pitch(*p1, frame=frame1)
        b = self.to_pitch(*p2, frame=f2)
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    def pixel_distance(self, p1, p2) -> float:
        return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))


# ────────────────────────────────────────────────────────────
#  EVENT
# ────────────────────────────────────────────────────────────
@dataclass
class Event:
    frame: int
    timestamp_s: float
    type: str
    player_id: Optional[int] = None
    team: Optional[str] = None
    position: Optional[Tuple[float, float]] = None
    data: dict = field(default_factory=dict)

    def __repr__(self):
        mm = int(self.timestamp_s // 60)
        ss = self.timestamp_s % 60
        pid = f"P{self.player_id}" if self.player_id is not None else "-"
        team = self.team or "-"
        return f"[{mm:02d}:{ss:05.2f}] {self.type:<22} {team:<2} {pid:<5} {self.data}"


# ────────────────────────────────────────────────────────────
#  HELPERS
# ────────────────────────────────────────────────────────────
def group_by_frame(tracking_data: List[dict]) -> Dict[int, List[dict]]:
    frames = defaultdict(list)
    for e in tracking_data:
        frames[int(e["frame"])].append(e)
    return dict(sorted(frames.items()))


def smooth_ball_trajectory(tracking_data: List[dict], config: StatsConfig) -> List[dict]:
    """Two-stage cleaning of raw ball detections: reject implausible
    jumps (compared against the last ACCEPTED point, not the raw
    previous point, to avoid cascading rejections), then median-smooth."""
    pts = sorted([e for e in tracking_data if e.get("class") == "ball"], key=lambda e: e["frame"])
    if len(pts) < 2:
        return pts

    px_per_frame_cap = (config.max_ball_speed_ms / config.fps) / config.pixel_to_meter

    filtered = [pts[0]]
    last_accepted = pts[0]
    for cur in pts[1:]:
        gap = int(cur["frame"]) - int(last_accepted["frame"])
        if gap <= 0:
            continue
        if gap > config.motion_max_frame_gap:
            filtered.append(cur)
            last_accepted = cur
            continue
        dist_px = np.hypot(cur["center"][0] - last_accepted["center"][0],
                            cur["center"][1] - last_accepted["center"][1])
        if dist_px / gap > px_per_frame_cap:
            continue
        filtered.append(cur)
        last_accepted = cur

    if len(filtered) <= config.ball_smoothing_window:
        return filtered

    half = config.ball_smoothing_window // 2
    centers = np.array([p["center"] for p in filtered], dtype=float)
    out = []
    for i, p in enumerate(filtered):
        s, e = max(0, i - half), min(len(filtered), i + half + 1)
        c = np.median(centers[s:e], axis=0)
        out.append({**p, "center": [float(c[0]), float(c[1])]})
    return out


def smooth_player_trajectory(points: List[dict], config: StatsConfig) -> List[dict]:
    """Moving-median smoothing for a single player's track."""
    if len(points) <= config.position_smoothing_window:
        return points
    half = config.position_smoothing_window // 2
    centers = np.array([p["center"] for p in points], dtype=float)
    out = []
    for i, p in enumerate(points):
        s, e = max(0, i - half), min(len(points), i + half + 1)
        c = np.median(centers[s:e], axis=0)
        out.append({**p, "center": [float(c[0]), float(c[1])]})
    return out


# ────────────────────────────────────────────────────────────
#  POSSESSION STATE MACHINE
# ────────────────────────────────────────────────────────────
class PossessionStateMachine:
    """Decides who controls the ball, frame by frame, without oscillating.

    - A player must be closest AND within `gain_radius` for `min_frames`
      consecutive frames before possession is awarded.
    - Possession is lost only when the owner drifts beyond the larger
      `lose_radius` (hysteresis), the ball is missing too long, or the
      owner's own detection is missing longer than
      `possession_owner_occlusion_grace_frames` (tolerates brief
      tracker dropout without ending possession).
    - A different player can only take possession via the same
      candidate -> min_frames gate, preventing single-frame proximity
      swaps from stealing possession.
    """

    def __init__(self, config: StatsConfig):
        self.cfg = config
        self.owner_id: Optional[int] = None
        self.owner_team: Optional[str] = None
        self.owner_since_frame: Optional[int] = None
        self._candidate_id: Optional[int] = None
        self._candidate_count: int = 0
        self._last_ball_frame: Optional[int] = None
        self._owner_missing_streak: int = 0

    def _nearest_player(self, ball_center, players) -> Tuple[Optional[dict], float]:
        if not players:
            return None, float("inf")
        centers = np.array([p["center"] for p in players], dtype=float)
        dists = np.linalg.norm(centers - np.asarray(ball_center, dtype=float), axis=1)
        idx = int(np.argmin(dists))
        return players[idx], float(dists[idx])

    def update(self, frame: int, ball_center, players: List[dict]) -> List[Event]:
        events: List[Event] = []
        ts = frame / self.cfg.fps

        if ball_center is None:
            if self._last_ball_frame is not None and self.owner_id is not None:
                if frame - self._last_ball_frame > self.cfg.possession_max_ball_gap_frames:
                    events.append(self._end_possession(frame, ts, reason="ball_lost"))
            return events
        self._last_ball_frame = frame

        nearest, dist = self._nearest_player(ball_center, players)

        if self.owner_id is not None:
            owner_entry = next((p for p in players if p["track_id"] == self.owner_id), None)

            if owner_entry is None:
                self._owner_missing_streak += 1
                if self._owner_missing_streak <= self.cfg.possession_owner_occlusion_grace_frames:
                    self._reset_candidate()
                    return events
                else:
                    events.append(self._end_possession(frame, ts, reason="owner_occluded_too_long"))
            else:
                self._owner_missing_streak = 0
                owner_dist = np.hypot(owner_entry["center"][0] - ball_center[0],
                                       owner_entry["center"][1] - ball_center[1])
                if owner_dist <= self.cfg.possession_lose_radius_px:
                    if nearest is not None and nearest["track_id"] != self.owner_id and dist <= self.cfg.possession_gain_radius_px:
                        self._advance_candidate(nearest)
                        if self._candidate_count >= self.cfg.possession_min_frames:
                            events.append(self._end_possession(frame, ts, reason="challenged"))
                            events.append(self._start_possession(nearest, frame, ts))
                    else:
                        self._reset_candidate()
                    return events
                else:
                    events.append(self._end_possession(frame, ts, reason="drifted_away"))

        if nearest is not None and dist <= self.cfg.possession_gain_radius_px:
            self._advance_candidate(nearest)
            if self._candidate_count >= self.cfg.possession_min_frames:
                events.append(self._start_possession(nearest, frame, ts))
        else:
            self._reset_candidate()

        return events

    def _advance_candidate(self, player):
        if self._candidate_id == player["track_id"]:
            self._candidate_count += 1
        else:
            self._candidate_id = player["track_id"]
            self._candidate_count = 1

    def _reset_candidate(self):
        self._candidate_id = None
        self._candidate_count = 0

    def _start_possession(self, player, frame, ts) -> Event:
        self.owner_id = player["track_id"]
        self.owner_team = player.get("team")
        self.owner_since_frame = frame
        self._owner_missing_streak = 0
        self._reset_candidate()
        return Event(frame, ts, "possession_gain", player_id=self.owner_id,
                     team=self.owner_team, position=tuple(player["center"]))

    def _end_possession(self, frame, ts, reason: str) -> Event:
        ev = Event(frame, ts, "possession_loss", player_id=self.owner_id,
                   team=self.owner_team, data={"reason": reason})
        self.owner_id = None
        self.owner_team = None
        self.owner_since_frame = None
        self._owner_missing_streak = 0
        return ev


# ────────────────────────────────────────────────────────────
#  MAIN EVENT ENGINE
# ────────────────────────────────────────────────────────────
class EventEngine:
    """Walks the entire match once and produces a temporally consistent
    event stream. All statistics modules consume this output instead of
    re-deriving logic from raw tracking_data."""

    def __init__(self, config: StatsConfig, transformer: Optional[CoordinateTransformer] = None):
        self.cfg = config
        self.transformer = transformer or CoordinateTransformer(config)
        self.possession_fsm = PossessionStateMachine(config)

    def run(self, tracking_data: List[dict]) -> List[Event]:
        cfg = self.cfg
        events: List[Event] = []

        frames = group_by_frame(tracking_data)
        ball_pts = {p["frame"]: p["center"] for p in smooth_ball_trajectory(tracking_data, cfg)}

        pending_release: Optional[dict] = None
        last_owner_team_stable: Optional[str] = None

        all_frame_indices = sorted(frames.keys())
        for frame in all_frame_indices:
            entries = frames[frame]
            players = [e for e in entries if e.get("class") == "player" and e.get("team") in ("A", "B")]
            ball_center = ball_pts.get(frame)
            ts_now = frame / cfg.fps

            if (pending_release is not None
                    and not pending_release.get("is_shot")
                    and (ts_now - pending_release["timestamp_s"]) > cfg.pass_max_duration_s):
                events.append(Event(pending_release["frame"], pending_release["timestamp_s"],
                                     "pass_failed", player_id=pending_release["player_id"],
                                     team=pending_release["team"], position=pending_release["position"],
                                     data={"reason": "timeout_or_out_of_play"}))
                pending_release = None
            elif pending_release is not None and pending_release.get("is_shot") and \
                    (ts_now - pending_release["timestamp_s"]) > cfg.pass_max_duration_s:
                pending_release = None

            poss_events = self.possession_fsm.update(frame, ball_center, players)

            for ev in poss_events:
                events.append(ev)

                if ev.type == "possession_gain":
                    events.append(Event(ev.frame, ev.timestamp_s, "touch",
                                         player_id=ev.player_id, team=ev.team, position=ev.position))

                    if pending_release is not None:
                        resolved = self._resolve_release(pending_release, ev, cfg)
                        events.extend(resolved)
                        pending_release = None

                    if last_owner_team_stable is not None and ev.team != last_owner_team_stable:
                        events.append(Event(ev.frame, ev.timestamp_s, "ball_recovery",
                                             player_id=ev.player_id, team=ev.team,
                                             data={"recovered_from": last_owner_team_stable}))
                        events.append(Event(ev.frame, ev.timestamp_s, "turnover",
                                             team=last_owner_team_stable,
                                             data={"lost_to": ev.team, "player_id": ev.player_id}))
                    last_owner_team_stable = ev.team

                elif ev.type == "possession_loss":
                    release_pos = ev.position if ev.position else self._last_known_pos(ev.player_id, entries)
                    pending_release = {
                        "frame": ev.frame, "timestamp_s": ev.timestamp_s,
                        "player_id": ev.player_id, "team": ev.team,
                        "position": release_pos, "is_shot": False,
                        "ball_position": ball_pts.get(ev.frame, ball_center),
                    }
                    shot_ev = self._classify_shot(pending_release, ball_pts, cfg)
                    if shot_ev is not None:
                        events.append(shot_ev)
                        pending_release["is_shot"] = True

        if pending_release is not None and not pending_release.get("is_shot"):
            events.append(Event(pending_release["frame"], pending_release["timestamp_s"],
                                 "pass_failed", player_id=pending_release["player_id"],
                                 team=pending_release["team"],
                                 data={"reason": "unresolved_end_of_data"}))

        events.sort(key=lambda e: e.frame)
        return events

    # ── Shot classifier ───────────────────────────────────────────────────
    def _classify_shot(self, release: dict, ball_pts: Dict[int, list], cfg: StatsConfig) -> Optional[Event]:
        """A release is a shot if the ball, shortly after leaving the
        player, travels at high speed roughly toward the opponent's goal.
        Runs independently of pass resolution (a shot may be saved/
        blocked/out of play without ever being "received")."""
        motion_start_pos = release.get("ball_position") or release["position"]
        if motion_start_pos is None or release["team"] not in cfg.attack_direction:
            return None

        window_frames = int(cfg.fps * 1.0)
        start_frame = release["frame"]
        candidates = sorted(f for f in ball_pts.keys() if start_frame < f <= start_frame + window_frames)
        if not candidates:
            return None

        prev_pos, prev_f = motion_start_pos, start_frame
        best_speed = 0.0
        end_pos = None
        for f in candidates:
            cur_pos = ball_pts[f]
            gap = f - prev_f
            if gap <= 0 or gap > cfg.motion_max_frame_gap:
                prev_pos, prev_f = cur_pos, f
                continue
            speed = self.transformer.distance_m(prev_pos, cur_pos) / (gap / cfg.fps)
            speed = min(speed, cfg.max_ball_speed_ms)
            if speed > best_speed:
                best_speed = speed
                end_pos = cur_pos
            prev_pos, prev_f = cur_pos, f

        if end_pos is None or best_speed < cfg.shot_min_speed_ms:
            return None

        direction_sign = cfg.attack_direction[release["team"]]
        dx = (end_pos[0] - motion_start_pos[0]) * direction_sign
        dy = end_pos[1] - motion_start_pos[1]
        if dx <= 0:
            return None
        angle_deg = np.degrees(np.arctan2(abs(dy), dx))
        if angle_deg > cfg.shot_goal_cone_deg:
            return None

        return Event(release["frame"], release["timestamp_s"], "shot",
                     player_id=release["player_id"], team=release["team"],
                     position=release["position"],
                     data={"speed_ms": round(best_speed, 2), "angle_deg": round(angle_deg, 1)})

    # ── Resolve a ball release into pass / shot / interception ──────────────
    def _resolve_release(self, release: dict, gain_event: Event, cfg: StatsConfig) -> List[Event]:
        out: List[Event] = []

        if release.get("is_shot"):
            return out

        dt = gain_event.timestamp_s - release["timestamp_s"]
        if release["position"] is None or gain_event.position is None:
            return out

        if release["player_id"] == gain_event.player_id:
            return out

        travel_px = self.transformer.pixel_distance(release["position"], gain_event.position)

        if dt > cfg.pass_max_duration_s or travel_px < cfg.pass_min_travel_px:
            return out

        same_team = release["team"] == gain_event.team
        ev_type = "pass" if same_team else "pass_intercepted"
        out.append(Event(
            gain_event.frame, gain_event.timestamp_s, ev_type,
            player_id=gain_event.player_id, team=release["team"],
            position=gain_event.position,
            data={
                "from_player": release["player_id"],
                "to_player": gain_event.player_id,
                "from_team": release["team"],
                "to_team": gain_event.team,
                "from_position": release["position"],
                "to_position": gain_event.position,
                "from_frame": release["frame"],
                "to_frame": gain_event.frame,
                "duration_s": round(dt, 2),
                "travel_m": round(self.transformer.distance_m(
                    release["position"], gain_event.position,
                    frame1=release["frame"], frame2=gain_event.frame), 2),
            },
        ))
        return out

    @staticmethod
    def _last_known_pos(player_id, entries):
        for e in entries:
            if e.get("track_id") == player_id:
                return tuple(e["center"])
        return None

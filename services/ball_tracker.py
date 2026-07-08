"""
Kalman-based ball tracker (v9 — capped visual coasting).

Ported unchanged in logic from the original notebook. Combines a
constant-velocity Kalman filter with multi-candidate gating, velocity/
direction penalties, a static false-positive blacklist (fixes markings
such as the penalty spot being misdetected as the ball), and a
confirm/re-acquire state machine so the tracker doesn't jump onto noise
during occlusions.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np


class _KalmanCV:
    """Minimal constant-velocity Kalman filter, state = [x, y, vx, vy]."""

    def __init__(self, dt: float = 1.0, process_noise: float = 8.0,
                 measurement_noise: float = 12.0):
        self.dt = dt
        self.F = np.array([[1, 0, dt, 0],
                            [0, 1, 0, dt],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]], dtype=float)
        self.H = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0]], dtype=float)
        self.Q = process_noise * np.eye(4)
        self.base_R = measurement_noise * np.eye(2)
        self.R = self.base_R.copy()
        self.P = np.eye(4) * 500.0
        self.x = np.zeros((4, 1))
        self.initialized = False

    def init_state(self, x: float, y: float, vx: float = 0.0, vy: float = 0.0) -> None:
        self.x = np.array([[x], [y], [vx], [vy]], dtype=float)
        self.P = np.eye(4) * 500.0
        self.initialized = True

    def predict(self) -> Tuple[float, float]:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return float(self.x[0, 0]), float(self.x[1, 0])

    def update(self, x: float, y: float) -> None:
        z = np.array([[x], [y]], dtype=float)
        residual = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ residual
        self.P = (np.eye(4) - K @ self.H) @ self.P

    def inflate_R(self, factor: float) -> None:
        self.R = self.base_R * factor

    def decay_R(self, step_factor: float = 0.7, floor: float = 1.0) -> None:
        self.R = self.base_R + (self.R - self.base_R) * step_factor
        if np.max(np.abs(self.R - self.base_R)) < floor:
            self.R = self.base_R.copy()

    @property
    def position(self) -> Tuple[float, float]:
        return float(self.x[0, 0]), float(self.x[1, 0])

    @property
    def velocity(self) -> Tuple[float, float]:
        return float(self.x[2, 0]), float(self.x[3, 0])


def find_static_false_positive_zones(
    all_candidates: List[Tuple[int, float, float, float]],
    frame_count: int,
    cluster_radius_px: float = 15.0,
    min_presence_ratio: float = 0.35,
) -> List[Tuple[float, float]]:
    """Detects static pitch markings (e.g. penalty spot) that repeatedly
    fire as false ball detections, so they can be blacklisted."""
    if not all_candidates or frame_count == 0:
        return []

    pts = np.array([(c[1], c[2]) for c in all_candidates])
    used = np.zeros(len(pts), dtype=bool)
    blacklist: List[Tuple[float, float]] = []

    for i in range(len(pts)):
        if used[i]:
            continue
        dist = np.hypot(pts[:, 0] - pts[i, 0], pts[:, 1] - pts[i, 1])
        cluster_mask = dist <= cluster_radius_px
        frames_hit = len(set(all_candidates[j][0] for j in np.where(cluster_mask)[0]))
        if frames_hit / frame_count >= min_presence_ratio:
            cx = float(pts[cluster_mask, 0].mean())
            cy = float(pts[cluster_mask, 1].mean())
            blacklist.append((cx, cy))
        used |= cluster_mask

    return blacklist


def interpolate_ball(positions: Dict[int, Tuple[float, float]], max_gap: int = 10) -> Dict[int, Tuple[float, float]]:
    """Bridges real gaps in ball tracking with straight-line interpolation
    between two real detections (never a stale constant-velocity guess)."""
    if not positions:
        return {}

    frames = sorted(positions.keys())
    filled = dict(positions)

    for i in range(len(frames) - 1):
        f0, f1 = frames[i], frames[i + 1]
        gap = f1 - f0
        if gap <= 1 or gap > max_gap:
            continue

        x0, y0 = positions[f0]
        x1, y1 = positions[f1]
        for step_i in range(1, gap):
            t = step_i / gap
            filled[f0 + step_i] = (x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)

    return filled


class BallKalmanTracker:
    """v9 — capped visual coasting + local-static rejection + dynamic
    reinit radius + debug logging. See module docstring."""

    def __init__(self, fps: float = 25.0, base_gate_px: float = 90.0,
                 max_lost_frames: int = 40, max_missed_frames: Optional[int] = None,
                 process_noise: float = 8.0, measurement_noise: float = 12.0,
                 blacklist_zones: Optional[List[Tuple[float, float]]] = None,
                 blacklist_radius_px: float = 18.0,
                 size_gate_ratio: float = 2.2, size_ema_alpha: float = 0.25,
                 direction_weight: float = 25.0, direction_min_speed: float = 3.0,
                 confirm_frames: int = 2, confirm_hits: Optional[int] = None,
                 confirm_radius_px: float = 25.0,
                 confirm_min_lost: int = 3,
                 speed_gate_scale: float = 1.6,
                 reacquire_inflate: float = 6.0, reacquire_decay: float = 0.6,
                 frame_w: Optional[int] = None, pitch_visible_m: Optional[float] = None,
                 max_speed_kmh: Optional[float] = None,
                 reinit_min_conf: Optional[float] = None,
                 confirm_min_conf: Optional[float] = None,
                 confirm_speed_budget_px: Optional[float] = None,
                 reinit_confirm_frames: int = 2,
                 reinit_confirm_radius_px: float = 40.0,
                 debug_enabled: bool = True,
                 local_static_enabled: bool = True,
                 local_static_window_frames: int = 15,
                 local_static_radius_px: float = 15.0,
                 local_static_min_hits: int = 8,
                 coast_visible_frames: int = 5):
        self.kf = _KalmanCV(dt=1.0, process_noise=process_noise, measurement_noise=measurement_noise)
        self.fps = fps
        self.base_gate = base_gate_px

        self.max_lost = max_missed_frames if max_missed_frames is not None else max_lost_frames
        self.confirm_frames = confirm_hits if confirm_hits is not None else confirm_frames
        self.confirm_min_lost = confirm_min_lost

        self.lost_count = 0
        self.blacklist_zones = blacklist_zones or []
        self.blacklist_radius = blacklist_radius_px
        self.history: Dict[int, Tuple[float, float]] = {}
        self.is_interp: Dict[int, bool] = {}

        self.expected_size: Optional[float] = None
        self.size_gate_ratio = size_gate_ratio
        self.size_ema_alpha = size_ema_alpha
        self.direction_weight = direction_weight
        self.direction_min_speed = direction_min_speed
        self.confirm_radius = confirm_radius_px
        self.speed_gate_scale = speed_gate_scale
        self.reacquire_inflate = reacquire_inflate
        self.reacquire_decay = reacquire_decay

        self._pending = None
        self._reacquire_cooldown = 0

        self.frame_w = frame_w
        self.pitch_visible_m = pitch_visible_m
        self.max_speed_kmh = max_speed_kmh
        self.max_speed_px_per_frame = None
        if frame_w and pitch_visible_m and max_speed_kmh and fps:
            px_per_m = frame_w / pitch_visible_m
            max_speed_m_per_s = max_speed_kmh * 1000.0 / 3600.0
            self.max_speed_px_per_frame = (max_speed_m_per_s * px_per_m) / fps

        self.reinit_min_conf = reinit_min_conf
        self.confirm_min_conf = confirm_min_conf
        self.confirm_speed_budget_px = confirm_speed_budget_px

        self.reinit_confirm_frames = reinit_confirm_frames
        self.reinit_confirm_radius = reinit_confirm_radius_px
        self._reinit_pending = None

        self.debug_enabled = debug_enabled
        self.debug_log: List[dict] = []

        self.local_static_enabled = local_static_enabled
        self.local_static_window = local_static_window_frames
        self.local_static_radius = local_static_radius_px
        self.local_static_min_hits = local_static_min_hits
        self._raw_history = deque()

        self.coast_visible_frames = coast_visible_frames

    # ── internal helpers ────────────────────────────────────────────────
    def _is_blacklisted(self, cx, cy):
        for bx, by in self.blacklist_zones:
            if np.hypot(cx - bx, cy - by) <= self.blacklist_radius:
                return True
        return False

    def _current_speed(self):
        vx, vy = self.kf.velocity
        return float(np.hypot(vx, vy))

    def _gate_radius(self):
        lost_term = self.base_gate * (1.0 + 0.18 * min(self.lost_count, 15))
        speed_term = self.speed_gate_scale * self._current_speed()
        gate = lost_term + speed_term

        if self.max_speed_px_per_frame is not None:
            physical_cap = self.max_speed_px_per_frame * max(self.lost_count, 1) * 1.5
            gate = min(gate, max(physical_cap, self.base_gate))
        return gate

    def _effective_confirm_distance(self):
        if self.confirm_speed_budget_px is not None:
            return max(self.confirm_speed_budget_px * max(self.lost_count, 1), self.confirm_radius)
        if self.max_speed_px_per_frame is not None:
            return max(self.confirm_radius, self.max_speed_px_per_frame * 1.5)
        return max(self.confirm_radius, self.base_gate * 0.5)

    def _effective_reinit_confirm_distance(self):
        if self.confirm_speed_budget_px is not None:
            return max(self.confirm_speed_budget_px, self.reinit_confirm_radius)
        if self.max_speed_px_per_frame is not None:
            return max(self.reinit_confirm_radius, self.max_speed_px_per_frame * 1.5)
        return self.reinit_confirm_radius

    def _size_ok(self, size):
        if size is None or self.expected_size is None:
            return True, 0.0
        ratio = max(size, 1e-6) / max(self.expected_size, 1e-6)
        ratio = max(ratio, 1.0 / ratio)
        if ratio > self.size_gate_ratio:
            return False, ratio
        return True, ratio

    def _direction_penalty(self, cx, cy, pred_x, pred_y):
        speed = self._current_speed()
        if speed < self.direction_min_speed:
            return 0.0
        vx, vy = self.kf.velocity
        move_vec = np.array([cx - pred_x, cy - pred_y])
        vel_vec = np.array([vx, vy])
        if np.linalg.norm(move_vec) < 1e-6:
            return 0.0
        cos_sim = np.dot(move_vec, vel_vec) / (np.linalg.norm(move_vec) * np.linalg.norm(vel_vec) + 1e-6)
        return self.direction_weight * (1.0 - cos_sim)

    def _update_expected_size(self, size):
        if size is None:
            return
        if self.expected_size is None:
            self.expected_size = size
        else:
            a = self.size_ema_alpha
            self.expected_size = (1 - a) * self.expected_size + a * size

    def _best_by_conf(self, candidates, min_conf=None):
        pool = candidates if min_conf is None else [c for c in candidates if c[2] >= min_conf]
        if not pool:
            return None
        return max(pool, key=lambda c: c[2])

    def _record_raw_candidates(self, frame_idx, clean):
        for cx, cy, conf, size in clean:
            self._raw_history.append((frame_idx, cx, cy))
        cutoff = frame_idx - self.local_static_window
        while self._raw_history and self._raw_history[0][0] < cutoff:
            self._raw_history.popleft()

    def _is_locally_static(self, cx, cy):
        if not self.local_static_enabled:
            return False
        hit_frames = set()
        for fi, hx, hy in self._raw_history:
            if np.hypot(hx - cx, hy - cy) <= self.local_static_radius:
                hit_frames.add(fi)
        return len(hit_frames) >= self.local_static_min_hits

    def _maybe_record_coast(self, frame_idx, pred_x, pred_y):
        """Only record the Kalman coast prediction for the first
        `coast_visible_frames` of a loss; beyond that leave a real gap for
        `interpolate_ball` to bridge with a straight line."""
        if self.lost_count <= self.coast_visible_frames:
            self.history[frame_idx] = (pred_x, pred_y)
            self.is_interp[frame_idx] = True

    # ── main entry point ────────────────────────────────────────────────
    def step(self, frame_idx: int, candidates: List[tuple]) -> None:
        """candidates: list of (cx, cy, conf) or (cx, cy, conf, size)."""
        norm = []
        for c in candidates:
            if len(c) == 4:
                norm.append(c)
            else:
                norm.append((c[0], c[1], c[2], None))

        clean = [c for c in norm if not self._is_blacklisted(c[0], c[1])]
        lost_before = self.lost_count

        self._record_raw_candidates(frame_idx, clean)

        if not self.kf.initialized:
            best = self._best_by_conf(clean, min_conf=self.reinit_min_conf)
            if best is not None:
                self.kf.init_state(best[0], best[1])
                self.lost_count = 0
                self.history[frame_idx] = (best[0], best[1])
                self.is_interp[frame_idx] = False
                self._update_expected_size(best[3])
                self._log(frame_idx, 'init', lost_before, (best[0], best[1]),
                           best[2], best[3], len(norm), len(clean))
            else:
                self._log(frame_idx, 'init_wait', lost_before,
                           n_candidates=len(norm), n_after_blacklist=len(clean),
                           reason='no_candidate_above_reinit_min_conf')
            return

        pred_x, pred_y = self.kf.predict()

        # ── DEAD TRACK — safe reinit, local-static rejection ────────────
        if self.lost_count > self.max_lost:
            pool = []
            n_rejected_static = 0
            for c in clean:
                cx, cy, conf, size = c
                if self.reinit_min_conf is not None and conf < self.reinit_min_conf:
                    continue
                size_ok, ratio = self._size_ok(size)
                if not size_ok:
                    continue
                if self._is_locally_static(cx, cy):
                    n_rejected_static += 1
                    continue
                pool.append(c)

            candidate = self._best_by_conf(pool)

            if candidate is not None:
                cx, cy, conf, size = candidate
                eff_reinit_radius = self._effective_reinit_confirm_distance()
                if self._reinit_pending is not None and \
                   np.hypot(cx - self._reinit_pending['pos'][0],
                             cy - self._reinit_pending['pos'][1]) <= eff_reinit_radius:
                    self._reinit_pending['count'] += 1
                    self._reinit_pending['pos'] = (cx, cy)
                    self._reinit_pending['size'] = size
                else:
                    self._reinit_pending = {'pos': (cx, cy), 'size': size, 'count': 1}

                if self._reinit_pending['count'] >= self.reinit_confirm_frames:
                    self.kf.init_state(cx, cy)
                    self.lost_count = 0
                    self._pending = None
                    self._reinit_pending = None
                    self._reacquire_cooldown = 3
                    self.kf.inflate_R(self.reacquire_inflate)
                    self.history[frame_idx] = (cx, cy)
                    self.is_interp[frame_idx] = False
                    self._update_expected_size(size)
                    self._log(frame_idx, 'dead_reinit', lost_before, (cx, cy),
                               conf, size, len(norm), len(clean),
                               reason=f'confirmed_after_{self.reinit_confirm_frames}_frames,rejected_static={n_rejected_static}')
                    return
                self._log(frame_idx, 'dead_wait', lost_before, (cx, cy), conf, size,
                           len(norm), len(clean),
                           reason=f'reinit_pending_count={self._reinit_pending["count"]},rejected_static={n_rejected_static}')
                return
            else:
                self._reinit_pending = None
                self._log(frame_idx, 'dead_wait', lost_before, None, None, None,
                           len(norm), len(clean),
                           reason=f'no_sane_candidate,rejected_static={n_rejected_static}' if clean else 'no_candidates')
                return

        if self._reacquire_cooldown > 0:
            self.kf.decay_R(step_factor=self.reacquire_decay)
            self._reacquire_cooldown -= 1
        else:
            self.kf.R = self.kf.base_R.copy()

        chosen = None
        chosen_size = None
        chosen_conf = None
        rejected_size = 0
        rejected_gate = 0
        if clean:
            gate = self._gate_radius()
            best_score = None
            for cx, cy, conf, size in clean:
                dist = np.hypot(cx - pred_x, cy - pred_y)
                if dist > gate:
                    rejected_gate += 1
                    continue

                size_ok, _ = self._size_ok(size)
                if not size_ok:
                    rejected_size += 1
                    continue

                dir_penalty = self._direction_penalty(cx, cy, pred_x, pred_y)
                score = dist - 35.0 * conf + dir_penalty
                if best_score is None or score < best_score:
                    best_score, chosen, chosen_size, chosen_conf = score, (cx, cy), size, conf

        was_lost = self.lost_count >= self.confirm_min_lost

        if was_lost and self.confirm_frames > 1:
            confirm_pool = clean if self.confirm_min_conf is None else \
                [c for c in clean if c[2] >= self.confirm_min_conf]
            confirm_pool = [c for c in confirm_pool if not self._is_locally_static(c[0], c[1])]

            confirm_candidate = None
            cc_conf = cc_size = None
            if confirm_pool:
                gate = self._gate_radius()
                best_score = None
                for cx, cy, conf, size in confirm_pool:
                    dist = np.hypot(cx - pred_x, cy - pred_y)
                    if dist > gate:
                        continue
                    size_ok, _ = self._size_ok(size)
                    if not size_ok:
                        continue
                    dir_penalty = self._direction_penalty(cx, cy, pred_x, pred_y)
                    score = dist - 35.0 * conf + dir_penalty
                    if best_score is None or score < best_score:
                        best_score, confirm_candidate, cc_conf, cc_size = score, (cx, cy), conf, size

            if confirm_candidate is not None:
                eff_dist = self._effective_confirm_distance()
                if self._pending is not None and \
                   np.hypot(confirm_candidate[0] - self._pending['pos'][0],
                             confirm_candidate[1] - self._pending['pos'][1]) <= eff_dist:
                    self._pending['count'] += 1
                    self._pending['pos'] = confirm_candidate
                else:
                    self._pending = {'pos': confirm_candidate, 'count': 1}

                if self._pending['count'] < self.confirm_frames:
                    self.lost_count += 1
                    self._maybe_record_coast(frame_idx, pred_x, pred_y)
                    self._log(frame_idx, 'confirm_wait', lost_before, confirm_candidate,
                               cc_conf, cc_size, len(norm), len(clean),
                               reason=f'pending_count={self._pending["count"]}')
                    return
                else:
                    self._pending = None
                    self._reacquire_cooldown = 3
                    self.kf.inflate_R(self.reacquire_inflate)
                    chosen, chosen_size, chosen_conf = confirm_candidate, cc_size, cc_conf
                    self._log(frame_idx, 'confirm_commit', lost_before, chosen,
                               chosen_conf, chosen_size, len(norm), len(clean))
            else:
                self.lost_count += 1
                self._maybe_record_coast(frame_idx, pred_x, pred_y)
                self._log(frame_idx, 'confirm_wait', lost_before, None, None, None,
                           len(norm), len(clean), reason='no_confirm_candidate_or_all_static')
                return
        elif chosen is not None:
            self._pending = None

        if chosen is not None:
            self.kf.update(chosen[0], chosen[1])
            self.lost_count = 0
            self.history[frame_idx] = self.kf.position
            self.is_interp[frame_idx] = False
            self._update_expected_size(chosen_size)
            self._log(frame_idx, 'normal', lost_before, chosen, chosen_conf, chosen_size,
                       len(norm), len(clean),
                       reason=f'rejected_gate={rejected_gate},rejected_size={rejected_size}')
        else:
            self._pending = None
            self.lost_count += 1
            self._maybe_record_coast(frame_idx, pred_x, pred_y)
            self._log(frame_idx, 'coast', lost_before, None, None, None,
                       len(norm), len(clean),
                       reason=f'rejected_gate={rejected_gate},rejected_size={rejected_size}')

    def update(self, frame_idx: int, candidates: List[tuple]) -> None:
        """Alias for step(), for pipeline-call compatibility."""
        return self.step(frame_idx, candidates)

    def _log(self, frame_idx, path, lost_count, chosen=None, chosen_conf=None,
              chosen_size=None, n_candidates=0, n_after_blacklist=0, reason=""):
        if not self.debug_enabled:
            return
        self.debug_log.append({
            'frame_idx': frame_idx,
            'path': path,
            'lost_count': lost_count,
            'chosen_x': None if chosen is None else chosen[0],
            'chosen_y': None if chosen is None else chosen[1],
            'chosen_conf': chosen_conf,
            'chosen_size': chosen_size,
            'expected_size': self.expected_size,
            'n_candidates': n_candidates,
            'n_after_blacklist': n_after_blacklist,
            'reason': reason,
        })

    def get_debug_df(self):
        import pandas as pd
        return pd.DataFrame(self.debug_log)

    @property
    def n_tracked(self) -> int:
        return sum(1 for v in self.is_interp.values() if not v)

    @property
    def positions(self) -> Dict[int, Tuple[float, float]]:
        return self.history

    def smoothed_positions(self, window: int = 5) -> Dict[int, Tuple[float, float]]:
        if not self.history:
            return {}

        frames = sorted(self.history.keys())
        xs = np.array([self.history[f][0] for f in frames])
        ys = np.array([self.history[f][1] for f in frames])

        half = window // 2
        smoothed = {}
        for i, f in enumerate(frames):
            lo, hi = max(0, i - half), min(len(frames), i + half + 1)
            smoothed[f] = (float(xs[lo:hi].mean()), float(ys[lo:hi].mean()))
        return smoothed

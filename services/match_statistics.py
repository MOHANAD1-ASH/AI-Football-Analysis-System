"""
Match Statistics modules.

Every function here is independent, reuses the Event Engine's output
(never re-derives possession/pass logic itself), and has a single
responsibility — mirroring how a professional analytics platform
separates "event detection" from "metric aggregation".

Two families:
  (a) EVENT-BASED stats  — derived purely from `events` (passes,
      possession %, recovery, turnover, touches, shots, attacks,
      progressive passes, through balls, crosses).
  (b) TRAJECTORY-BASED stats — distance, speed, sprints, ball speed,
      heatmaps. Inherently continuous, computed from smoothed
      trajectories rather than the event stream.

If a statistic cannot be reliably computed from available tracking
data, it is reported as "Unavailable" rather than estimated.
"""
from __future__ import annotations

import dataclasses
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import StatsConfig
from services.event_engine import (
    CoordinateTransformer,
    Event,
    EventEngine,
    group_by_frame,
    smooth_ball_trajectory,
    smooth_player_trajectory,
)
from services.pitch_detection import CONFIG, HomographyStore

UNAVAILABLE = "Unavailable"


# ────────────────────────────────────────────────────────────
#  (a) EVENT-BASED STATISTICS
# ────────────────────────────────────────────────────────────
def compute_possession_percentage(events: List[Event], total_frames: int, cfg: StatsConfig) -> Dict[str, float]:
    """Possession % of TOTAL match time (frames held / total_frames),
    matching how broadcast analytics report the stat. A+B may fall short
    of 100% by however much time the ball was loose/uncontrolled."""
    time_held = {"A": 0, "B": 0}
    owner_team, owner_since = None, None
    last_frame = 0
    for ev in events:
        if ev.frame > last_frame:
            last_frame = ev.frame
        if ev.type == "possession_gain":
            owner_team, owner_since = ev.team, ev.frame
        elif ev.type == "possession_loss" and owner_team is not None:
            time_held[owner_team] = time_held.get(owner_team, 0) + (ev.frame - owner_since)
            owner_team, owner_since = None, None

    if owner_team is not None and owner_since is not None:
        end_frame = max(last_frame, total_frames - 1)
        time_held[owner_team] = time_held.get(owner_team, 0) + (end_frame - owner_since)

    if total_frames <= 0:
        return {"A": 0.0, "B": 0.0}
    return {
        "A": round(100 * time_held.get("A", 0) / total_frames, 1),
        "B": round(100 * time_held.get("B", 0) / total_frames, 1),
    }


def compute_touches(events: List[Event]) -> Dict[str, dict]:
    per_player = Counter()
    per_team = Counter()
    for ev in events:
        if ev.type == "touch":
            per_player[ev.player_id] += 1
            if ev.team:
                per_team[ev.team] += 1
    return {"per_player": dict(per_player), "per_team": dict(per_team)}


def compute_passes(events: List[Event]) -> dict:
    """Pass attempts / completions / accuracy, plus the raw successful
    pass events (consumed downstream for progressive/through/cross
    classification and the passing network)."""
    attempts = Counter()
    completions = Counter()
    successful_passes = []
    for ev in events:
        if ev.type == "pass":
            attempts[ev.team] += 1
            completions[ev.team] += 1
            successful_passes.append(ev)
        elif ev.type == "pass_intercepted":
            attempts[ev.team] += 1
        elif ev.type == "pass_failed":
            if ev.team:
                attempts[ev.team] += 1

    accuracy = {}
    for team in ("A", "B"):
        att = attempts.get(team, 0)
        accuracy[team] = round(100 * completions.get(team, 0) / att, 1) if att else 0.0

    return {
        "attempts": dict(attempts),
        "completions": dict(completions),
        "accuracy_pct": accuracy,
        "successful_pass_events": successful_passes,
    }


def compute_recovery_and_turnover(events: List[Event]) -> dict:
    recoveries = Counter()
    turnovers = Counter()
    for ev in events:
        if ev.type == "ball_recovery":
            recoveries[ev.team] += 1
        elif ev.type == "turnover":
            turnovers[ev.team] += 1
    return {"recoveries": dict(recoveries), "turnovers": dict(turnovers)}


def compute_shots(events: List[Event]) -> dict:
    """Shots detected via the velocity/direction heuristic in the Event
    Engine. "Shots on target" cannot be reliably inferred without goal-
    frame / goalkeeper-save detection, so it is reported as Unavailable."""
    per_team = Counter()
    shot_events = []
    for ev in events:
        if ev.type == "shot":
            per_team[ev.team] += 1
            shot_events.append(ev)
    return {
        "per_team": dict(per_team),
        "on_target_per_team": {"A": UNAVAILABLE, "B": UNAVAILABLE},
        "events": shot_events,
    }


def compute_attacks(events: List[Event], cfg: StatsConfig) -> Dict[str, int]:
    """An "attack" = a sustained possession spell (>= attack_min_possession_frames)
    during which the ball progresses forward AND enters that team's
    attacking third — debounced with a cooldown so the same sustained
    attack isn't re-counted every frame."""
    attacks = Counter()
    cooldown_until = {"A": -1e9, "B": -1e9}
    cooldown_frames = cfg.attack_cooldown_s * cfg.fps

    owner_team, owner_start, start_pos, last_pos = None, None, None, None
    spell_frames = 0
    for ev in events:
        if ev.type in ("possession_gain",):
            owner_team, owner_start = ev.team, ev.frame
            start_pos = ev.position
            last_pos = ev.position
            spell_frames = 0
        elif ev.type in ("touch",) and owner_team is not None:
            spell_frames += 1
            last_pos = ev.position or last_pos
        elif ev.type == "possession_loss" and owner_team is not None:
            if (spell_frames >= cfg.attack_min_possession_frames
                    and start_pos is not None and last_pos is not None
                    and ev.frame - cooldown_until[owner_team] > cooldown_frames):
                direction = cfg.attack_direction.get(owner_team, 1)
                dx_forward = (last_pos[0] - start_pos[0]) * direction
                attacking_third_x = (
                    cfg.frame_w * (1 - cfg.attacking_third_fraction) if direction > 0
                    else cfg.frame_w * cfg.attacking_third_fraction
                )
                entered_attacking_third = (
                    last_pos[0] >= attacking_third_x if direction > 0 else last_pos[0] <= attacking_third_x
                )
                if dx_forward > 0 and entered_attacking_third:
                    attacks[owner_team] += 1
                    cooldown_until[owner_team] = ev.frame
            owner_team, start_pos, last_pos, spell_frames = None, None, None, 0

    return {"A": attacks.get("A", 0), "B": attacks.get("B", 0)}


def _goal_position(team: str, cfg: StatsConfig) -> Tuple[float, float]:
    """Real-world pitch position (metres) of the goal `team` is
    attacking, using the pitch-configuration vertices (cm -> m)."""
    direction = cfg.attack_direction.get(team, 1)
    goal_x_m = (CONFIG.length / 100.0) if direction > 0 else 0.0
    goal_y_m = (CONFIG.width / 100.0) / 2.0
    return goal_x_m, goal_y_m


def classify_progressive_passes(successful_passes: List[Event], transformer: CoordinateTransformer,
                                 cfg: StatsConfig) -> List[Event]:
    progressive = []
    for ev in successful_passes:
        team = ev.data["from_team"]
        goal = _goal_position(team, cfg)
        from_m = transformer.to_pitch(*ev.data["from_position"], frame=ev.data.get("from_frame"))
        to_m = transformer.to_pitch(*ev.data["to_position"], frame=ev.data.get("to_frame"))
        d_from = float(np.hypot(from_m[0] - goal[0], from_m[1] - goal[1]))
        d_to = float(np.hypot(to_m[0] - goal[0], to_m[1] - goal[1]))
        reduction_frac = (d_from - d_to) / d_from if d_from > 0 else 0
        length_m = ev.data["travel_m"]
        is_prog = (reduction_frac >= cfg.progressive_pass_min_reduction_fraction
                   and length_m >= cfg.progressive_pass_min_length_m)
        ev.data["progressive"] = is_prog
        ev.data["goal_distance_reduction_pct"] = round(reduction_frac * 100, 1)
        if is_prog:
            progressive.append(ev)
    return progressive


def classify_through_balls(successful_passes: List[Event], tracking_data: List[dict],
                            transformer: CoordinateTransformer, cfg: StatsConfig) -> List[Event]:
    """A through ball must (1) be forward/progressive and (2) put the
    receiver beyond the opponent's most advanced outfield defender at
    the moment of release."""
    frames = group_by_frame(tracking_data)
    through = []
    for ev in successful_passes:
        team = ev.data["from_team"]
        opponent = "B" if team == "A" else "A"
        direction = cfg.attack_direction.get(team, 1)

        release_frame = None
        for f in sorted(frames.keys()):
            if f <= ev.frame:
                release_frame = f
            else:
                break
        if release_frame is None:
            continue

        opponents = [e for e in frames[release_frame]
                     if e.get("class") == "player" and e.get("team") == opponent]
        if not opponents:
            continue
        defender_line_x = (max(o["center"][0] for o in opponents) if direction > 0
                            else min(o["center"][0] for o in opponents))

        from_x = ev.data["from_position"][0]
        to_x = ev.data["to_position"][0]
        receiver_beyond_line = (to_x > defender_line_x) if direction > 0 else (to_x < defender_line_x)
        was_behind_before = (from_x <= defender_line_x) if direction > 0 else (from_x >= defender_line_x)

        if receiver_beyond_line and was_behind_before and ev.data.get("progressive"):
            ev.data["through_ball"] = True
            through.append(ev)
        else:
            ev.data["through_ball"] = False
    return through


def classify_crosses(successful_passes: List[Event], cfg: StatsConfig) -> List[Event]:
    """A cross originates from a wide channel in the attacking half and
    lands inside the opponent's penalty area."""
    crosses = []
    pen_x_frac = cfg.penalty_area_length_fraction
    pen_w_frac = cfg.penalty_area_width_fraction
    half_w_margin = (1 - pen_w_frac) / 2.0

    for ev in successful_passes:
        team = ev.data["from_team"]
        direction = cfg.attack_direction.get(team, 1)
        fy = ev.data["from_position"][1] / cfg.frame_h
        from_wide = fy <= cfg.cross_wide_zone_fraction or fy >= (1 - cfg.cross_wide_zone_fraction)

        tx = ev.data["to_position"][0] / cfg.frame_w
        ty = ev.data["to_position"][1] / cfg.frame_h
        in_box_x = tx >= (1 - pen_x_frac) if direction > 0 else tx <= pen_x_frac
        in_box_y = half_w_margin <= ty <= (1 - half_w_margin)

        is_cross = from_wide and in_box_x and in_box_y
        ev.data["cross"] = is_cross
        if is_cross:
            crosses.append(ev)
    return crosses


# ────────────────────────────────────────────────────────────
#  (b) TRAJECTORY-BASED STATISTICS
# ────────────────────────────────────────────────────────────
def compute_distance_and_speed(tracking_data: List[dict], transformer: CoordinateTransformer,
                                cfg: StatsConfig) -> Dict[str, dict]:
    """Per-player distance covered and speed profile, computed strictly
    from smoothed positions -> real elapsed time -> speed. Frame gaps
    beyond `motion_max_frame_gap` are skipped entirely rather than
    bridged."""
    tracks = defaultdict(list)
    teams = {}
    for e in tracking_data:
        if e.get("class") != "player":
            continue
        tracks[e["track_id"]].append(e)
        if e.get("team"):
            teams[e["track_id"]] = e["team"]

    out = {}
    for tid, pts in tracks.items():
        pts.sort(key=lambda e: e["frame"])
        pts = smooth_player_trajectory(pts, cfg)

        dist_m = 0.0
        speeds = []
        for p, c in zip(pts, pts[1:]):
            gap = c["frame"] - p["frame"]
            if gap <= 0 or gap > cfg.motion_max_frame_gap:
                continue
            d = transformer.distance_m(p["center"], c["center"], frame1=p["frame"], frame2=c["frame"])
            dt = gap / cfg.fps
            spd = d / dt
            if spd > cfg.max_player_speed_ms:
                continue
            dist_m += d
            speeds.append(spd)

        out[str(tid)] = {
            "team": teams.get(tid),
            "distance_m": round(dist_m, 1),
            "avg_speed_ms": round(float(np.mean(speeds)), 2) if speeds else 0.0,
            "max_speed_ms": round(float(np.max(speeds)), 2) if speeds else 0.0,
            "_speeds": speeds,
        }
    return out


def compute_sprints(speed_profiles: Dict[str, dict], cfg: StatsConfig) -> Dict[str, dict]:
    """A sprint requires speed above threshold for a minimum duration,
    not a single fast frame."""
    min_run = max(1, int(cfg.sprint_min_duration_s * cfg.fps))
    per_player = {}
    team_counts = Counter()
    for tid, profile in speed_profiles.items():
        speeds = profile["_speeds"]
        run = 0
        count = 0
        for spd in speeds:
            if spd >= cfg.sprint_speed_threshold_ms:
                run += 1
            else:
                if run >= min_run:
                    count += 1
                run = 0
        if run >= min_run:
            count += 1
        per_player[tid] = count
        if profile.get("team"):
            team_counts[profile["team"]] += count
    return {"per_player": per_player, "per_team": dict(team_counts)}


def compute_ball_speed(tracking_data: List[dict], transformer: CoordinateTransformer, cfg: StatsConfig) -> dict:
    pts = smooth_ball_trajectory(tracking_data, cfg)
    if len(pts) < 2:
        return {"avg_ms": 0.0, "max_ms": 0.0, "avg_kmh": 0.0, "max_kmh": 0.0}
    speeds = []
    for p, c in zip(pts, pts[1:]):
        gap = c["frame"] - p["frame"]
        if gap <= 0 or gap > cfg.motion_max_frame_gap:
            continue
        spd = transformer.distance_m(p["center"], c["center"], frame1=p["frame"], frame2=c["frame"]) / (gap / cfg.fps)
        if spd > cfg.max_ball_speed_ms:
            continue
        speeds.append(spd)
    if not speeds:
        return {"avg_ms": 0.0, "max_ms": 0.0, "avg_kmh": 0.0, "max_kmh": 0.0}
    avg, peak = float(np.mean(speeds)), float(np.max(speeds))
    return {"avg_ms": round(avg, 2), "max_ms": round(peak, 2),
            "avg_kmh": round(avg * 3.6, 1), "max_kmh": round(peak * 3.6, 1)}


def compute_heatmap(tracking_data: List[dict], transformer: CoordinateTransformer, cfg: StatsConfig,
                     track_id: Optional[int] = None, team: Optional[str] = None,
                     bins: Tuple[int, int] = (30, 20)) -> dict:
    """Heatmap in PITCH coordinates (metres), not raw camera pixels — so
    it reflects real ground covered regardless of camera zoom/angle."""
    pts = [e for e in tracking_data if e.get("class") == "player"
           and (track_id is None or e["track_id"] == track_id)
           and (team is None or e.get("team") == team)]
    if not pts:
        return {"xedges": [], "yedges": [], "hist": [], "xs": [], "ys": []}
    xs, ys = [], []
    for e in pts:
        x_m, y_m = transformer.to_pitch(*e["center"], frame=e["frame"])
        xs.append(x_m)
        ys.append(y_m)
    hist, xedges, yedges = np.histogram2d(
        xs, ys, bins=bins,
        range=[[0, cfg.pitch_visible_m], [0, cfg.frame_h * cfg.pixel_to_meter_y]],
    )
    return {"xedges": xedges.tolist(), "yedges": yedges.tolist(), "hist": hist.T.tolist(),
            "xs": xs, "ys": ys}


def compute_passing_network(successful_passes: List[Event]) -> dict:
    """Node = player, Edge = successful pass, edge weight = pass count.
    Also returns each node's mean pitch-pixel position for spatial
    layout of the network graph."""
    edges = Counter()
    positions_sum: Dict[int, np.ndarray] = defaultdict(lambda: np.zeros(2))
    positions_count: Dict[int, int] = defaultdict(int)

    for ev in successful_passes:
        src, dst = ev.data["from_player"], ev.data["to_player"]
        if src is None or dst is None or src == dst:
            continue
        edges[(src, dst)] += 1
        positions_sum[src] += np.asarray(ev.data["from_position"], dtype=float)
        positions_count[src] += 1
        positions_sum[dst] += np.asarray(ev.data["to_position"], dtype=float)
        positions_count[dst] += 1

    nodes = sorted({p for pair in edges for p in pair})
    node_positions = {
        n: (positions_sum[n] / positions_count[n]).tolist()
        for n in nodes if positions_count[n] > 0
    }
    return {
        "nodes": nodes,
        "edges": [{"from": s, "to": d, "weight": w} for (s, d), w in edges.items()],
        "node_positions": node_positions,
    }


def _counter_by_team(pass_events: List[Event]) -> dict:
    out = {"A": 0, "B": 0}
    for ev in pass_events:
        team = ev.data.get("from_team")
        if team in out:
            out[team] += 1
    return out


# ────────────────────────────────────────────────────────────
#  ORCHESTRATOR
# ────────────────────────────────────────────────────────────
def run_full_analysis(tracking_data: List[dict], config: StatsConfig,
                       frame_homographies: Optional[dict] = None) -> dict:
    """Single entry point: wires the Event Engine + every stats module
    together and returns one clean results dict."""
    homography_store = HomographyStore(frame_homographies) if frame_homographies else None
    transformer = CoordinateTransformer(config, homography_store=homography_store)
    engine = EventEngine(config, transformer)
    events: List[Event] = engine.run(tracking_data)

    total_frames = int(max((e["frame"] for e in tracking_data), default=0)) + 1

    passes = compute_passes(events)
    progressive = classify_progressive_passes(passes["successful_pass_events"], transformer, config)
    through_balls = classify_through_balls(passes["successful_pass_events"], tracking_data, transformer, config)
    crosses = classify_crosses(passes["successful_pass_events"], config)

    speed_profiles = compute_distance_and_speed(tracking_data, transformer, config)
    sprints = compute_sprints(speed_profiles, config)

    per_player_clean = {
        tid: {k: v for k, v in p.items() if not k.startswith("_")}
        for tid, p in speed_profiles.items()
    }
    for tid, p in per_player_clean.items():
        p["sprints"] = sprints["per_player"].get(tid, 0)

    possession_pct = compute_possession_percentage(events, total_frames, config)

    return {
        "events": events,
        "match_time_s": round(total_frames / config.fps, 1),
        "possession_pct": possession_pct,
        "possession_uncontrolled_pct": round(max(0.0, 100 - sum(possession_pct.values())), 1),
        "touches": compute_touches(events),
        "passes": {
            "attempts": passes["attempts"],
            "completions": passes["completions"],
            "accuracy_pct": passes["accuracy_pct"],
            "progressive_count": _counter_by_team(progressive),
            "through_ball_count": _counter_by_team(through_balls),
            "cross_count": _counter_by_team(crosses),
        },
        "shots": compute_shots(events),
        "attacks": compute_attacks(events, config),
        "ball_speed": compute_ball_speed(tracking_data, transformer, config),
        "per_player": per_player_clean,
        "sprints_per_team": sprints["per_team"],
        "passing_network": compute_passing_network(passes["successful_pass_events"]),
        "homography_calibrated": frame_homographies is not None,
        "transformer": transformer,
    }


def serialize_results(results: dict) -> dict:
    """Returns a fully JSON-serialisable copy of `results` (Event
    dataclasses -> plain dicts). Drops the non-serialisable transformer."""
    def convert(obj):
        if isinstance(obj, Event):
            return dataclasses.asdict(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items() if k != "transformer"}
        if isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        return obj

    return convert(results)


def build_match_report_text(results: dict, team_names: Optional[Dict[str, str]] = None) -> str:
    """Renders a human-readable text report (used for the Streamlit
    'download report' option and for debugging).

    team_names: optional {"A": "...", "B": "..."} display-name override.
    Falls back to "Team A" / "Team B" when not provided, so existing
    callers keep working unchanged.
    """
    names = team_names or {}

    def label(t: str) -> str:
        return names.get(t) or f"Team {t}"

    lines = []

    def section(title):
        lines.append(f"\n{'=' * 54}\n  {title}\n{'=' * 54}")

    section("MATCH TIME")
    lines.append(f"  {results['match_time_s']} s analysed")

    if not results.get("homography_calibrated", False):
        lines.append("\n  Note: no pitch homography available — directional stats "
                      "(progressive passes, through balls, crosses, attacks, shot "
                      "angle) use an approximate goal position and should be read "
                      "as indicative, not exact.")

    section("POSSESSION (%)")
    for t, v in results["possession_pct"].items():
        lines.append(f"  {label(t)} : {v}%")
    lines.append(f"  Uncontrolled/loose ball : {results['possession_uncontrolled_pct']}%")

    section("PASSING")
    p = results["passes"]
    for t in ("A", "B"):
        att = p["attempts"].get(t, 0)
        comp = p["completions"].get(t, 0)
        lines.append(f"  {label(t)} : {comp}/{att} ({p['accuracy_pct'].get(t, 0)}%)  "
                      f"| progressive={p['progressive_count'].get(t, 0)}  "
                      f"through={p['through_ball_count'].get(t, 0)}  "
                      f"crosses={p['cross_count'].get(t, 0)}")


    section("SHOTS")
    for t, v in results["shots"]["per_team"].items():
        lines.append(f"  {label(t)} : {v}  (on target: {results['shots']['on_target_per_team'].get(t, UNAVAILABLE)})")

    section("ATTACKS")
    for t, v in results["attacks"].items():
        lines.append(f"  {label(t)} : {v}")

    section("SPRINTS (per team)")
    for t, v in results["sprints_per_team"].items():
        lines.append(f"  {label(t)} : {v}")

    section("BALL SPEED")
    bs = results["ball_speed"]
    lines.append(f"  Avg : {bs['avg_ms']} m/s ({bs['avg_kmh']} km/h)")


    section("TOP 10 PLAYERS BY DISTANCE")
    pp = sorted(results["per_player"].items(), key=lambda x: x[1]["distance_m"], reverse=True)[:10]
    lines.append(f"  {'ID':>5} {'Team':>5} {'Dist(m)':>8} {'AvgSpd':>7} {'MaxSpd':>7} {'Sprints':>8}")
    for tid, d in pp:
        lines.append(f"  {tid:>5} {str(d.get('team')):>5} {d['distance_m']:>8.1f} "
                      f"{d['avg_speed_ms']:>7.2f} {d['max_speed_ms']:>7.2f} {d['sprints']:>8}")

    return "\n".join(lines)

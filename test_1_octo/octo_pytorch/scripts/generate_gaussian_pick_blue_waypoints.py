#!/usr/bin/env python3
"""Generate waypoint JSON files for a simple pick-up-the-blue-object benchmark.

Design goals
------------
- Match the user's waypoint JSON format.
- Use a 2D Gaussian bias around the home pose in x-y (truncated to the training workspace).
- Sample a single region that covers the full training workspace.
- Keep z as fixed phases with small jitter.
- Sample dwell times independently for: home, pre, descend_open, close_hold, lift_hold.
- Make yaw camera-friendly by using a simple x-y dependent yaw map plus noise.
- Enforce minimum spacing between sampled x-y points.

Notes
-----
- Yaw is generated so that sign(yaw) matches sign(y), and |yaw| grows with both x and |y|.
- The default yaw map is a practical starting point, not a calibrated solution.
  If you later collect anchor poses, replace the mapping with your fitted function.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


# -----------------------------
# User-aligned defaults
# -----------------------------
HOME_POSE = [0.30, 0.00, 0.10, 0.00, 3.14, 0.00]  # [x, y, z, roll, pitch, yaw]
SEGMENT_TIME_S = 1.0
INSTRUCTION = "pick up the blue block"
BASE_NAME = "pick_up_the_blue_block"

# Training workspace requested by the user.
TRAIN_X_MIN, TRAIN_X_MAX = 0.15, 0.45
TRAIN_Y_MIN, TRAIN_Y_MAX = -0.20, 0.20

# Sampling workspace (use the full training workspace).
SAMPLE_X_MIN, SAMPLE_X_MAX = TRAIN_X_MIN, TRAIN_X_MAX
SAMPLE_Y_MIN, SAMPLE_Y_MAX = TRAIN_Y_MIN, TRAIN_Y_MAX

# 2D Gaussian around home pose.
MU_X, MU_Y = HOME_POSE[0], HOME_POSE[1]
SIGMA_X = 0.055
SIGMA_Y = 0.070

# Region quotas (single region).
REGION_QUOTAS = {
    "ALL": 100,
}

# z phases with small jitter.
Z_PRE_RANGE = (0.078, 0.082)
Z_GRASP_RANGE = (0.009, 0.012)
Z_LIFT_RANGE = (0.145, 0.155)

# Dwell-time ranges. Edit these if you want a different [3, x] policy.
DWELL_RANGES = {
    "home": (2.0, 4.0),
    "pre": (2.0, 4.0),
    "descend_open": (2.0, 4.0),
    "close_hold": (2.0, 4.0),
    "lift_hold": (2.0, 4.0),
}

# Gripper convention matches the user's example.
GRIPPER_OPEN = 1.0
GRIPPER_CLOSED = 0.0
ROLL_FIXED = 0.0
PITCH_FIXED = 3.14

# Yaw model.
YAW_MIN, YAW_MAX = -0.8, 0.8
# yaw_ref(x, y) = sign(y) * yaw_ref_max_abs * YAW_GAIN * x_norm * y_norm
#   x_norm = clamp((x - TRAIN_X_MIN) / (TRAIN_X_MAX - TRAIN_X_MIN), 0, 1)
#   y_norm = clamp(|y| / max(|TRAIN_Y_MIN|, |TRAIN_Y_MAX|), 0, 1)
YAW_REF_CLIP = (-0.8, 0.8)
YAW_GAIN = 3.0
YAW_SIGMA_BY_REGION = {
    "ALL": 0.12,
}

# Diversity constraints.
MIN_POINT_SPACING_M = 0.015
NEARBY_POINT_RADIUS_M = 0.03
MIN_YAW_DIFF_IF_NEARBY_RAD = 0.10

# Outer-region coverage helper.
FORCED_TAIL_COUNT = 5
TAIL_X_LOW_MAX = 0.22
TAIL_X_HIGH_MIN = 0.40
TAIL_ABS_Y_MIN = 0.14

MAX_TRIES_PER_SAMPLE = 4000


@dataclass
class EpisodeSpec:
    episode_idx: int
    region: str
    x: float
    y: float
    yaw: float
    z_pre: float
    z_grasp: float
    z_lift: float
    dwell_home: float
    dwell_pre: float
    dwell_descend_open: float
    dwell_close_hold: float
    dwell_lift_hold: float
    forced_tail: bool = False


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def trunc_normal(rng: random.Random, mean: float, std: float, lo: float, hi: float) -> float:
    for _ in range(2048):
        v = rng.gauss(mean, std)
        if lo <= v <= hi:
            return v
    # Fallback if the range is narrow.
    return clamp(mean, lo, hi)


def uniform_in_range(rng: random.Random, lo: float, hi: float) -> float:
    return rng.uniform(lo, hi)


def ellipse_radius(x: float, y: float) -> float:
    rx = (x - MU_X) / SIGMA_X
    ry = (y - MU_Y) / SIGMA_Y
    return math.sqrt(rx * rx + ry * ry)


def angle_diff(a: float, b: float) -> float:
    d = a - b
    while d > math.pi:
        d -= 2.0 * math.pi
    while d < -math.pi:
        d += 2.0 * math.pi
    return abs(d)


def yaw_ref(x: float, y: float) -> float:
    # Normalize x and y into [0, 1] so that:
    # - x closer (near TRAIN_X_MIN) => yaw magnitude closer to 0
    # - |y| closer to 0 => yaw magnitude closer to 0
    x_den = float(TRAIN_X_MAX - TRAIN_X_MIN)
    x_norm = (x - TRAIN_X_MIN) / x_den if x_den > 0 else 0.0
    x_norm = clamp(float(x_norm), 0.0, 1.0)

    y_den = float(max(abs(TRAIN_Y_MIN), abs(TRAIN_Y_MAX)))
    y_norm = abs(y) / y_den if y_den > 0 else 0.0
    y_norm = clamp(float(y_norm), 0.0, 1.0)

    yaw_ref_max_abs = float(max(abs(YAW_REF_CLIP[0]), abs(YAW_REF_CLIP[1])))
    mag = yaw_ref_max_abs * float(YAW_GAIN) * x_norm * y_norm
    if y == 0.0:
        return 0.0
    return clamp(math.copysign(mag, y), YAW_REF_CLIP[0], YAW_REF_CLIP[1])


def sample_yaw(rng: random.Random, x: float, y: float, region: str) -> float:
    sigma = YAW_SIGMA_BY_REGION[region]
    center = yaw_ref(x, y)
    for _ in range(2048):
        v = rng.gauss(center, sigma)
        if YAW_MIN <= v <= YAW_MAX:
            return v
    return clamp(center, YAW_MIN, YAW_MAX)


def sample_z_triplet(rng: random.Random) -> Tuple[float, float, float]:
    z_pre = uniform_in_range(rng, *Z_PRE_RANGE)
    z_grasp = uniform_in_range(rng, *Z_GRASP_RANGE)
    z_lift = uniform_in_range(rng, *Z_LIFT_RANGE)
    return z_pre, z_grasp, z_lift


def sample_dwell_times(rng: random.Random) -> Dict[str, float]:
    return {
        key: round(uniform_in_range(rng, lo, hi), 3)
        for key, (lo, hi) in DWELL_RANGES.items()
    }


def passes_diversity_constraints(existing: Sequence[EpisodeSpec], x: float, y: float, yaw: float) -> bool:
    for ep in existing:
        d = math.hypot(x - ep.x, y - ep.y)
        if d < MIN_POINT_SPACING_M:
            return False
    return True


def sample_xy_for_region(rng: random.Random, region: str, forced_tail: bool = False) -> Tuple[float, float]:
    # Single-region mode: sample across the full training workspace.
    x_lo, x_hi = TRAIN_X_MIN, TRAIN_X_MAX
    y_lo, y_hi = TRAIN_Y_MIN, TRAIN_Y_MAX

    for _ in range(MAX_TRIES_PER_SAMPLE):
        x = trunc_normal(rng, MU_X, SIGMA_X, x_lo, x_hi)
        y = trunc_normal(rng, MU_Y, SIGMA_Y, y_lo, y_hi)
        if forced_tail:
            if not (x < TAIL_X_LOW_MAX or x > TAIL_X_HIGH_MIN or abs(y) > TAIL_ABS_Y_MIN):
                continue
        return x, y

    raise RuntimeError(f"Could not sample x-y for region {region} after many tries.")


def build_episode_spec(rng: random.Random, episode_idx: int, region: str, existing: Sequence[EpisodeSpec], forced_tail: bool = False) -> EpisodeSpec:
    for _ in range(MAX_TRIES_PER_SAMPLE):
        x, y = sample_xy_for_region(rng, region, forced_tail=forced_tail)
        yaw = sample_yaw(rng, x, y, region)
        if not passes_diversity_constraints(existing, x, y, yaw):
            continue
        z_pre, z_grasp, z_lift = sample_z_triplet(rng)
        dwells = sample_dwell_times(rng)
        return EpisodeSpec(
            episode_idx=episode_idx,
            region=region,
            x=round(x, 4),
            y=round(y, 4),
            yaw=round(yaw, 4),
            z_pre=round(z_pre, 4),
            z_grasp=round(z_grasp, 4),
            z_lift=round(z_lift, 4),
            dwell_home=dwells["home"],
            dwell_pre=dwells["pre"],
            dwell_descend_open=dwells["descend_open"],
            dwell_close_hold=dwells["close_hold"],
            dwell_lift_hold=dwells["lift_hold"],
            forced_tail=forced_tail,
        )
    raise RuntimeError(f"Could not build a valid episode spec for episode {episode_idx}.")


def make_waypoint_json(ep: EpisodeSpec) -> Dict:
    home_pose = [HOME_POSE[0], HOME_POSE[1], HOME_POSE[2], ROLL_FIXED, PITCH_FIXED, HOME_POSE[5]]
    grasp_pose = [ep.x, ep.y, ep.z_grasp, ROLL_FIXED, PITCH_FIXED, ep.yaw]
    pre_pose = [ep.x, ep.y, ep.z_pre, ROLL_FIXED, PITCH_FIXED, ep.yaw]
    lift_pose = [ep.x, ep.y, ep.z_lift, ROLL_FIXED, PITCH_FIXED, ep.yaw]

    return {
        "name": f"{BASE_NAME}_episode_{ep.episode_idx:06d}",
        "instruction": INSTRUCTION,
        "segment_time_s": SEGMENT_TIME_S,
        "waypoints": [
            {
                "pose": [round(v, 4) for v in home_pose],
                "gripper": GRIPPER_OPEN,
                "dwell_time_s": ep.dwell_home,
            },
            {
                "pose": [round(v, 4) for v in pre_pose],
                "gripper": GRIPPER_OPEN,
                "dwell_time_s": ep.dwell_pre,
            },
            {
                "pose": [round(v, 4) for v in grasp_pose],
                "gripper": GRIPPER_OPEN,
                "dwell_time_s": ep.dwell_descend_open,
            },
            {
                "pose": [round(v, 4) for v in grasp_pose],
                "gripper": GRIPPER_CLOSED,
                "dwell_time_s": ep.dwell_close_hold,
            },
            {
                "pose": [round(v, 4) for v in lift_pose],
                "gripper": GRIPPER_CLOSED,
                "dwell_time_s": ep.dwell_lift_hold,
            },
        ],
    }


def write_manifest_csv(path: Path, episodes: Sequence[EpisodeSpec]) -> None:
    fieldnames = [
        "episode_id",
        "region",
        "forced_tail",
        "x",
        "y",
        "yaw",
        "z_pre",
        "z_grasp",
        "z_lift",
        "dwell_home",
        "dwell_pre",
        "dwell_descend_open",
        "dwell_close_hold",
        "dwell_lift_hold",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ep in episodes:
            writer.writerow(
                {
                    "episode_id": f"episode_{ep.episode_idx:06d}",
                    "region": ep.region,
                    "forced_tail": int(ep.forced_tail),
                    "x": ep.x,
                    "y": ep.y,
                    "yaw": ep.yaw,
                    "z_pre": ep.z_pre,
                    "z_grasp": ep.z_grasp,
                    "z_lift": ep.z_lift,
                    "dwell_home": ep.dwell_home,
                    "dwell_pre": ep.dwell_pre,
                    "dwell_descend_open": ep.dwell_descend_open,
                    "dwell_close_hold": ep.dwell_close_hold,
                    "dwell_lift_hold": ep.dwell_lift_hold,
                }
            )


def generate_episodes(num_episodes: int, seed: int) -> List[EpisodeSpec]:
    if num_episodes != sum(REGION_QUOTAS.values()):
        raise ValueError(
            f"This script is currently configured for exactly {sum(REGION_QUOTAS.values())} episodes, "
            f"but got {num_episodes}."
        )

    rng = random.Random(seed)
    episodes: List[EpisodeSpec] = []
    episode_idx = 1

    forced_tail_left = min(FORCED_TAIL_COUNT, num_episodes)
    for i in range(num_episodes):
        episodes.append(
            build_episode_spec(
                rng,
                episode_idx,
                "ALL",
                episodes,
                forced_tail=bool(i < forced_tail_left),
            )
        )
        episode_idx += 1

    return episodes


def save_waypoints(output_dir: Path, episodes: Sequence[EpisodeSpec]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for ep in episodes:
        payload = make_waypoint_json(ep)
        file_name = f"episode_{ep.episode_idx:06d}_wpts.json"
        with (output_dir / file_name).open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.write("\n")

    write_manifest_csv(output_dir / "episode_manifest.csv", episodes)

    summary = {
        "base_name": BASE_NAME,
        "instruction": INSTRUCTION,
        "segment_time_s": SEGMENT_TIME_S,
        "num_episodes": len(episodes),
        "home_pose": HOME_POSE,
        "training_workspace_xy": {
            "x": [TRAIN_X_MIN, TRAIN_X_MAX],
            "y": [TRAIN_Y_MIN, TRAIN_Y_MAX],
        },
        "sampling_workspace_xy": {
            "x": [SAMPLE_X_MIN, SAMPLE_X_MAX],
            "y": [SAMPLE_Y_MIN, SAMPLE_Y_MAX],
        },
        "sigma_xy_m": {"x": SIGMA_X, "y": SIGMA_Y},
        "region_quotas": REGION_QUOTAS,
        "yaw_model": {
            "mode": "sign(y) * yaw_ref_max_abs * yaw_gain * x_norm * y_norm",
            "yaw_min": YAW_MIN,
            "yaw_max": YAW_MAX,
            "yaw_ref_clip": list(YAW_REF_CLIP),
            "yaw_gain": YAW_GAIN,
            "x_norm_range": [TRAIN_X_MIN, TRAIN_X_MAX],
            "y_norm_max_abs": max(abs(TRAIN_Y_MIN), abs(TRAIN_Y_MAX)),
            "yaw_sigma_by_region": YAW_SIGMA_BY_REGION,
        },
        "z_ranges_m": {
            "pre": list(Z_PRE_RANGE),
            "grasp": list(Z_GRASP_RANGE),
            "lift": list(Z_LIFT_RANGE),
        },
        "dwell_ranges_s": {k: list(v) for k, v in DWELL_RANGES.items()},
        "diversity_constraints": {
            "min_point_spacing_m": MIN_POINT_SPACING_M,
            "nearby_point_radius_m": NEARBY_POINT_RADIUS_M,
            "min_yaw_diff_if_nearby_rad": MIN_YAW_DIFF_IF_NEARBY_RAD,
        },
    }
    with (output_dir / "generation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate waypoint JSON files for pick-up-the-blue-object data collection.")
    parser.add_argument("--output-dir", type=Path, default=Path("./generated_pick_blue_waypoints"), help="Directory to write JSON files into.")
    parser.add_argument("--num-episodes", type=int, default=100, help="Must currently be 100 to match the configured quota.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episodes = generate_episodes(num_episodes=args.num_episodes, seed=args.seed)
    save_waypoints(args.output_dir, episodes)
    print(f"Generated {len(episodes)} waypoint files in: {args.output_dir}")
    print(f"Manifest: {args.output_dir / 'episode_manifest.csv'}")
    print(f"Summary : {args.output_dir / 'generation_summary.json'}")


if __name__ == "__main__":
    main()

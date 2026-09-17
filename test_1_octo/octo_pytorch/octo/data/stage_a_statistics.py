from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

import octo.data.stage_a_tfds  # noqa: F401
from octo.data.dataset import make_dataset_from_rlds
from octo.utils.spec import ModuleSpec


DATASET_NAME = "myarm_stage_a_dataset"
STANDARDIZE_FN = ModuleSpec.create(
    "octo.data.oxe.oxe_standardization_transforms:myarm_stage_a_dataset_transform"
)
DEFAULT_IMAGE_OBS_KEYS = {"primary": "image_primary", "wrist": None}


def _stats_file_stem(builder_name: str) -> str:
    # Keep legacy filenames for the default builder (most configs expect this).
    if builder_name == DATASET_NAME:
        return DATASET_NAME
    return builder_name.replace("/", "__")


@dataclass
class EpisodeSummary:
    episode_id: str
    instruction: str
    num_steps: int
    num_frames: int
    duration_s: float
    approx_control_hz: float
    achieved_state_min: List[float]
    achieved_state_max: List[float]
    action_mean: List[float]
    action_std: List[float]
    action_min: List[float]
    action_max: List[float]
    commanded_tracking_error_mean: List[float]
    commanded_tracking_error_max_abs: List[float]
    first_timestamp_s: float
    last_timestamp_s: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "instruction": self.instruction,
            "num_steps": self.num_steps,
            "num_frames": self.num_frames,
            "duration_s": self.duration_s,
            "approx_control_hz": self.approx_control_hz,
            "achieved_state_min": self.achieved_state_min,
            "achieved_state_max": self.achieved_state_max,
            "action_mean": self.action_mean,
            "action_std": self.action_std,
            "action_min": self.action_min,
            "action_max": self.action_max,
            "commanded_tracking_error_mean": self.commanded_tracking_error_mean,
            "commanded_tracking_error_max_abs": self.commanded_tracking_error_max_abs,
            "first_timestamp_s": self.first_timestamp_s,
            "last_timestamp_s": self.last_timestamp_s,
        }


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _safe_list(arr: np.ndarray) -> List[float]:
    return np.asarray(arr, dtype=np.float64).tolist()


def summarize_raw_episode(episode_dir: str | Path) -> EpisodeSummary:
    episode_dir = Path(episode_dir)
    meta = _load_json(episode_dir / "meta.json")
    instruction = _load_text(episode_dir / "instruction.txt") if (episode_dir / "instruction.txt").exists() else meta.get("instruction", "")

    achieved = np.load(episode_dir / "achieved_state.npy").astype(np.float32)
    commanded = np.load(episode_dir / "commanded_state.npy").astype(np.float32)
    action_path = episode_dir / "commanded_action.npy"
    if not action_path.exists():
        action_path = episode_dir / "delta_action_from_get_coords.npy"
        if not action_path.exists():
            action_path = episode_dir / "action.npy"
    action = np.load(action_path).astype(np.float32)
    timestamps = np.load(episode_dir / "timestamps.npy").astype(np.float32)

    primary_dir = episode_dir / "frames" / "image_primary"
    if not primary_dir.exists():
        primary_dir = episode_dir / "frames" / "primary"
    num_frames = len(list(primary_dir.glob("*.jpg"))) if primary_dir.exists() else 0

    usable_steps = int(min(len(achieved), len(commanded), len(timestamps)))
    if usable_steps <= 0:
        raise ValueError(f"Episode has no usable steps: {episode_dir}")

    achieved = achieved[:usable_steps]
    commanded = commanded[:usable_steps]
    timestamps = timestamps[:usable_steps]
    if len(action) > 0:
        usable_actions = min(len(action), max(usable_steps - 1, 0))
        action = action[:usable_actions]
    else:
        action = np.zeros((0, 7), dtype=np.float32)

    duration_s = float(max(timestamps[-1] - timestamps[0], 0.0)) if len(timestamps) > 1 else 0.0
    approx_control_hz = float((usable_steps - 1) / duration_s) if duration_s > 1e-9 and usable_steps > 1 else float(meta.get("control_hz", 0.0))
    tracking_error = achieved - commanded

    if len(action) > 0:
        action_mean = _safe_list(action.mean(axis=0))
        action_std = _safe_list(action.std(axis=0))
        action_min = _safe_list(action.min(axis=0))
        action_max = _safe_list(action.max(axis=0))
    else:
        zero7 = np.zeros((7,), dtype=np.float32)
        action_mean = _safe_list(zero7)
        action_std = _safe_list(zero7)
        action_min = _safe_list(zero7)
        action_max = _safe_list(zero7)

    return EpisodeSummary(
        episode_id=episode_dir.name,
        instruction=instruction,
        num_steps=usable_steps,
        num_frames=num_frames,
        duration_s=duration_s,
        approx_control_hz=approx_control_hz,
        achieved_state_min=_safe_list(achieved.min(axis=0)),
        achieved_state_max=_safe_list(achieved.max(axis=0)),
        action_mean=action_mean,
        action_std=action_std,
        action_min=action_min,
        action_max=action_max,
        commanded_tracking_error_mean=_safe_list(tracking_error.mean(axis=0)),
        commanded_tracking_error_max_abs=_safe_list(np.abs(tracking_error).max(axis=0)),
        first_timestamp_s=float(timestamps[0]),
        last_timestamp_s=float(timestamps[-1]),
    )


def summarize_raw_dataset(raw_dir: str | Path) -> Dict[str, Any]:
    raw_dir = Path(raw_dir)
    episode_dirs = sorted(p for p in raw_dir.glob("episode_*") if p.is_dir())
    summaries = [summarize_raw_episode(ep) for ep in episode_dirs]

    total_steps = int(sum(s.num_steps for s in summaries))
    total_frames = int(sum(s.num_frames for s in summaries))
    total_duration_s = float(sum(s.duration_s for s in summaries))
    if summaries:
        hz_values = np.asarray([s.approx_control_hz for s in summaries], dtype=np.float64)
        per_episode_steps = np.asarray([s.num_steps for s in summaries], dtype=np.float64)
    else:
        hz_values = np.asarray([], dtype=np.float64)
        per_episode_steps = np.asarray([], dtype=np.float64)

    aggregate = {
        "raw_dir": str(raw_dir),
        "num_episodes": len(summaries),
        "num_steps_total": total_steps,
        "num_frames_total": total_frames,
        "total_duration_s": total_duration_s,
        "mean_episode_steps": float(per_episode_steps.mean()) if len(per_episode_steps) else 0.0,
        "median_episode_steps": float(np.median(per_episode_steps)) if len(per_episode_steps) else 0.0,
        "mean_control_hz": float(hz_values.mean()) if len(hz_values) else 0.0,
        "min_control_hz": float(hz_values.min()) if len(hz_values) else 0.0,
        "max_control_hz": float(hz_values.max()) if len(hz_values) else 0.0,
        "episodes": [s.to_dict() for s in summaries],
    }
    return aggregate


def compute_stage_a_dataset_statistics(
    *,
    data_dir: str,
    builder_name: str = DATASET_NAME,
    force_recompute: bool = False,
    dataset_statistics: Optional[str | Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute/load OXE-style action/proprio statistics for Stage-A TFDS data.

    The returned format matches the statistics structure expected by Octo/OXE datasets,
    so it can be used for normalization during training and denormalization during eval.
    """
    _, statistics = make_dataset_from_rlds(
        name=builder_name,
        data_dir=data_dir,
        tfds_name=builder_name,
        train=True,
        shuffle=False,
        image_obs_keys=DEFAULT_IMAGE_OBS_KEYS,
        proprio_obs_key="proprio",
        language_key="language_instruction",
        standardize_fn=STANDARDIZE_FN,
        action_proprio_normalization_type="normal",
        action_normalization_mask=[True, True, True, True, True, True, False],
        dataset_statistics=dataset_statistics,
        force_recompute_dataset_statistics=force_recompute,
    )

    def _to_builtin(value: Any):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, dict):
            return {k: _to_builtin(v) for k, v in value.items()}
        return value

    return _to_builtin(statistics)


def save_stage_a_statistics_bundle(
    *,
    builder_data_dir: str | Path,
    output_dir: str | Path | None = None,
    builder_name: str = DATASET_NAME,
    force_recompute: bool = False,
) -> Dict[str, Path]:
    builder_data_dir = Path(builder_data_dir).resolve()
    output_dir = Path(output_dir).resolve() if output_dir is not None else builder_data_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_stats = compute_stage_a_dataset_statistics(
        data_dir=str(builder_data_dir),
        builder_name=builder_name,
        force_recompute=force_recompute,
    )

    raw_dir = builder_data_dir / "downloads" / "manual" / "raw"
    raw_summary = summarize_raw_dataset(raw_dir) if raw_dir.exists() else {
        "raw_dir": str(raw_dir),
        "num_episodes": 0,
        "num_steps_total": 0,
        "num_frames_total": 0,
        "total_duration_s": 0.0,
        "mean_episode_steps": 0.0,
        "median_episode_steps": 0.0,
        "mean_control_hz": 0.0,
        "min_control_hz": 0.0,
        "max_control_hz": 0.0,
        "episodes": [],
    }

    generated_at = datetime.now(timezone.utc).isoformat()
    file_stem = _stats_file_stem(builder_name)
    dataset_stats_payload = {
        **dataset_stats,
        "dataset_name": builder_name,
        "tfds_name": builder_name,
        "source_format": "stage_a_raw_v0_0_5",
        "generated_at_utc": generated_at,
        "builder_data_dir": str(builder_data_dir),
    }

    summary_payload = {
        "dataset_name": builder_name,
        "tfds_name": builder_name,
        "source_format": "stage_a_raw_v0_0_5",
        "generated_at_utc": generated_at,
        "builder_data_dir": str(builder_data_dir),
        "raw_summary": raw_summary,
        "normalization_statistics_path": str(output_dir / f"{file_stem}_statistics.json"),
    }

    stats_path = output_dir / f"{file_stem}_statistics.json"
    summary_path = output_dir / f"{file_stem}_summary.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(dataset_stats_payload, f, indent=2)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    return {
        "statistics_json": stats_path,
        "summary_json": summary_path,
    }

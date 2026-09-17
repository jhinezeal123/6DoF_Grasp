from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from octo.robot.collector import load_absolute_waypoints
from octo.robot.config import RobotPipelineConfig, WorkspaceBounds
from octo.robot.stage_a_naming import format_episode_id, parse_episode_index, parse_waypoint_episode_index


@dataclass
class Sequence3D:
    sequence_id: str
    points_7d: np.ndarray
    instruction: str = ""
    source_path: str = ""
    source_kind: str = "waypoint"

    @property
    def points_xyz(self) -> np.ndarray:
        return self.points_7d[:, :3]


def _to_builtin(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {k: _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    return value


def _draw_workspace_box(ax, workspace: WorkspaceBounds) -> None:
    x0, y0, z0 = workspace.translation_min_m
    x1, y1, z1 = workspace.translation_max_m
    corners = np.array(
        [
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
        ],
        dtype=np.float64,
    )
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for i, j in edges:
        ax.plot(
            [corners[i, 0], corners[j, 0]],
            [corners[i, 1], corners[j, 1]],
            [corners[i, 2], corners[j, 2]],
            linestyle="--",
            linewidth=0.8,
            alpha=0.7,
        )


def _workspace_mask(points_7d: np.ndarray, workspace: WorkspaceBounds) -> np.ndarray:
    points_7d = np.asarray(points_7d, dtype=np.float64)
    tmin = np.asarray(workspace.translation_min_m, dtype=np.float64)
    tmax = np.asarray(workspace.translation_max_m, dtype=np.float64)
    rmin = np.asarray(workspace.rotation_min_rad, dtype=np.float64)
    rmax = np.asarray(workspace.rotation_max_rad, dtype=np.float64)
    trans_ok = np.logical_and(points_7d[:, :3] >= tmin[None, :], points_7d[:, :3] <= tmax[None, :]).all(axis=1)
    rot_ok = np.logical_and(points_7d[:, 3:6] >= rmin[None, :], points_7d[:, 3:6] <= rmax[None, :]).all(axis=1)
    return np.logical_and(trans_ok, rot_ok)


def _polyline_length_m(points_xyz: np.ndarray) -> float:
    if len(points_xyz) < 2:
        return 0.0
    diffs = np.diff(points_xyz.astype(np.float64), axis=0)
    return float(np.linalg.norm(diffs, axis=1).sum())


def compute_sequence_statistics(sequences: Sequence[Sequence3D], workspace: WorkspaceBounds) -> Dict[str, Any]:
    if not sequences:
        return {
            "num_sequences": 0,
            "num_points_total": 0,
            "workspace": _to_builtin(workspace.__dict__),
            "per_sequence": [],
        }

    all_points = np.concatenate([seq.points_7d for seq in sequences], axis=0).astype(np.float64)
    inside_mask = _workspace_mask(all_points, workspace)
    path_lengths = np.asarray([_polyline_length_m(seq.points_xyz) for seq in sequences], dtype=np.float64)

    per_sequence = []
    for seq in sequences:
        seq_mask = _workspace_mask(seq.points_7d, workspace)
        per_sequence.append(
            {
                "sequence_id": seq.sequence_id,
                "instruction": seq.instruction,
                "source_kind": seq.source_kind,
                "source_path": seq.source_path,
                "num_points": int(len(seq.points_7d)),
                "translation_min_m": _to_builtin(seq.points_7d[:, :3].min(axis=0)),
                "translation_max_m": _to_builtin(seq.points_7d[:, :3].max(axis=0)),
                "rotation_min_rad": _to_builtin(seq.points_7d[:, 3:6].min(axis=0)),
                "rotation_max_rad": _to_builtin(seq.points_7d[:, 3:6].max(axis=0)),
                "gripper_min": float(seq.points_7d[:, 6].min()),
                "gripper_max": float(seq.points_7d[:, 6].max()),
                "path_length_m": _polyline_length_m(seq.points_xyz),
                "workspace_out_of_bounds_points": int((~seq_mask).sum()),
                "workspace_out_of_bounds_fraction": float((~seq_mask).mean()) if len(seq_mask) else 0.0,
            }
        )

    return {
        "num_sequences": len(sequences),
        "num_points_total": int(len(all_points)),
        "workspace": _to_builtin(workspace.__dict__),
        "translation_min_m": _to_builtin(all_points[:, :3].min(axis=0)),
        "translation_max_m": _to_builtin(all_points[:, :3].max(axis=0)),
        "translation_mean_m": _to_builtin(all_points[:, :3].mean(axis=0)),
        "translation_std_m": _to_builtin(all_points[:, :3].std(axis=0)),
        "rotation_min_rad": _to_builtin(all_points[:, 3:6].min(axis=0)),
        "rotation_max_rad": _to_builtin(all_points[:, 3:6].max(axis=0)),
        "rotation_mean_rad": _to_builtin(all_points[:, 3:6].mean(axis=0)),
        "rotation_std_rad": _to_builtin(all_points[:, 3:6].std(axis=0)),
        "gripper_min": float(all_points[:, 6].min()),
        "gripper_max": float(all_points[:, 6].max()),
        "gripper_mean": float(all_points[:, 6].mean()),
        "gripper_std": float(all_points[:, 6].std()),
        "workspace_out_of_bounds_points": int((~inside_mask).sum()),
        "workspace_out_of_bounds_fraction": float((~inside_mask).mean()) if len(inside_mask) else 0.0,
        "path_length_total_m": float(path_lengths.sum()),
        "path_length_mean_m": float(path_lengths.mean()) if len(path_lengths) else 0.0,
        "path_length_median_m": float(np.median(path_lengths)) if len(path_lengths) else 0.0,
        "per_sequence": per_sequence,
    }


def plot_sequences_3d(
    sequences: Sequence[Sequence3D],
    *,
    workspace: WorkspaceBounds,
    title: str,
    output_png: str | Path,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_xlim(workspace.translation_min_m[0], workspace.translation_max_m[0])
    ax.set_ylim(workspace.translation_min_m[1], workspace.translation_max_m[1])
    ax.set_zlim(workspace.translation_min_m[2], workspace.translation_max_m[2])
    _draw_workspace_box(ax, workspace)

    for idx, seq in enumerate(sequences):
        pts = seq.points_xyz.astype(np.float64)
        if len(pts) == 0:
            continue
        label = seq.sequence_id if idx < 12 else None
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], linewidth=1.8, alpha=0.85, label=label)
        ax.scatter(pts[0, 0], pts[0, 1], pts[0, 2], marker="o", s=30, alpha=0.9)
        ax.scatter(pts[-1, 0], pts[-1, 1], pts[-1, 2], marker="^", s=36, alpha=0.9)

    if sequences:
        ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    plt.close(fig)
    return output_png


def _sorted_waypoint_files(
    waypoint_dir: str | Path,
    *,
    waypoint_glob: str = "episode_*_wpts.json",
    start_episode: Optional[int] = None,
    end_episode: Optional[int] = None,
) -> List[Path]:
    waypoint_dir = Path(waypoint_dir)
    items = []
    for path in waypoint_dir.glob(waypoint_glob):
        idx = parse_waypoint_episode_index(path)
        if idx is None:
            continue
        if start_episode is not None and idx < start_episode:
            continue
        if end_episode is not None and idx > end_episode:
            continue
        items.append((idx, path))
    return [path for _, path in sorted(items, key=lambda x: x[0])]


def load_waypoint_sequences(
    waypoint_dir: str | Path,
    *,
    waypoint_glob: str = "episode_*_wpts.json",
    start_episode: Optional[int] = None,
    end_episode: Optional[int] = None,
) -> List[Sequence3D]:
    sequences: List[Sequence3D] = []
    for path in _sorted_waypoint_files(
        waypoint_dir,
        waypoint_glob=waypoint_glob,
        start_episode=start_episode,
        end_episode=end_episode,
    ):
        idx = parse_waypoint_episode_index(path)
        if idx is None:
            continue
        waypoints, _, meta = load_absolute_waypoints(path)
        points_7d = np.stack(
            [np.concatenate([wp.pose_m_rad, [wp.gripper_open]], axis=0) for wp in waypoints],
            axis=0,
        ).astype(np.float32)
        sequences.append(
            Sequence3D(
                sequence_id=format_episode_id(idx),
                points_7d=points_7d,
                instruction=str(meta.get("instruction", "")),
                source_path=str(path),
                source_kind="waypoint",
            )
        )
    return sequences


def save_waypoint_visualization_bundle(
    *,
    robot_config_path: str | Path,
    waypoint_dir: str | Path,
    output_dir: str | Path,
    waypoint_glob: str = "episode_*_wpts.json",
    start_episode: Optional[int] = None,
    end_episode: Optional[int] = None,
) -> Dict[str, Path]:
    cfg = RobotPipelineConfig.from_json(robot_config_path)
    sequences = load_waypoint_sequences(
        waypoint_dir,
        waypoint_glob=waypoint_glob,
        start_episode=start_episode,
        end_episode=end_episode,
    )
    stats = compute_sequence_statistics(sequences, cfg.robot.workspace)
    stats.update(
        {
            "source_type": "waypoint_dir",
            "waypoint_dir": str(Path(waypoint_dir).resolve()),
            "waypoint_glob": waypoint_glob,
            "config_path": str(Path(robot_config_path).resolve()),
            "start_episode": start_episode,
            "end_episode": end_episode,
        }
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "stage_a_waypoints_3d.png"
    json_path = output_dir / "stage_a_waypoints_3d_stats.json"
    plot_sequences_3d(sequences, workspace=cfg.robot.workspace, title="Stage-A waypoint sequences", output_png=png_path)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_to_builtin(stats), f, indent=2)
    return {"visualization_png": png_path, "statistics_json": json_path}


def load_raw_dataset_sequences(raw_dir: str | Path) -> List[Sequence3D]:
    raw_dir = Path(raw_dir)
    sequences: List[Sequence3D] = []
    for episode_dir in sorted(p for p in raw_dir.glob("episode_*") if p.is_dir()):
        meta_path = episode_dir / "meta.json"
        instruction = ""
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            instruction = str(meta.get("instruction", ""))
        achieved_path = episode_dir / "achieved_state.npy"
        if not achieved_path.exists():
            continue
        points_7d = np.load(achieved_path).astype(np.float32)
        if points_7d.ndim != 2 or points_7d.shape[1] != 7 or len(points_7d) == 0:
            continue
        sequences.append(
            Sequence3D(
                sequence_id=episode_dir.name,
                points_7d=points_7d,
                instruction=instruction,
                source_path=str(episode_dir),
                source_kind="raw_episode",
            )
        )
    return sequences


def save_raw_dataset_visualization_bundle(
    *,
    robot_config_path: str | Path,
    raw_dir: str | Path,
    output_dir: str | Path,
) -> Dict[str, Path]:
    cfg = RobotPipelineConfig.from_json(robot_config_path)
    sequences = load_raw_dataset_sequences(raw_dir)
    stats = compute_sequence_statistics(sequences, cfg.robot.workspace)
    stats.update(
        {
            "source_type": "raw_dataset",
            "raw_dir": str(Path(raw_dir).resolve()),
            "config_path": str(Path(robot_config_path).resolve()),
        }
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "myarm_stage_a_dataset_3d.png"
    json_path = output_dir / "myarm_stage_a_dataset_3d_stats.json"
    plot_sequences_3d(sequences, workspace=cfg.robot.workspace, title="Stage-A raw dataset trajectories", output_png=png_path)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_to_builtin(stats), f, indent=2)
    return {"visualization_png": png_path, "statistics_json": json_path}

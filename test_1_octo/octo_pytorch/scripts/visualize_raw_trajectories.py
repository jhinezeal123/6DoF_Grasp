#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

from octo.robot.config import RobotPipelineConfig, WorkspaceBounds
from octo.robot.stage_a_naming import format_episode_id, parse_episode_index
from octo.robot.utils import (
	rotation_angle_between_matrices,
	rpy_to_rotation_matrix,
	wrap_to_pi,
)


@dataclass
class EpisodeTrajectory:
	episode_id: str
	episode_dir: Path
	achieved_7d: np.ndarray
	delta_actions_7d: np.ndarray
	integrated_7d: np.ndarray
	action_path: Path
	translation_error_m: np.ndarray
	rotation_error_rad: np.ndarray

	@property
	def num_steps(self) -> int:
		return int(self.integrated_7d.shape[0])


@dataclass
class InteractionPoint:
	episode_id: str
	episode_dir: Path
	step_index: int
	xyz_m: np.ndarray
	gripper_open_before: float
	gripper_open_after: float


def _parse_args():
	ap = argparse.ArgumentParser(
		description=(
			"Visualize Stage-A raw dataset trajectories by integrating delta_action_from_get_coords.npy. "
			"Exports 2D projections and a 3D plot in the robot base frame."
		)
	)
	ap.add_argument("--config", required=True, help="Robot/camera JSON config (used for workspace + home pose)")
	ap.add_argument(
		"--raw_dir",
		required=True,
		help="Path to raw/ directory containing episode_xxxxxx folders",
	)
	ap.add_argument(
		"--output_dir",
		default=None,
		help="Where to write PNG + JSON outputs. Default: <raw_dir>/../viz_delta_actions",
	)
	ap.add_argument(
		"--start_episode",
		"--strat_episode",
		dest="start_episode",
		default=None,
		help="Inclusive start episode selector, e.g. 12 or episode_000012",
	)
	ap.add_argument(
		"--end_episode",
		default=None,
		help="Inclusive end episode selector, e.g. 20 or episode_000020",
	)
	ap.add_argument(
		"--all",
		action="store_true",
		help="Visualize all episodes under raw_dir (ignores start/end)",
	)
	ap.add_argument(
		"--episode_glob",
		default="episode_*",
		help="Episode folder glob under raw_dir",
	)
	ap.add_argument(
		"--axis_len",
		type=float,
		default=0.05,
		help="Axis length (m) for drawing base and end-effector coordinate frames",
	)
	return ap.parse_args()


def _draw_workspace_box_3d(ax, workspace: WorkspaceBounds) -> None:
	x0, y0, z0 = workspace.translation_min_m
	x1, y1, z1 = workspace.translation_max_m
	corners = np.array(
		[
			[x0, y0, z0],
			[x1, y0, z0],
			[x1, y1, z0],
			[x0, y1, z0],
			[x0, y0, z1],
			[x1, y0, z1],
			[x1, y1, z1],
			[x0, y1, z1],
		],
		dtype=np.float64,
	)
	edges = [
		(0, 1),
		(1, 2),
		(2, 3),
		(3, 0),
		(4, 5),
		(5, 6),
		(6, 7),
		(7, 4),
		(0, 4),
		(1, 5),
		(2, 6),
		(3, 7),
	]
	for i, j in edges:
		ax.plot(
			[corners[i, 0], corners[j, 0]],
			[corners[i, 1], corners[j, 1]],
			[corners[i, 2], corners[j, 2]],
			linestyle="--",
			linewidth=0.8,
			alpha=0.6,
			color="gray",
		)


def _draw_frame_3d(
	ax,
	*,
	origin_xyz: Sequence[float],
	rotation: np.ndarray,
	axis_len: float,
	alpha: float = 1.0,
	label_prefix: Optional[str] = None,
) -> None:
	origin = np.asarray(origin_xyz, dtype=np.float64).reshape(3)
	r = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
	# Columns are the frame axes expressed in base/world coordinates.
	axes = [r[:, 0], r[:, 1], r[:, 2]]
	colors = ["r", "g", "b"]
	names = ["x", "y", "z"]
	for vec, color, name in zip(axes, colors, names):
		v = vec * float(axis_len)
		ax.quiver(
			origin[0],
			origin[1],
			origin[2],
			v[0],
			v[1],
			v[2],
			color=color,
			linewidth=2.0,
			alpha=float(alpha),
			arrow_length_ratio=0.25,
		)
		if label_prefix:
			ax.text(
				origin[0] + v[0],
				origin[1] + v[1],
				origin[2] + v[2],
				f"{label_prefix}{name}",
				color=color,
				fontsize=8,
			)


def _episode_dirs_in_range(
	raw_dir: Path,
	episode_glob: str,
	*,
	start_episode: Optional[int],
	end_episode: Optional[int],
) -> List[Tuple[int, Path]]:
	items: List[Tuple[int, Path]] = []
	for episode_dir in raw_dir.glob(episode_glob):
		if not episode_dir.is_dir():
			continue
		try:
			idx = parse_episode_index(episode_dir.name)
		except Exception:
			continue
		if idx is None:
			continue
		if start_episode is not None and idx < start_episode:
			continue
		if end_episode is not None and idx > end_episode:
			continue
		items.append((int(idx), episode_dir))
	items.sort(key=lambda x: x[0])
	return items


def _load_delta_actions(episode_dir: Path) -> Tuple[np.ndarray, Path]:
	action_path = episode_dir / "delta_action_from_get_coords.npy"
	if not action_path.exists():
		action_path = episode_dir / "action.npy"
	if not action_path.exists():
		raise FileNotFoundError(f"Missing delta-action file in {episode_dir}: expected delta_action_from_get_coords.npy")
	actions = np.load(action_path).astype(np.float64)
	if actions.ndim != 2 or actions.shape[1] < 6:
		raise ValueError(f"Invalid delta-action array shape {actions.shape} in {action_path}")
	if actions.shape[1] == 6:
		# Add a dummy gripper column so downstream code is uniform.
		actions = np.concatenate([actions, np.full((actions.shape[0], 1), np.nan, dtype=np.float64)], axis=1)
	return actions[:, :7], action_path


def _integrate_delta_actions(
	*,
	delta_actions_7d: np.ndarray,
	start_pose_7d: np.ndarray,
) -> np.ndarray:
	actions = np.asarray(delta_actions_7d, dtype=np.float64)
	start = np.asarray(start_pose_7d, dtype=np.float64).reshape(7)
	out = np.zeros((actions.shape[0] + 1, 7), dtype=np.float64)
	out[0] = start

	for t in range(actions.shape[0]):
		out[t + 1, :6] = out[t, :6] + actions[t, :6]
		out[t + 1, 3:6] = wrap_to_pi(out[t + 1, 3:6])
		grip = actions[t, 6]
		if not np.isnan(grip):
			out[t + 1, 6] = float(grip)
		else:
			out[t + 1, 6] = out[t, 6]
	return out.astype(np.float32)


def _compute_pose_errors(
	*,
	achieved_7d: np.ndarray,
	integrated_7d: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
	n = int(min(len(achieved_7d), len(integrated_7d)))
	achieved = np.asarray(achieved_7d[:n], dtype=np.float64)
	integrated = np.asarray(integrated_7d[:n], dtype=np.float64)
	trans_err = np.linalg.norm(integrated[:, :3] - achieved[:, :3], axis=1).astype(np.float32)

	rot_err = np.zeros((n,), dtype=np.float32)
	for i in range(n):
		ra = rpy_to_rotation_matrix(*integrated[i, 3:6])
		rb = rpy_to_rotation_matrix(*achieved[i, 3:6])
		rot_err[i] = float(rotation_angle_between_matrices(ra, rb))
	return trans_err, rot_err


def _load_episode_trajectory(
	*,
	episode_idx: int,
	episode_dir: Path,
) -> EpisodeTrajectory:
	achieved_path = episode_dir / "achieved_state.npy"
	if not achieved_path.exists():
		raise FileNotFoundError(f"Missing achieved_state.npy in {episode_dir}")
	achieved = np.load(achieved_path).astype(np.float32)
	if achieved.ndim != 2 or achieved.shape[1] != 7 or len(achieved) < 2:
		raise ValueError(f"Invalid achieved_state shape {achieved.shape} in {achieved_path}")

	actions, action_path = _load_delta_actions(episode_dir)
	integrated = _integrate_delta_actions(delta_actions_7d=actions, start_pose_7d=achieved[0])

	# Align lengths (some episodes may have slightly mismatched arrays).
	n = int(min(len(achieved), len(integrated)))
	achieved = achieved[:n]
	integrated = integrated[:n]
	actions = actions[: max(0, n - 1)]

	trans_err, rot_err = _compute_pose_errors(achieved_7d=achieved, integrated_7d=integrated)

	return EpisodeTrajectory(
		episode_id=format_episode_id(int(episode_idx)),
		episode_dir=episode_dir,
		achieved_7d=achieved,
		delta_actions_7d=actions.astype(np.float32),
		integrated_7d=integrated,
		action_path=action_path,
		translation_error_m=trans_err,
		rotation_error_rad=rot_err,
	)


def _axis_limits_from_workspace_and_data(
	*,
	workspace: WorkspaceBounds,
	sequences: Sequence[EpisodeTrajectory],
) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
	pts = []
	for seq in sequences:
		if seq.integrated_7d.size:
			pts.append(np.asarray(seq.integrated_7d[:, :3], dtype=np.float64))
	if pts:
		all_pts = np.concatenate(pts, axis=0)
		data_min = all_pts.min(axis=0)
		data_max = all_pts.max(axis=0)
	else:
		data_min = np.asarray(workspace.translation_min_m, dtype=np.float64)
		data_max = np.asarray(workspace.translation_max_m, dtype=np.float64)

	ws_min = np.asarray(workspace.translation_min_m, dtype=np.float64)
	ws_max = np.asarray(workspace.translation_max_m, dtype=np.float64)
	base_origin = np.zeros(3, dtype=np.float64)

	lo = np.minimum(np.minimum(ws_min, data_min), base_origin)
	hi = np.maximum(np.maximum(ws_max, data_max), base_origin)

	# Add a small margin.
	span = np.maximum(hi - lo, 1e-6)
	margin = 0.04 * span
	lo = lo - margin
	hi = hi + margin

	return (float(lo[0]), float(hi[0])), (float(lo[1]), float(hi[1])), (float(lo[2]), float(hi[2]))


def _plot_trajectories_3d(
	*,
	sequences: Sequence[EpisodeTrajectory],
	workspace: WorkspaceBounds,
	home_pose_m_rad: Sequence[float],
	axis_len: float,
	output_png: Path,
) -> Path:
	import matplotlib

	matplotlib.use("Agg")
	import matplotlib.pyplot as plt

	output_png = Path(output_png)
	output_png.parent.mkdir(parents=True, exist_ok=True)

	fig = plt.figure(figsize=(10, 8))
	ax = fig.add_subplot(111, projection="3d")
	ax.set_title("Raw Stage-A trajectories (integrated from delta_action_from_get_coords)")
	ax.set_xlabel("X [m]")
	ax.set_ylabel("Y [m]")
	ax.set_zlabel("Z [m]")

	xlim, ylim, zlim = _axis_limits_from_workspace_and_data(workspace=workspace, sequences=sequences)
	ax.set_xlim(*xlim)
	ax.set_ylim(*ylim)
	ax.set_zlim(*zlim)

	_draw_workspace_box_3d(ax, workspace)

	# Base frame at origin.
	_draw_frame_3d(
		ax,
		origin_xyz=(0.0, 0.0, 0.0),
		rotation=np.eye(3, dtype=np.float64),
		axis_len=float(axis_len),
		alpha=0.9,
		label_prefix="base_",
	)

	# Home pose marker.
	home_pose = np.asarray(list(home_pose_m_rad), dtype=np.float64).reshape(6)
	ax.scatter([home_pose[0]], [home_pose[1]], [home_pose[2]], marker="x", s=55, color="black", alpha=0.9)
	ax.text(home_pose[0], home_pose[1], home_pose[2], "home", color="black")

	num_episodes = int(len(sequences))
	cmap = plt.get_cmap("turbo")
	denom = max(1, num_episodes - 1)
	show_colorbar = num_episodes > 20
	show_legend = num_episodes <= 20
	show_markers = num_episodes <= 50

	if num_episodes <= 30:
		line_alpha = 0.9
		line_width = 1.8
	elif num_episodes <= 100:
		line_alpha = 0.65
		line_width = 1.0
	else:
		line_alpha = 0.5
		line_width = 0.7

	for i, seq in enumerate(sequences):
		pts = np.asarray(seq.integrated_7d[:, :3], dtype=np.float64)
		if len(pts) < 2:
			continue
		color = cmap(float(i) / float(denom))
		label = seq.episode_id if show_legend else None
		ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], linewidth=float(line_width), alpha=float(line_alpha), color=color, label=label)
		if show_markers:
			ax.scatter(pts[0, 0], pts[0, 1], pts[0, 2], marker="o", s=16, alpha=float(line_alpha), color=color)
			ax.scatter(pts[-1, 0], pts[-1, 1], pts[-1, 2], marker="^", s=18, alpha=float(line_alpha), color=color)

		# End-effector frame for the first sequence only (to reduce clutter).
		if i == 0:
			start_pose = np.asarray(seq.integrated_7d[0, :6], dtype=np.float64)
			end_pose = np.asarray(seq.integrated_7d[-1, :6], dtype=np.float64)
			_draw_frame_3d(
				ax,
				origin_xyz=start_pose[:3],
				rotation=rpy_to_rotation_matrix(*start_pose[3:6]),
				axis_len=float(axis_len) * 0.8,
				alpha=0.95,
				label_prefix="eef0_",
			)
			_draw_frame_3d(
				ax,
				origin_xyz=end_pose[:3],
				rotation=rpy_to_rotation_matrix(*end_pose[3:6]),
				axis_len=float(axis_len) * 0.8,
				alpha=0.95,
				label_prefix="eefT_",
			)

	if show_legend and sequences:
		ax.legend(loc="upper left", fontsize=8)
	elif show_colorbar and sequences:
		norm = matplotlib.colors.Normalize(vmin=0, vmax=float(denom))
		sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
		sm.set_array([])
		cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
		cbar.set_label("episode index (sorted)")
		ticks, labels = _episode_colorbar_ticks_and_labels([s.episode_id for s in sequences])
		if ticks:
			cbar.set_ticks(ticks)
			cbar.set_ticklabels(labels)
	fig.tight_layout()
	fig.savefig(output_png, dpi=180)
	plt.close(fig)
	return output_png


def _draw_workspace_rect_2d(ax, workspace: WorkspaceBounds, *, plane: str) -> None:
	plane = plane.upper()
	tmin = np.asarray(workspace.translation_min_m, dtype=np.float64)
	tmax = np.asarray(workspace.translation_max_m, dtype=np.float64)

	if plane == "XY":
		lo = (tmin[0], tmin[1])
		hi = (tmax[0], tmax[1])
		ax.set_xlabel("X [m]")
		ax.set_ylabel("Y [m]")
	elif plane == "XZ":
		lo = (tmin[0], tmin[2])
		hi = (tmax[0], tmax[2])
		ax.set_xlabel("X [m]")
		ax.set_ylabel("Z [m]")
	elif plane == "YZ":
		lo = (tmin[1], tmin[2])
		hi = (tmax[1], tmax[2])
		ax.set_xlabel("Y [m]")
		ax.set_ylabel("Z [m]")
	else:
		raise ValueError(f"Unsupported plane: {plane}")

	x0, y0 = lo
	x1, y1 = hi
	ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], linestyle="--", linewidth=0.8, color="gray", alpha=0.6)


def _plot_trajectories_2d(
	*,
	sequences: Sequence[EpisodeTrajectory],
	workspace: WorkspaceBounds,
	home_pose_m_rad: Sequence[float],
	output_png: Path,
) -> Path:
	import matplotlib

	matplotlib.use("Agg")
	import matplotlib.pyplot as plt

	output_png = Path(output_png)
	output_png.parent.mkdir(parents=True, exist_ok=True)

	fig, axes = plt.subplots(1, 3, figsize=(16, 5))
	planes = ["XY", "XZ", "YZ"]

	num_episodes = int(len(sequences))
	cmap = plt.get_cmap("turbo")
	denom = max(1, num_episodes - 1)
	show_colorbar = num_episodes > 20
	show_legend = num_episodes <= 20

	if num_episodes <= 30:
		line_alpha = 0.9
		line_width = 1.3
	elif num_episodes <= 100:
		line_alpha = 0.7
		line_width = 0.9
	else:
		line_alpha = 0.55
		line_width = 0.6

	home_pose = np.asarray(list(home_pose_m_rad), dtype=np.float64).reshape(6)

	for ax, plane in zip(axes, planes):
		ax.set_title(f"{plane} projection")
		ax.grid(True, alpha=0.25)
		ax.set_aspect("equal", adjustable="box")
		_draw_workspace_rect_2d(ax, workspace, plane=plane)

		# Base origin marker + axis hints.
		ax.scatter([0.0], [0.0], s=18, color="black", alpha=0.8)
		if plane == "XY":
			ax.arrow(0.0, 0.0, 0.06, 0.0, head_width=0.01, color="r", alpha=0.7, length_includes_head=True)
			ax.arrow(0.0, 0.0, 0.0, 0.06, head_width=0.01, color="g", alpha=0.7, length_includes_head=True)
		elif plane == "XZ":
			ax.arrow(0.0, 0.0, 0.06, 0.0, head_width=0.01, color="r", alpha=0.7, length_includes_head=True)
			ax.arrow(0.0, 0.0, 0.0, 0.06, head_width=0.01, color="b", alpha=0.7, length_includes_head=True)
		elif plane == "YZ":
			ax.arrow(0.0, 0.0, 0.06, 0.0, head_width=0.01, color="g", alpha=0.7, length_includes_head=True)
			ax.arrow(0.0, 0.0, 0.0, 0.06, head_width=0.01, color="b", alpha=0.7, length_includes_head=True)

		# Home marker in this plane.
		if plane == "XY":
			ax.scatter([home_pose[0]], [home_pose[1]], marker="x", s=45, color="black", alpha=0.85)
		elif plane == "XZ":
			ax.scatter([home_pose[0]], [home_pose[2]], marker="x", s=45, color="black", alpha=0.85)
		else:
			ax.scatter([home_pose[1]], [home_pose[2]], marker="x", s=45, color="black", alpha=0.85)

		for i, seq in enumerate(sequences):
			pts = np.asarray(seq.integrated_7d[:, :3], dtype=np.float64)
			if len(pts) < 2:
				continue
			color = cmap(float(i) / float(denom))
			label = seq.episode_id if (show_legend and plane == "XY") else None
			if plane == "XY":
				ax.plot(pts[:, 0], pts[:, 1], linewidth=float(line_width), alpha=float(line_alpha), color=color, label=label)
			elif plane == "XZ":
				ax.plot(pts[:, 0], pts[:, 2], linewidth=float(line_width), alpha=float(line_alpha), color=color)
			else:
				ax.plot(pts[:, 1], pts[:, 2], linewidth=float(line_width), alpha=float(line_alpha), color=color)

		if show_legend and plane == "XY" and sequences:
			ax.legend(loc="upper left", fontsize=8)

	fig.suptitle("Raw Stage-A trajectories (integrated from delta_action_from_get_coords)")
	if show_colorbar and sequences:
		fig.tight_layout(rect=[0.0, 0.0, 0.88, 0.93])
	else:
		fig.tight_layout()

	if show_colorbar and sequences:
		norm = matplotlib.colors.Normalize(vmin=0, vmax=float(denom))
		sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
		sm.set_array([])
		cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
		cbar.set_label("episode index (sorted)")
		ticks, labels = _episode_colorbar_ticks_and_labels([s.episode_id for s in sequences])
		if ticks:
			cbar.set_ticks(ticks)
			cbar.set_ticklabels(labels)
	fig.savefig(output_png, dpi=180)
	plt.close(fig)
	return output_png


def _find_first_gripper_close_index(
	gripper_open: np.ndarray,
	*,
	threshold: float,
	sticky_steps: int = 1,
) -> Optional[int]:
	g = np.asarray(gripper_open, dtype=np.float64).reshape(-1)
	if g.size < 2:
		return None
	sticky_steps = max(1, int(sticky_steps))
	closed = g < float(threshold)
	for i in range(1, int(g.size)):
		if (not bool(closed[i - 1])) and bool(closed[i]):
			# Require the "closed" state to persist for sticky_steps (or until the end).
			end = min(int(g.size), int(i + sticky_steps))
			if bool(np.all(closed[i:end])):
				return int(i)
	return None


def _auto_gripper_close_threshold(
	gripper_open: np.ndarray,
	*,
	low_percentile: float = 5.0,
	high_percentile: float = 95.0,
	min_range: float = 0.02,
) -> Optional[float]:
	g = np.asarray(gripper_open, dtype=np.float64).reshape(-1)
	if g.size < 2:
		return None
	lo = float(np.percentile(g, float(low_percentile)))
	hi = float(np.percentile(g, float(high_percentile)))
	if not (math.isfinite(lo) and math.isfinite(hi)):
		return None
	if float(hi - lo) < float(min_range):
		return None
	return 0.5 * (hi + lo)


def _episode_colorbar_ticks_and_labels(
	episode_ids: Sequence[str],
	*,
	max_ticks: int = 6,
) -> Tuple[List[int], List[str]]:
	n = int(len(episode_ids))
	if n <= 0:
		return [], []
	if n == 1:
		idx = parse_episode_index(episode_ids[0])
		label = f"{int(idx):06d}" if idx is not None else str(episode_ids[0])
		return [0], [label]

	num = min(int(max_ticks), n)
	positions = np.linspace(0, n - 1, num=num).round().astype(int)
	labels: List[str] = []
	for p in positions:
		idx = parse_episode_index(episode_ids[int(p)])
		labels.append(f"{int(idx):06d}" if idx is not None else str(episode_ids[int(p)]))
	return positions.tolist(), labels


def _axis_limits_from_workspace_and_points(
	*,
	workspace: WorkspaceBounds,
	points_xyz: np.ndarray,
) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
	ws_min = np.asarray(workspace.translation_min_m, dtype=np.float64)
	ws_max = np.asarray(workspace.translation_max_m, dtype=np.float64)
	base_origin = np.zeros(3, dtype=np.float64)

	pts = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3) if np.asarray(points_xyz).size else np.zeros((0, 3), dtype=np.float64)
	if pts.size:
		data_min = pts.min(axis=0)
		data_max = pts.max(axis=0)
	else:
		data_min = ws_min
		data_max = ws_max

	lo = np.minimum(np.minimum(ws_min, data_min), base_origin)
	hi = np.maximum(np.maximum(ws_max, data_max), base_origin)
	span = np.maximum(hi - lo, 1e-6)
	margin = 0.04 * span
	lo = lo - margin
	hi = hi + margin

	return (float(lo[0]), float(hi[0])), (float(lo[1]), float(hi[1])), (float(lo[2]), float(hi[2]))


def _plot_interaction_points_3d(
	*,
	points: Sequence[InteractionPoint],
	workspace: WorkspaceBounds,
	home_pose_m_rad: Sequence[float],
	axis_len: float,
	gripper_threshold: float,
	output_png: Path,
) -> Path:
	import matplotlib

	matplotlib.use("Agg")
	import matplotlib.pyplot as plt

	output_png = Path(output_png)
	output_png.parent.mkdir(parents=True, exist_ok=True)

	fig = plt.figure(figsize=(10, 8))
	ax = fig.add_subplot(111, projection="3d")
	ax.set_title(f"Interaction points: first gripper close (auto threshold, median={gripper_threshold:.3f})")
	ax.set_xlabel("X [m]")
	ax.set_ylabel("Y [m]")
	ax.set_zlabel("Z [m]")

	pts_xyz = np.stack([p.xyz_m for p in points], axis=0) if points else np.zeros((0, 3), dtype=np.float64)
	xlim, ylim, zlim = _axis_limits_from_workspace_and_points(workspace=workspace, points_xyz=pts_xyz)
	ax.set_xlim(*xlim)
	ax.set_ylim(*ylim)
	ax.set_zlim(*zlim)

	_draw_workspace_box_3d(ax, workspace)

	_draw_frame_3d(
		ax,
		origin_xyz=(0.0, 0.0, 0.0),
		rotation=np.eye(3, dtype=np.float64),
		axis_len=float(axis_len),
		alpha=0.9,
		label_prefix="base_",
	)

	home_pose = np.asarray(list(home_pose_m_rad), dtype=np.float64).reshape(6)
	ax.scatter([home_pose[0]], [home_pose[1]], [home_pose[2]], marker="x", s=55, color="black", alpha=0.9)
	ax.text(home_pose[0], home_pose[1], home_pose[2], "home", color="black")

	num_points = int(len(points))
	cmap = plt.get_cmap("turbo")
	denom = max(1, num_points - 1)
	show_colorbar = num_points > 20
	show_legend = num_points <= 20

	if num_points <= 50:
		point_size = 70
		point_alpha = 0.95
	elif num_points <= 100:
		point_size = 55
		point_alpha = 0.9
	else:
		point_size = 40
		point_alpha = 0.8

	for i, p in enumerate(points):
		color = cmap(float(i) / float(denom))
		label = p.episode_id if show_legend else None
		ax.scatter(
			[p.xyz_m[0]],
			[p.xyz_m[1]],
			[p.xyz_m[2]],
			marker="X",
			s=float(point_size),
			alpha=float(point_alpha),
			color=color,
			label=label,
		)

	if show_legend and points:
		ax.legend(loc="upper left", fontsize=8)
	elif show_colorbar and points:
		norm = matplotlib.colors.Normalize(vmin=0, vmax=float(denom))
		sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
		sm.set_array([])
		cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
		cbar.set_label("episode index (sorted)")
		ticks, labels = _episode_colorbar_ticks_and_labels([p.episode_id for p in points])
		if ticks:
			cbar.set_ticks(ticks)
			cbar.set_ticklabels(labels)
	fig.tight_layout()
	fig.savefig(output_png, dpi=180)
	plt.close(fig)
	return output_png


def _plot_interaction_points_2d(
	*,
	points: Sequence[InteractionPoint],
	workspace: WorkspaceBounds,
	home_pose_m_rad: Sequence[float],
	gripper_threshold: float,
	output_png: Path,
) -> Path:
	import matplotlib

	matplotlib.use("Agg")
	import matplotlib.pyplot as plt

	output_png = Path(output_png)
	output_png.parent.mkdir(parents=True, exist_ok=True)

	fig, axes = plt.subplots(1, 3, figsize=(16, 5))
	planes = ["XY", "XZ", "YZ"]

	num_points = int(len(points))
	cmap = plt.get_cmap("turbo")
	denom = max(1, num_points - 1)
	show_colorbar = num_points > 20
	show_legend = num_points <= 20

	if num_points <= 50:
		point_size = 60
		point_alpha = 0.95
	elif num_points <= 100:
		point_size = 48
		point_alpha = 0.9
	else:
		point_size = 36
		point_alpha = 0.8

	home_pose = np.asarray(list(home_pose_m_rad), dtype=np.float64).reshape(6)

	for ax, plane in zip(axes, planes):
		ax.set_title(f"{plane} interaction points")
		ax.grid(True, alpha=0.25)
		ax.set_aspect("equal", adjustable="box")
		_draw_workspace_rect_2d(ax, workspace, plane=plane)

		# Base origin marker + axis hints.
		ax.scatter([0.0], [0.0], s=18, color="black", alpha=0.8)
		if plane == "XY":
			ax.arrow(0.0, 0.0, 0.06, 0.0, head_width=0.01, color="r", alpha=0.7, length_includes_head=True)
			ax.arrow(0.0, 0.0, 0.0, 0.06, head_width=0.01, color="g", alpha=0.7, length_includes_head=True)
		elif plane == "XZ":
			ax.arrow(0.0, 0.0, 0.06, 0.0, head_width=0.01, color="r", alpha=0.7, length_includes_head=True)
			ax.arrow(0.0, 0.0, 0.0, 0.06, head_width=0.01, color="b", alpha=0.7, length_includes_head=True)
		elif plane == "YZ":
			ax.arrow(0.0, 0.0, 0.06, 0.0, head_width=0.01, color="g", alpha=0.7, length_includes_head=True)
			ax.arrow(0.0, 0.0, 0.0, 0.06, head_width=0.01, color="b", alpha=0.7, length_includes_head=True)

		# Home marker in this plane.
		if plane == "XY":
			ax.scatter([home_pose[0]], [home_pose[1]], marker="x", s=45, color="black", alpha=0.85)
		elif plane == "XZ":
			ax.scatter([home_pose[0]], [home_pose[2]], marker="x", s=45, color="black", alpha=0.85)
		else:
			ax.scatter([home_pose[1]], [home_pose[2]], marker="x", s=45, color="black", alpha=0.85)

		for i, p in enumerate(points):
			color = cmap(float(i) / float(denom))
			label = p.episode_id if (show_legend and plane == "XY") else None
			if plane == "XY":
				ax.scatter([p.xyz_m[0]], [p.xyz_m[1]], marker="X", s=float(point_size), alpha=float(point_alpha), color=color, label=label)
			elif plane == "XZ":
				ax.scatter([p.xyz_m[0]], [p.xyz_m[2]], marker="X", s=float(point_size), alpha=float(point_alpha), color=color)
			else:
				ax.scatter([p.xyz_m[1]], [p.xyz_m[2]], marker="X", s=float(point_size), alpha=float(point_alpha), color=color)

		if show_legend and plane == "XY" and points:
			ax.legend(loc="upper left", fontsize=8)

	fig.suptitle(f"Interaction points: first gripper close (auto threshold, median={gripper_threshold:.3f})")
	if show_colorbar and points:
		fig.tight_layout(rect=[0.0, 0.0, 0.88, 0.93])
	else:
		fig.tight_layout()

	if show_colorbar and points:
		norm = matplotlib.colors.Normalize(vmin=0, vmax=float(denom))
		sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
		sm.set_array([])
		cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
		cbar.set_label("episode index (sorted)")
		ticks, labels = _episode_colorbar_ticks_and_labels([p.episode_id for p in points])
		if ticks:
			cbar.set_ticks(ticks)
			cbar.set_ticklabels(labels)
	fig.savefig(output_png, dpi=180)
	plt.close(fig)
	return output_png


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


def main() -> None:
	args = _parse_args()
	cfg = RobotPipelineConfig.from_json(args.config)

	raw_dir = Path(args.raw_dir).resolve()
	if not raw_dir.exists():
		raise FileNotFoundError(f"raw_dir does not exist: {raw_dir}")

	if args.output_dir is None:
		output_dir = raw_dir.parent / "viz_delta_actions"
	else:
		output_dir = Path(args.output_dir)
	output_dir = output_dir.resolve()
	output_dir.mkdir(parents=True, exist_ok=True)

	if args.all:
		start_episode = None
		end_episode = None
	else:
		start_episode = parse_episode_index(args.start_episode)
		end_episode = parse_episode_index(args.end_episode)
		if start_episode is not None and end_episode is not None and start_episode > end_episode:
			raise ValueError("start_episode must be <= end_episode")

	episode_dirs = _episode_dirs_in_range(
		raw_dir,
		args.episode_glob,
		start_episode=start_episode,
		end_episode=end_episode,
	)
	if not episode_dirs:
		raise FileNotFoundError(f"No episode folders matched {args.episode_glob!r} in {raw_dir}")

	trajectories: List[EpisodeTrajectory] = []
	skipped: List[Dict[str, str]] = []
	for episode_idx, episode_dir in episode_dirs:
		try:
			traj = _load_episode_trajectory(episode_idx=episode_idx, episode_dir=episode_dir)
			trajectories.append(traj)
		except Exception as e:
			skipped.append({"episode_dir": str(episode_dir), "error": repr(e)})

	if not trajectories:
		raise RuntimeError(f"All episodes failed to load. Skipped={skipped[:3]}")

	out_3d = _plot_trajectories_3d(
		sequences=trajectories,
		workspace=cfg.robot.workspace,
		home_pose_m_rad=cfg.robot.home_pose_m_rad,
		axis_len=float(args.axis_len),
		output_png=output_dir / "raw_delta_actions_3d.png",
	)
	out_2d = _plot_trajectories_2d(
		sequences=trajectories,
		workspace=cfg.robot.workspace,
		home_pose_m_rad=cfg.robot.home_pose_m_rad,
		output_png=output_dir / "raw_delta_actions_2d.png",
	)

	gripper_sticky_steps = int(getattr(cfg.robot, "gripper_sticky_steps", 1))
	interaction_thresholds: List[float] = []

	interaction_points: List[InteractionPoint] = []
	episodes_without_close: List[Dict[str, str]] = []
	per_episode = []
	for traj in trajectories:
		g = traj.integrated_7d[:, 6]
		close_threshold = _auto_gripper_close_threshold(g)
		close_idx = None
		if close_threshold is not None:
			close_idx = _find_first_gripper_close_index(
				g,
				threshold=float(close_threshold),
				sticky_steps=gripper_sticky_steps,
			)
		interaction = None
		if close_idx is not None:
			interaction_thresholds.append(float(close_threshold))
			p = InteractionPoint(
				episode_id=traj.episode_id,
				episode_dir=traj.episode_dir,
				step_index=int(close_idx),
				xyz_m=np.asarray(traj.integrated_7d[int(close_idx), :3], dtype=np.float64),
				gripper_open_before=float(traj.integrated_7d[int(close_idx) - 1, 6]),
				gripper_open_after=float(traj.integrated_7d[int(close_idx), 6]),
			)
			interaction_points.append(p)
			interaction = {
				"first_close_step_index": int(p.step_index),
				"first_close_xyz_m": [float(p.xyz_m[0]), float(p.xyz_m[1]), float(p.xyz_m[2])],
				"gripper_open_before": float(p.gripper_open_before),
				"gripper_open_after": float(p.gripper_open_after),
				"close_threshold": float(close_threshold),
			}
		else:
			episodes_without_close.append(
				{
					"episode_id": traj.episode_id,
					"episode_dir": str(traj.episode_dir),
					"reason": "no_gripper_close_detected",
				}
			)

		per_episode.append(
			{
				"episode_id": traj.episode_id,
				"episode_dir": str(traj.episode_dir),
				"action_path": str(traj.action_path),
				"num_steps": int(traj.num_steps),
				"translation_error_m_mean": float(np.mean(traj.translation_error_m)),
				"translation_error_m_max": float(np.max(traj.translation_error_m)),
				"rotation_error_rad_mean": float(np.mean(traj.rotation_error_rad)),
				"rotation_error_rad_max": float(np.max(traj.rotation_error_rad)),
				"interaction": interaction,
			}
		)

	threshold_median = float(np.median(np.asarray(interaction_thresholds, dtype=np.float64))) if interaction_thresholds else float(cfg.robot.gripper_threshold)

	out_interaction_3d = _plot_interaction_points_3d(
		points=interaction_points,
		workspace=cfg.robot.workspace,
		home_pose_m_rad=cfg.robot.home_pose_m_rad,
		axis_len=float(args.axis_len),
		gripper_threshold=threshold_median,
		output_png=output_dir / "raw_interaction_points_3d.png",
	)
	out_interaction_2d = _plot_interaction_points_2d(
		points=interaction_points,
		workspace=cfg.robot.workspace,
		home_pose_m_rad=cfg.robot.home_pose_m_rad,
		gripper_threshold=threshold_median,
		output_png=output_dir / "raw_interaction_points_2d.png",
	)

	summary = {
		"config_path": str(Path(args.config).resolve()),
		"raw_dir": str(raw_dir),
		"episode_glob": str(args.episode_glob),
		"start_episode": start_episode,
		"end_episode": end_episode,
		"num_episodes_loaded": int(len(trajectories)),
		"num_episodes_skipped": int(len(skipped)),
		"skipped": skipped,
		"output_3d_png": str(out_3d),
		"output_2d_png": str(out_2d),
		"output_interaction_3d_png": str(out_interaction_3d),
		"output_interaction_2d_png": str(out_interaction_2d),
		"interaction": {
			"detection": "auto_midpoint_percentiles",
			"low_percentile": 5.0,
			"high_percentile": 95.0,
			"min_range": 0.02,
			"threshold_median": float(threshold_median),
			"gripper_sticky_steps": int(gripper_sticky_steps),
			"num_interaction_points": int(len(interaction_points)),
			"num_episodes_without_close": int(len(episodes_without_close)),
			"episodes_without_close": episodes_without_close,
		},
		"per_episode": per_episode,
	}
	summary_path = output_dir / "raw_delta_actions_summary.json"
	summary_path.write_text(json.dumps(_to_builtin(summary), indent=2), encoding="utf-8")

	print(f"Loaded episodes: {len(trajectories)} (skipped {len(skipped)})")
	print(f"3D plot: {out_3d}")
	print(f"2D plot: {out_2d}")
	print(f"Interaction points 3D: {out_interaction_3d}")
	print(f"Interaction points 2D: {out_interaction_2d}")
	print(f"Summary JSON: {summary_path}")


if __name__ == "__main__":
	main()

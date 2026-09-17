from __future__ import annotations

import csv
import json
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from octo.robot.cameras import MultiCameraRig
from octo.robot.interfaces import RobotInterface
from octo.robot.stage_a_naming import normalized_waypoint_destination_name


class StageAAbortRequested(RuntimeError):
    pass


@dataclass
class AbsoluteWaypoint:
    pose_m_rad: np.ndarray
    gripper_open: float
    dwell_time_s: float = 0.0

    @classmethod
    def from_7d(cls, values: Sequence[float], dwell_time_s: float = 0.0) -> "AbsoluteWaypoint":
        arr = np.asarray(values, dtype=np.float64)
        if arr.shape != (7,):
            raise ValueError(f"Expected waypoint shape (7,), got {arr.shape}")
        return cls(pose_m_rad=arr[:6], gripper_open=float(arr[6]), dwell_time_s=float(dwell_time_s))


def load_absolute_waypoints(path: str | Path) -> Tuple[List[AbsoluteWaypoint], float, Dict[str, object]]:
    """Load absolute 7D waypoints from JSON.

    Supported formats:
      1) [[x,y,z,rx,ry,rz,grip], ...]
      2) {"segment_time_s": 1.5, "instruction": "...", "waypoints": [[...], ...]}
      3) {"segment_time_s": 1.5, "instruction": "...", "waypoints": [{"pose": [...], "gripper": 1.0, "dwell_time_s": 0.2}, ...]}
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    segment_time_s = 1.0
    meta: Dict[str, object] = {"waypoint_source": str(path)}
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = data.get("waypoints", [])
        segment_time_s = float(data.get("segment_time_s", 1.0))
        for k in ("instruction", "name", "tags"):
            if k in data:
                meta[k] = data[k]
    else:
        raise ValueError("Unsupported waypoint JSON format")

    waypoints: List[AbsoluteWaypoint] = []
    for item in items:
        if isinstance(item, list):
            waypoints.append(AbsoluteWaypoint.from_7d(item))
        elif isinstance(item, dict):
            if "pose" in item:
                pose = np.asarray(item["pose"], dtype=np.float64)
                grip = float(item.get("gripper", item.get("gripper_open", 1.0)))
                if pose.shape != (6,):
                    raise ValueError(f"Expected 'pose' length 6, got {pose.shape}")
                waypoints.append(
                    AbsoluteWaypoint(
                        pose_m_rad=pose,
                        gripper_open=grip,
                        dwell_time_s=float(item.get("dwell_time_s", 0.0)),
                    )
                )
            else:
                arr = np.asarray(item.get("waypoint", item.get("value", [])), dtype=np.float64)
                waypoints.append(AbsoluteWaypoint.from_7d(arr, dwell_time_s=float(item.get("dwell_time_s", 0.0))))
        else:
            raise ValueError(f"Unsupported waypoint entry type: {type(item)!r}")

    if len(waypoints) < 2:
        raise ValueError("Need at least 2 waypoints for Stage-A collection")
    return waypoints, segment_time_s, meta


def interpolate_waypoints(
    waypoints: Sequence[AbsoluteWaypoint],
    *,
    control_hz: float,
    segment_time_s: float,
) -> Tuple[np.ndarray, List[dict]]:
    if control_hz <= 0:
        raise ValueError("control_hz must be positive")
    if segment_time_s <= 0:
        raise ValueError("segment_time_s must be positive")

    dense_rows: List[np.ndarray] = []
    dense_meta: List[dict] = []

    for seg_idx, (wp0, wp1) in enumerate(zip(waypoints[:-1], waypoints[1:])):
        n_steps = max(2, int(round(segment_time_s * control_hz)))
        for i in range(n_steps):
            alpha = (i + 1) / n_steps
            pose = (1.0 - alpha) * wp0.pose_m_rad + alpha * wp1.pose_m_rad
            grip = float((1.0 - alpha) * wp0.gripper_open + alpha * wp1.gripper_open)
            row = np.concatenate([pose, [grip]], axis=0)
            dense_rows.append(row.astype(np.float32))
            dense_meta.append(
                {
                    "segment_index": seg_idx,
                    "segment_step_index": i,
                    "segment_num_steps": n_steps,
                    "alpha": float(alpha),
                    "is_dwell": False,
                }
            )
        if wp1.dwell_time_s > 0:
            dwell_steps = max(1, int(round(wp1.dwell_time_s * control_hz)))
            row = np.concatenate([wp1.pose_m_rad, [wp1.gripper_open]], axis=0).astype(np.float32)
            for dwell_idx in range(dwell_steps):
                dense_rows.append(row.copy())
                dense_meta.append(
                    {
                        "segment_index": seg_idx,
                        "segment_step_index": dwell_idx,
                        "segment_num_steps": dwell_steps,
                        "alpha": 1.0,
                        "is_dwell": True,
                    }
                )

    return np.stack(dense_rows, axis=0), dense_meta


class StageAEpisodeRecorder:
    def __init__(
        self,
        episode_dir: str | Path,
        *,
        instruction: str,
        control_hz: float,
        robot_name: str,
        enabled_cameras: Sequence[str],
        segment_time_s: float,
        waypoint_source: Optional[str] = None,
        run_kind: str = "record",
    ):
        self.episode_dir = Path(episode_dir)
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        self.instruction = instruction
        self.control_hz = float(control_hz)
        self.robot_name = robot_name
        self.enabled_cameras = list(enabled_cameras)
        self.segment_time_s = float(segment_time_s)
        self.waypoint_source = waypoint_source
        self.run_kind = run_kind
        self.frames_root = self.episode_dir / "frames"
        self.frames_root.mkdir(exist_ok=True)
        self._debug_rows: List[dict] = []

    def save_metadata(self, *, waypoints_7d: np.ndarray, dense_targets_7d: np.ndarray, extra_meta: Optional[Dict[str, object]] = None) -> None:
        meta = {
            "format_version": "0.0.5",
            "instruction": self.instruction,
            "control_hz": self.control_hz,
            "robot_name": self.robot_name,
            "segment_time_s": self.segment_time_s,
            "enabled_cameras": self.enabled_cameras,
            "waypoint_source": self.waypoint_source,
            "run_kind": self.run_kind,
            "num_waypoints": int(len(waypoints_7d)),
            "num_dense_targets": int(len(dense_targets_7d)),
            "waypoints_7d": waypoints_7d.tolist(),
        }
        if extra_meta:
            meta.update(extra_meta)
        (self.episode_dir / "instruction.txt").write_text(self.instruction, encoding="utf-8")
        (self.episode_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        np.save(self.episode_dir / "commanded_dense_targets.npy", dense_targets_7d.astype(np.float32))
        np.save(self.episode_dir / "commanded_waypoints.npy", waypoints_7d.astype(np.float32))

    def copy_waypoint_file(self, src: str | Path) -> None:
        src = Path(src)
        if src.exists():
            waypoint_dir = self.episode_dir / "waypoints"
            waypoint_dir.mkdir(parents=True, exist_ok=True)

            try:
                dst_name = normalized_waypoint_destination_name(self.episode_dir.name)
            except ValueError:
                dst_name = None
                for parent in self.episode_dir.parents:
                    try:
                        dst_name = normalized_waypoint_destination_name(parent.name)
                        break
                    except ValueError:
                        continue
                if dst_name is None:
                    raise

            shutil.copy2(src, waypoint_dir / dst_name)

    def record_step(
        self,
        *,
        step_idx: int,
        timestamp_s: float,
        achieved_state_7d: np.ndarray,
        commanded_state_7d: np.ndarray,
        frames: Mapping[str, np.ndarray],
        dense_meta: Optional[Mapping[str, float]] = None,
    ) -> None:
        for cam_name, image_rgb in frames.items():
            import imageio.v2 as imageio

            cam_dir = self.frames_root / cam_name
            cam_dir.mkdir(parents=True, exist_ok=True)
            imageio.imwrite(cam_dir / f"{step_idx:06d}.jpg", image_rgb)

        row = {
            "step_index": int(step_idx),
            "timestamp_s": float(timestamp_s),
            **{f"achieved_{k}": float(v) for k, v in zip(["x_m", "y_m", "z_m", "rx_rad", "ry_rad", "rz_rad", "gripper_open"], achieved_state_7d)},
            **{f"commanded_{k}": float(v) for k, v in zip(["x_m", "y_m", "z_m", "rx_rad", "ry_rad", "rz_rad", "gripper_open"], commanded_state_7d)},
        }
        if dense_meta:
            row.update({k: float(v) if isinstance(v, (float, np.floating)) else int(v) if isinstance(v, (int, np.integer, bool)) else v for k, v in dense_meta.items()})
        self._debug_rows.append(row)

    def finalize(
        self,
        *,
        achieved_states_7d: np.ndarray,
        commanded_states_7d: np.ndarray,
        timestamps_s: np.ndarray,
    ) -> None:
        if achieved_states_7d.shape[0] != commanded_states_7d.shape[0] or achieved_states_7d.shape[0] != timestamps_s.shape[0]:
            raise ValueError("Mismatched sequence lengths while finalizing episode")

        np.save(self.episode_dir / "achieved_state.npy", achieved_states_7d.astype(np.float32))
        np.save(self.episode_dir / "commanded_state.npy", commanded_states_7d.astype(np.float32))
        np.save(self.episode_dir / "timestamps.npy", timestamps_s.astype(np.float32))

        if achieved_states_7d.shape[0] < 2:
            raise RuntimeError("Episode is too short to derive action labels")

        obs_state = achieved_states_7d[:-1].astype(np.float32)
        action = np.zeros((obs_state.shape[0], 7), dtype=np.float32)
        action[:, :6] = achieved_states_7d[1:, :6] - achieved_states_7d[:-1, :6]
        action[:, 6] = achieved_states_7d[1:, 6]

        commanded_obs_state = commanded_states_7d[:-1].astype(np.float32)
        commanded_action = np.zeros((commanded_obs_state.shape[0], 7), dtype=np.float32)
        commanded_action[:, :6] = commanded_states_7d[1:, :6] - commanded_states_7d[:-1, :6]
        commanded_action[:, 6] = commanded_states_7d[1:, 6]

        np.save(self.episode_dir / "obs_state.npy", obs_state)
        np.save(self.episode_dir / "action.npy", action)
        np.save(self.episode_dir / "delta_action_from_get_coords.npy", action)
        np.save(self.episode_dir / "commanded_obs_state.npy", commanded_obs_state)
        np.save(self.episode_dir / "commanded_action.npy", commanded_action)

        if self._debug_rows:
            with open(self.episode_dir / "step_debug.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(self._debug_rows[0].keys()))
                writer.writeheader()
                writer.writerows(self._debug_rows)


def _snapshot_selected_cameras(cameras: MultiCameraRig, include_modalities: Optional[Sequence[str]] = None) -> Dict[str, np.ndarray]:
    frames = cameras.snapshot()
    if not include_modalities:
        return frames
    wanted = {f"image_{name}" for name in include_modalities}
    return {k: v for k, v in frames.items() if k in wanted}


def execute_stage_a_episode(
    *,
    robot: RobotInterface,
    cameras: MultiCameraRig,
    absolute_waypoints: Sequence[AbsoluteWaypoint],
    control_hz: float,
    segment_time_s: float,
    logger: Optional[logging.Logger] = None,
    initial_settle_s: float = 0.5,
    reset_to_home: bool = True,
    step_callback: Optional[Callable[[Dict[str, object]], bool]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict], Dict[str, List[np.ndarray]]]:
    logger = logger or logging.getLogger(__name__)
    dense_targets_7d, dense_meta = interpolate_waypoints(
        absolute_waypoints,
        control_hz=control_hz,
        segment_time_s=segment_time_s,
    )

    if reset_to_home:
        logger.info("Resetting robot to home pose before scripted episode...")
        robot.reset_to_home()
    else:
        logger.info("Skipping reset_to_home() before scripted episode")
    first_target = dense_targets_7d[0]
    robot.move_tcp_absolute(first_target[:6], float(first_target[6]))
    time.sleep(max(0.0, float(initial_settle_s)))

    dt = 1.0 / float(control_hz)
    start_t = time.time()
    achieved_states: List[np.ndarray] = []
    commanded_states: List[np.ndarray] = []
    timestamps: List[float] = []
    captured_frames: Dict[str, List[np.ndarray]] = {}

    for step_idx, target_7d in enumerate(dense_targets_7d):
        step_t0 = time.time()
        target_pose = target_7d[:6]
        target_gripper = float(target_7d[6])
        logger.debug("Stage-A step %04d | commanded_absolute_7d=%s", step_idx, np.array2string(target_7d, precision=4))
        robot.move_tcp_absolute(target_pose, target_gripper)
        achieved_pose = robot.get_tcp_pose_m_rad().astype(np.float32)
        achieved_gripper = float(robot.get_gripper_open_fraction())
        achieved_7d = np.concatenate([achieved_pose, [achieved_gripper]], axis=0).astype(np.float32)

        frames, frame_meta = cameras.snapshot_with_metadata()
        for k, v in frames.items():
            captured_frames.setdefault(k, []).append(v)
        timestamp_s = time.time() - start_t

        prev_achieved = achieved_states[-1] if achieved_states else None
        if step_callback is not None:
            should_continue = bool(
                step_callback(
                    {
                        "step_idx": step_idx,
                        "timestamp_s": float(timestamp_s),
                        "target_7d": target_7d.astype(np.float32).copy(),
                        "achieved_7d": achieved_7d.copy(),
                        "prev_achieved_7d": None if prev_achieved is None else prev_achieved.copy(),
                        "frames": {k: v for k, v in frames.items()},
                        "frame_meta": {k: dict(v) for k, v in frame_meta.items()},
                        "dense_meta": dict(dense_meta[step_idx]),
                    }
                )
            )
            if not should_continue:
                raise StageAAbortRequested(f"Operator aborted Stage-A collection at step {step_idx}")

        achieved_states.append(achieved_7d)
        commanded_states.append(target_7d.astype(np.float32))
        timestamps.append(timestamp_s)
        if prev_achieved is not None:
            delta_from_get_coords = achieved_7d - prev_achieved
            logger.debug(
                "Stage-A step %04d | achieved_absolute_7d=%s | delta_from_get_coords=%s",
                step_idx,
                np.array2string(achieved_7d, precision=4),
                np.array2string(delta_from_get_coords, precision=4),
            )
        else:
            logger.debug("Stage-A step %04d | achieved_absolute_7d=%s | delta_from_get_coords=<bootstrap>", step_idx, np.array2string(achieved_7d, precision=4))

        elapsed = time.time() - step_t0
        sleep_s = max(0.0, dt - elapsed)
        if sleep_s > 0:
            time.sleep(sleep_s)

    achieved_states_arr = np.stack(achieved_states, axis=0).astype(np.float32)
    commanded_states_arr = np.stack(commanded_states, axis=0).astype(np.float32)
    timestamps_arr = np.asarray(timestamps, dtype=np.float32)
    return achieved_states_arr, commanded_states_arr, timestamps_arr, dense_meta, captured_frames


def collect_stage_a_episode(
    *,
    robot: RobotInterface,
    cameras: MultiCameraRig,
    instruction: str,
    raw_episode_dir: str | Path,
    absolute_waypoints: Sequence[AbsoluteWaypoint],
    control_hz: float,
    segment_time_s: float,
    logger: Optional[logging.Logger] = None,
    initial_settle_s: float = 0.5,
    reset_to_home: bool = True,
    waypoint_source: Optional[str] = None,
    include_modalities: Optional[Sequence[str]] = None,
    run_kind: str = "record",
    extra_meta: Optional[Dict[str, object]] = None,
    copy_waypoint_json: bool = True,
    step_callback: Optional[Callable[[Dict[str, object]], bool]] = None,
) -> Path:
    logger = logger or logging.getLogger(__name__)
    raw_episode_dir = Path(raw_episode_dir)
    waypoints_7d = np.stack(
        [np.concatenate([wp.pose_m_rad, [wp.gripper_open]], axis=0) for wp in absolute_waypoints],
        axis=0,
    ).astype(np.float32)
    dense_targets_7d, _ = interpolate_waypoints(absolute_waypoints, control_hz=control_hz, segment_time_s=segment_time_s)

    recorder = StageAEpisodeRecorder(
        raw_episode_dir,
        instruction=instruction,
        control_hz=control_hz,
        robot_name=robot.__class__.__name__,
        enabled_cameras=list(include_modalities or cameras.available_modalities()),
        segment_time_s=segment_time_s,
        waypoint_source=waypoint_source,
        run_kind=run_kind,
    )
    recorder.save_metadata(waypoints_7d=waypoints_7d, dense_targets_7d=dense_targets_7d, extra_meta=extra_meta)
    if copy_waypoint_json and waypoint_source:
        recorder.copy_waypoint_file(waypoint_source)

    achieved_states_arr, commanded_states_arr, timestamps_arr, dense_meta, captured_frames = execute_stage_a_episode(
        robot=robot,
        cameras=cameras,
        absolute_waypoints=absolute_waypoints,
        control_hz=control_hz,
        segment_time_s=segment_time_s,
        logger=logger,
        initial_settle_s=initial_settle_s,
        reset_to_home=reset_to_home,
        step_callback=step_callback,
    )

    usable_frames: Dict[str, np.ndarray] = {}
    for k, seq in captured_frames.items():
        if include_modalities and k not in {f"image_{m}" for m in include_modalities}:
            continue
        usable_frames[k] = seq

    for step_idx, (achieved_7d, commanded_7d, timestamp_s) in enumerate(zip(achieved_states_arr, commanded_states_arr, timestamps_arr)):
        frames = {k: seq[step_idx] for k, seq in usable_frames.items() if step_idx < len(seq)}
        recorder.record_step(
            step_idx=step_idx,
            timestamp_s=float(timestamp_s),
            achieved_state_7d=achieved_7d,
            commanded_state_7d=commanded_7d,
            frames=frames,
            dense_meta=dense_meta[step_idx],
        )

    recorder.finalize(
        achieved_states_7d=achieved_states_arr,
        commanded_states_7d=commanded_states_arr,
        timestamps_s=timestamps_arr,
    )
    logger.info(
        "Stage-A collection complete | episode_dir=%s | num_frames=%d | duration=%.2fs",
        raw_episode_dir,
        len(achieved_states_arr),
        float(timestamps_arr[-1]) if len(timestamps_arr) else 0.0,
    )
    return raw_episode_dir

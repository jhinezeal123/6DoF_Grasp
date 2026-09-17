from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from octo.robot.config import RobotConfig
from octo.robot.interfaces import RobotInterface
from octo.robot.transform_integrator import BridgeStyleActionIntegrator, TransformStepResult
from octo.robot.utils import (
    absolute_pose_to_relative_neutral_pose,
    clamp_pose_to_workspace,
    get_neutral_orientation_rpy,
    pose_error_metrics,
)
from octo.robot.visualization import Trajectory3DVisualizer


class MockMyArmM750Robot(RobotInterface):
    def __init__(self, config: RobotConfig, visualize: bool = False):
        self.config = config
        self._connected = False
        self._pose_m_rad = np.asarray(config.home_pose_m_rad, dtype=np.float64).copy()
        self._pose_m_rad = clamp_pose_to_workspace(self._pose_m_rad, self.config.workspace)
        self._neutral_orientation_rpy = get_neutral_orientation_rpy(
            config.neutral_orientation_rpy_rad,
            config.home_pose_m_rad,
        )
        self._gripper_open = 1.0 if config.open_gripper_on_reset else 0.0
        self._integrator = BridgeStyleActionIntegrator(config)
        self._visualizer = None
        self._step_counter = 0
        self._last_debug: Dict[str, object] = {}
        if visualize:
            self._visualizer = Trajectory3DVisualizer(config.workspace, interactive=True)

    def connect(self) -> None:
        self._connected = True

    def configure_for_policy(self) -> None:
        if not self._connected:
            raise RuntimeError("Mock robot not connected")

    def disconnect(self) -> None:
        self._connected = False

    def reset_to_home(self) -> None:
        self._pose_m_rad = np.asarray(self.config.home_pose_m_rad, dtype=np.float64).copy()
        self._pose_m_rad = clamp_pose_to_workspace(self._pose_m_rad, self.config.workspace)
        self._integrator.reset(self._pose_m_rad)
        self._gripper_open = 1.0 if self.config.open_gripper_on_reset else 0.0
        self._step_counter = 0
        if self._visualizer is not None:
            self._visualizer.update(self._pose_m_rad, home_pose_m_rad=np.asarray(self.config.home_pose_m_rad))

    def get_tcp_pose_m_rad(self) -> np.ndarray:
        return self._pose_m_rad.copy()

    def _get_proprio_pose(self, pose_m_rad: np.ndarray) -> np.ndarray:
        pose = np.asarray(pose_m_rad, dtype=np.float64).reshape(6)
        if self.config.proprio_orientation_mode == "absolute":
            return pose
        return absolute_pose_to_relative_neutral_pose(pose, self._neutral_orientation_rpy)

    def get_proprio(self) -> np.ndarray:
        proprio_pose = self._get_proprio_pose(self._pose_m_rad)
        return np.concatenate([proprio_pose, [self.get_gripper_open_fraction()]], axis=0)

    def get_gripper_open_fraction(self) -> float:
        return float(self._gripper_open)

    def apply_action(self, delta_pose_m_rad: np.ndarray, gripper_open_fraction: float) -> np.ndarray:
        measured_before = self._pose_m_rad.copy()
        step_result: TransformStepResult = self._integrator.step(
            np.asarray(delta_pose_m_rad, dtype=np.float64),
            measured_eef_pose_m_rad=measured_before,
        )
        self._pose_m_rad = step_result.target_pose_m_rad.copy()
        self._gripper_open = float(np.clip(gripper_open_fraction, 0.0, 1.0))
        self._step_counter += 1
        err = pose_error_metrics(self._integrator.get_target_pose(), self._pose_m_rad)
        self._last_debug = {
            "step_counter": int(self._step_counter),
            "integration_mode": step_result.integration_mode,
            "pivot_source": step_result.pivot_source,
            "compose_base": step_result.compose_base_source,
            "delta_pose_m_rad": np.asarray(delta_pose_m_rad, dtype=np.float64).copy(),
            "clipped_delta_pose_m_rad": step_result.clipped_delta_pose_m_rad.copy(),
            "target_pose_before_step_m_rad": step_result.target_pose_before_step_m_rad.copy(),
            "target_pose_before_step_rel_m_rad": self._get_proprio_pose(step_result.target_pose_before_step_m_rad),
            "target_transform_before_step": step_result.target_transform_before_step.copy(),
            "target_pose_m_rad": self._integrator.get_target_pose().copy(),
            "target_pose_after_step_m_rad": self._integrator.get_target_pose().copy(),
            "target_pose_rel_m_rad": self._get_proprio_pose(self._integrator.get_target_pose()),
            "target_pose_after_step_rel_m_rad": self._get_proprio_pose(self._integrator.get_target_pose()),
            "target_transform": self._integrator.get_target_transform().copy(),
            "target_transform_after_step": self._integrator.get_target_transform().copy(),
            "compose_base_transform": step_result.compose_base_transform.copy(),
            "measured_pose_before_step_m_rad": measured_before.copy(),
            "measured_pose_before_step_rel_m_rad": self._get_proprio_pose(measured_before),
            "measured_pose_m_rad": self._pose_m_rad.copy(),
            "measured_pose_after_step_m_rad": self._pose_m_rad.copy(),
            "measured_pose_rel_m_rad": self._get_proprio_pose(self._pose_m_rad),
            "measured_pose_after_step_rel_m_rad": self._get_proprio_pose(self._pose_m_rad),
            "hardware_pose_m_rad": self._pose_m_rad.copy(),
            "neutral_orientation_rpy_rad": self._neutral_orientation_rpy.copy(),
            "pivot_position_m": step_result.pivot_position_m.copy(),
            "translation_error_m": float(err["translation_m"]),
            "rotation_error_rad": float(err["rotation_rad"]),
            "translation_error_norm_m": float(err["translation_m"]),
            "rotation_error_norm_rad": float(err["rotation_rad"]),
            "resync_applied": False,
            "resync_reason": "mock_disabled",
            "gripper_open": float(self._gripper_open),
        }
        if self._visualizer is not None:
            self._visualizer.update(self._pose_m_rad, home_pose_m_rad=np.asarray(self.config.home_pose_m_rad))
        return self._pose_m_rad.copy()

    def move_tcp_absolute(self, pose_m_rad: np.ndarray, gripper_open_fraction: float) -> np.ndarray:
        target_pose = clamp_pose_to_workspace(
            np.asarray(pose_m_rad, dtype=np.float64).reshape(6),
            self.config.workspace,
        )
        self._pose_m_rad = target_pose.copy()
        self._integrator.reset(self._pose_m_rad)
        self._gripper_open = float(np.clip(gripper_open_fraction, 0.0, 1.0))
        self._step_counter += 1
        if self._visualizer is not None:
            self._visualizer.update(self._pose_m_rad, home_pose_m_rad=np.asarray(self.config.home_pose_m_rad))
        return self._pose_m_rad.copy()

    def get_debug_snapshot(self) -> Dict[str, object]:
        if self._last_debug:
            return dict(self._last_debug)
        return {
            "step_counter": int(self._step_counter),
            "integration_mode": getattr(self.config, "se3_integration_mode", "bridge_exact"),
            "pivot_source": getattr(self.config, "pivot_source", "hardware_measured"),
            "compose_base": getattr(self.config, "compose_base", "previous_target"),
            "target_pose_before_step_m_rad": self._integrator.get_target_pose().copy(),
            "target_pose_before_step_rel_m_rad": self._get_proprio_pose(self._integrator.get_target_pose()),
            "target_transform_before_step": self._integrator.get_target_transform().copy(),
            "target_pose_m_rad": self._integrator.get_target_pose().copy(),
            "target_pose_after_step_m_rad": self._integrator.get_target_pose().copy(),
            "target_pose_rel_m_rad": self._get_proprio_pose(self._integrator.get_target_pose()),
            "target_pose_after_step_rel_m_rad": self._get_proprio_pose(self._integrator.get_target_pose()),
            "target_transform": self._integrator.get_target_transform().copy(),
            "target_transform_after_step": self._integrator.get_target_transform().copy(),
            "compose_base_transform": self._integrator.get_target_transform().copy(),
            "measured_pose_before_step_m_rad": self._pose_m_rad.copy(),
            "measured_pose_before_step_rel_m_rad": self._get_proprio_pose(self._pose_m_rad),
            "measured_pose_m_rad": self._pose_m_rad.copy(),
            "measured_pose_after_step_m_rad": self._pose_m_rad.copy(),
            "measured_pose_rel_m_rad": self._get_proprio_pose(self._pose_m_rad),
            "measured_pose_after_step_rel_m_rad": self._get_proprio_pose(self._pose_m_rad),
            "hardware_pose_m_rad": self._pose_m_rad.copy(),
            "neutral_orientation_rpy_rad": self._neutral_orientation_rpy.copy(),
            "pivot_position_m": self._pose_m_rad[:3].copy(),
            "translation_error_m": 0.0,
            "rotation_error_rad": 0.0,
            "translation_error_norm_m": 0.0,
            "rotation_error_norm_rad": 0.0,
            "resync_applied": False,
            "resync_reason": "snapshot_only",
            "gripper_open": float(self._gripper_open),
        }

    def save_plot(self, path: Optional[str]) -> None:
        if path and self._visualizer is not None:
            self._visualizer.save(path)

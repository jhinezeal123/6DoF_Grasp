from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from octo.robot.config import RobotConfig
from octo.robot.utils import (
    action_to_bridge_delta_transform,
    clamp_pose_to_workspace,
    clamp_transform_translation_to_workspace,
    clip_delta_action,
    pose_to_transform,
    select_rpy_near_reference,
    transform_to_pose,
)


@dataclass
class TransformStepResult:
    target_pose_m_rad: np.ndarray
    target_transform: np.ndarray
    target_pose_before_step_m_rad: np.ndarray
    target_transform_before_step: np.ndarray
    compose_base_transform: np.ndarray
    compose_base_source: str
    clipped_delta_pose_m_rad: np.ndarray
    pivot_position_m: np.ndarray
    pivot_source: str
    measured_pose_m_rad: Optional[np.ndarray]
    integration_mode: str


class BridgeStyleActionIntegrator:
    """SE(3) integrator with Bridge-exact and comparison modes.

    v0_2_3 semantics:
    - `bridge_exact`:     delta(measured-pivot) @ previous_target
    - `measured_compose`: delta(measured-pivot) @ measured_transform
    - `target_only`:      delta(target-pivot)   @ previous_target

    Translation clamping happens in transform space. Euler continuity handling happens
    only after the transform composition when converting back to xyz+rpy for the vendor API.
    """

    def __init__(self, config: RobotConfig):
        self.config = config
        self._target_pose_m_rad = clamp_pose_to_workspace(
            np.asarray(config.home_pose_m_rad, dtype=np.float64),
            config.workspace,
        )
        self._target_transform = pose_to_transform(self._target_pose_m_rad)

    def reset(self, pose_m_rad: np.ndarray) -> np.ndarray:
        pose = clamp_pose_to_workspace(np.asarray(pose_m_rad, dtype=np.float64), self.config.workspace)
        self._target_pose_m_rad = pose
        self._target_transform = pose_to_transform(pose)
        return self._target_pose_m_rad.copy()

    def get_target_pose(self) -> np.ndarray:
        return self._target_pose_m_rad.copy()

    def get_target_transform(self) -> np.ndarray:
        return self._target_transform.copy()

    def _effective_mode(self) -> str:
        return str(getattr(self.config, "se3_integration_mode", "bridge_exact"))

    def _effective_pivot_source(self) -> str:
        mode = self._effective_mode()
        if mode in {"bridge_exact", "measured_compose"}:
            return "hardware_measured"
        return str(getattr(self.config, "pivot_source", "target_transform"))

    def _effective_compose_base(self) -> str:
        mode = self._effective_mode()
        if mode == "measured_compose":
            return "hardware_measured"
        return str(getattr(self.config, "compose_base", "previous_target"))

    def step(
        self,
        delta_pose_m_rad: np.ndarray,
        *,
        measured_eef_pose_m_rad: Optional[np.ndarray] = None,
    ) -> TransformStepResult:
        delta = clip_delta_action(
            np.asarray(delta_pose_m_rad, dtype=np.float64),
            self.config.max_step_translation_m,
            self.config.max_step_rotation_rad,
        )

        target_before_pose = self._target_pose_m_rad.copy()
        target_before_transform = self._target_transform.copy()

        measured_pose = None
        measured_transform = None
        if measured_eef_pose_m_rad is not None:
            measured_pose = clamp_pose_to_workspace(
                np.asarray(measured_eef_pose_m_rad, dtype=np.float64).reshape(6),
                self.config.workspace,
            )
            measured_transform = pose_to_transform(measured_pose)

        pivot_source = self._effective_pivot_source()
        compose_base_source = self._effective_compose_base()

        if pivot_source == "hardware_measured":
            if measured_pose is not None and bool(getattr(self.config, "hardware_measured_enabled", True)):
                pivot_xyz = measured_pose[:3].copy()
                effective_pivot_source = "hardware_measured"
            else:
                pivot_xyz = target_before_transform[:3, 3].copy()
                effective_pivot_source = "target_transform_fallback"
        elif pivot_source == "measured_or_target":
            if measured_pose is not None and bool(getattr(self.config, "hardware_measured_enabled", True)):
                pivot_xyz = measured_pose[:3].copy()
                effective_pivot_source = "hardware_measured"
            else:
                pivot_xyz = target_before_transform[:3, 3].copy()
                effective_pivot_source = "target_transform"
        else:
            pivot_xyz = target_before_transform[:3, 3].copy()
            effective_pivot_source = "target_transform"

        if compose_base_source == "hardware_measured":
            if measured_transform is not None and bool(getattr(self.config, "hardware_measured_enabled", True)):
                compose_base_transform = measured_transform.copy()
                effective_compose_base = "hardware_measured"
            else:
                compose_base_transform = target_before_transform.copy()
                effective_compose_base = "previous_target_fallback"
        else:
            compose_base_transform = target_before_transform.copy()
            effective_compose_base = "previous_target"

        delta_transform = action_to_bridge_delta_transform(delta, pivot_xyz)
        next_transform = delta_transform @ compose_base_transform
        next_transform = clamp_transform_translation_to_workspace(next_transform, self.config.workspace)

        next_pose = transform_to_pose(next_transform)
        next_pose[3:6] = select_rpy_near_reference(next_pose[3:6], target_before_pose[3:6])
        if bool(getattr(self.config, "clamp_rotation_in_pose", False)):
            next_pose = clamp_pose_to_workspace(next_pose, self.config.workspace)
            next_transform = pose_to_transform(next_pose)

        self._target_pose_m_rad = next_pose
        self._target_transform = next_transform
        return TransformStepResult(
            target_pose_m_rad=next_pose.copy(),
            target_transform=next_transform.copy(),
            target_pose_before_step_m_rad=target_before_pose.copy(),
            target_transform_before_step=target_before_transform.copy(),
            compose_base_transform=compose_base_transform.copy(),
            compose_base_source=effective_compose_base,
            clipped_delta_pose_m_rad=delta.copy(),
            pivot_position_m=pivot_xyz.copy(),
            pivot_source=effective_pivot_source,
            measured_pose_m_rad=None if measured_pose is None else measured_pose.copy(),
            integration_mode=self._effective_mode(),
        )

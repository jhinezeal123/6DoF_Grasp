from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

import numpy as np


class RobotInterface(ABC):
    @abstractmethod
    def connect(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def configure_for_policy(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def disconnect(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def reset_to_home(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_tcp_pose_m_rad(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_gripper_open_fraction(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def apply_action(self, delta_pose_m_rad: np.ndarray, gripper_open_fraction: float) -> np.ndarray:
        """Apply a delta-TCP action in SI units and return the resulting absolute pose in SI units."""
        raise NotImplementedError

    def move_tcp_absolute(self, pose_m_rad: np.ndarray, gripper_open_fraction: float) -> np.ndarray:
        """Command an absolute TCP pose in SI units.

        Not all backends implement absolute setpoints natively. The default
        implementation converts an absolute target into a delta relative to the
        current TCP pose and routes through :meth:`apply_action`.
        """
        target_pose = np.asarray(pose_m_rad, dtype=np.float64).reshape(6)
        current_pose = np.asarray(self.get_tcp_pose_m_rad(), dtype=np.float64).reshape(6)
        delta = target_pose - current_pose
        return self.apply_action(delta, gripper_open_fraction)

    def get_proprio(self) -> np.ndarray:
        pose = self.get_tcp_pose_m_rad()
        return np.concatenate([pose, [self.get_gripper_open_fraction()]], axis=0)

    def get_debug_state(self) -> Dict[str, float]:
        pose = self.get_tcp_pose_m_rad()
        return {
            "x_m": float(pose[0]),
            "y_m": float(pose[1]),
            "z_m": float(pose[2]),
            "rx_rad": float(pose[3]),
            "ry_rad": float(pose[4]),
            "rz_rad": float(pose[5]),
            "gripper_open": float(self.get_gripper_open_fraction()),
        }

    def get_debug_snapshot(self) -> Dict[str, object]:
        pose = self.get_tcp_pose_m_rad()
        return {
            "measured_pose_m_rad": pose.copy(),
            "gripper_open": float(self.get_gripper_open_fraction()),
        }

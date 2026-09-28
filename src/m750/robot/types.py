"""Domain types for robot control.

This module intentionally depends only on the Python standard library so the
public API can be imported on machines without ROS, MuJoCo or pymycobot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Optional, Tuple


JointTuple = Tuple[float, ...]
JointLimits = Tuple[Tuple[float, float], ...]


def _finite_tuple(values, size: int, name: str) -> tuple:
    result = tuple(float(value) for value in values)
    if len(result) != size:
        raise ValueError("%s must contain exactly %d values" % (name, size))
    if not all(math.isfinite(value) for value in result):
        raise ValueError("%s must contain only finite values" % name)
    return result


@dataclass(frozen=True)
class TcpPose:
    """Tool pose in metres and ROS quaternion order [x, y, z, w]."""

    position_m: Tuple[float, float, float]
    quaternion_xyzw: Tuple[float, float, float, float]

    def __post_init__(self) -> None:
        position = _finite_tuple(self.position_m, 3, "position_m")
        quaternion = _finite_tuple(self.quaternion_xyzw, 4, "quaternion_xyzw")
        norm = math.sqrt(sum(value * value for value in quaternion))
        if norm < 1e-12:
            raise ValueError("quaternion_xyzw must be non-zero")
        normalized = tuple(value / norm for value in quaternion)
        object.__setattr__(self, "position_m", position)
        object.__setattr__(self, "quaternion_xyzw", normalized)


@dataclass(frozen=True)
class RobotState:
    """Backend-neutral robot snapshot.

    Missing hardware feedback is represented by None instead of synthetic
    zero or NaN vectors. Clients can therefore distinguish unknown state from
    a real zero pose without depending on a concrete driver.
    """

    joints_rad: Optional[JointTuple]
    gripper_opening_m: Optional[float]
    tcp_pose: Optional[TcpPose]
    connected: bool
    ready: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.joints_rad is not None:
            object.__setattr__(
                self,
                "joints_rad",
                _finite_tuple(self.joints_rad, 6, "joints_rad"),
            )
        if self.gripper_opening_m is not None:
            value = float(self.gripper_opening_m)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("gripper_opening_m must be finite and non-negative")
            object.__setattr__(self, "gripper_opening_m", value)


__all__ = ["JointLimits", "JointTuple", "RobotState", "TcpPose"]

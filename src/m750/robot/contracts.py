"""Small robot interfaces.

Application code depends on these abstractions. Concrete ROS, serial and
MuJoCo implementations live in adapters and are selected only by the
composition root.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from .types import JointLimits, RobotState, TcpPose


class RobotStateReader(ABC):
    @abstractmethod
    def read_state(self) -> RobotState:
        """Return the latest backend-neutral state snapshot."""


class JointPositionController(ABC):
    @property
    @abstractmethod
    def joint_limits_rad(self) -> JointLimits:
        """Six inclusive joint ranges in radians."""

    @abstractmethod
    def move_joints(self, joints_rad: Sequence[float]) -> bool:
        """Move to an absolute six-joint target without silently clipping it."""


class CartesianController(ABC):
    @abstractmethod
    def move_tcp(self, pose: TcpPose) -> bool:
        """Move tool0 to an absolute pose."""


class GripperController(ABC):
    @property
    @abstractmethod
    def max_gripper_opening_m(self) -> float:
        """Maximum total opening between fingertips in metres."""

    @abstractmethod
    def set_gripper(self, opening_m: float) -> bool:
        """Set total fingertip opening; invalid values must be rejected."""


class DriverLifecycle(ABC):
    @abstractmethod
    def close(self) -> None:
        """Release resources owned by this driver. Must be idempotent."""


class RobotDriver(
    RobotStateReader,
    JointPositionController,
    CartesianController,
    GripperController,
    DriverLifecycle,
    ABC,
):
    """Common contract implemented by real and simulated robots."""


class EmergencyStopController(ABC):
    """Real-controller capability kept out of RobotDriver for ISP/LSP."""

    @abstractmethod
    def emergency_stop(self) -> bool:
        """Latch the controller stop."""

    @abstractmethod
    def rearm(self) -> bool:
        """Clear an operator-recoverable stop/fault."""


class PowerController(ABC):
    """Optional physical power capability."""

    @abstractmethod
    def power_on(self) -> bool:
        pass

    @abstractmethod
    def power_off(self) -> bool:
        pass


__all__ = [
    "CartesianController",
    "DriverLifecycle",
    "EmergencyStopController",
    "GripperController",
    "JointPositionController",
    "PowerController",
    "RobotDriver",
    "RobotStateReader",
]

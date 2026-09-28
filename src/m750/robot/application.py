"""Robot-control application use cases."""

from __future__ import annotations

from typing import Sequence

from .contracts import RobotDriver
from .types import RobotState, TcpPose


class RobotControl:
    """Backend-neutral commands for either a real or simulated robot.

    Swapping ROS for MuJoCo requires replacing only the injected RobotDriver.
    This class intentionally has no import from adapters.
    """

    def __init__(self, driver: RobotDriver) -> None:
        self._driver = driver

    @property
    def driver(self) -> RobotDriver:
        return self._driver

    def state(self) -> RobotState:
        return self._driver.read_state()

    def move_joints(self, joints_rad: Sequence[float]) -> bool:
        return self._driver.move_joints(joints_rad)

    def move_tcp(self, position_m, quaternion_xyzw) -> bool:
        return self._driver.move_tcp(TcpPose(tuple(position_m), tuple(quaternion_xyzw)))

    def set_gripper(self, opening_m: float) -> bool:
        return self._driver.set_gripper(float(opening_m))

    def close(self) -> None:
        self._driver.close()

    def __enter__(self) -> "RobotControl":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False


__all__ = ["RobotControl"]

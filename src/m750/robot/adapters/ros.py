"""ROS implementation of the public RobotDriver contract."""

from __future__ import annotations

import math
from typing import Optional, Sequence

from m750.ros.robot import Robot

from ..contracts import EmergencyStopController, PowerController, RobotDriver
from ..types import JointLimits, RobotState, TcpPose


def _finite(values, size: int):
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError):
        return None
    if len(result) != size or not all(math.isfinite(value) for value in result):
        return None
    return result


class RosRobotDriver(RobotDriver, EmergencyStopController, PowerController):
    """Adapter over the existing ROS Robot facade.

    All validation that defines the common RobotDriver semantics happens here.
    The legacy ROS implementation can therefore evolve independently.
    """

    def __init__(self, robot: Optional[Robot] = None) -> None:
        self._robot = robot or Robot()

    @property
    def legacy_robot(self) -> Robot:
        return self._robot

    @property
    def joint_limits_rad(self) -> JointLimits:
        return tuple(
            (float(lo), float(hi))
            for lo, hi in zip(self._robot.rad_min, self._robot.rad_max)
        )

    @property
    def max_gripper_opening_m(self) -> float:
        return float(self._robot.MAX_GRIPPER_M)

    def read_state(self) -> RobotState:
        joints = _finite(self._robot.qpos, 6)
        tcp_position = _finite(self._robot.tcp_pos, 3)
        tcp_quaternion = _finite(self._robot.tcp_quat, 4)
        tcp_pose = None
        if tcp_position is not None and tcp_quaternion is not None:
            try:
                tcp_pose = TcpPose(tcp_position, tcp_quaternion)
            except ValueError:
                tcp_pose = None

        opening = self._robot.gripper
        opening_value = None
        try:
            opening_value = float(opening)
            if not math.isfinite(opening_value) or opening_value < 0.0:
                opening_value = None
        except (TypeError, ValueError):
            opening_value = None

        connected = bool(self._robot.is_real_connected)
        armed = bool(self._robot.is_armed)
        return RobotState(
            joints_rad=joints,
            gripper_opening_m=opening_value,
            tcp_pose=tcp_pose,
            connected=connected,
            ready=connected and armed,
            metadata={
                "backend": "ros",
                "safety_state": self._robot.safety_state,
                "temperatures_c": tuple(float(v) for v in self._robot.temperatures()),
            },
        )

    def move_joints(self, joints_rad: Sequence[float]) -> bool:
        values = _finite(joints_rad, 6)
        if values is None:
            return False
        for value, (lower, upper) in zip(values, self.joint_limits_rad):
            if value < lower or value > upper:
                return False
        return bool(self._robot.set_pose(values))

    def move_tcp(self, pose: TcpPose) -> bool:
        return bool(
            self._robot.set_tcp_pose(
                pose.position_m,
                pose.quaternion_xyzw,
            )
        )

    def set_gripper(self, opening_m: float) -> bool:
        try:
            opening = float(opening_m)
        except (TypeError, ValueError):
            return False
        if (
            not math.isfinite(opening)
            or opening < 0.0
            or opening > self.max_gripper_opening_m
        ):
            return False
        return bool(self._robot.set_opening(opening))

    def emergency_stop(self) -> bool:
        return bool(self._robot.stop())

    def rearm(self) -> bool:
        return bool(self._robot.rearm())

    def power_on(self) -> bool:
        return bool(self._robot.power_on())

    def power_off(self) -> bool:
        return bool(self._robot.power_off())

    def close(self) -> None:
        closer = getattr(self._robot, "close", None)
        if callable(closer):
            closer()


__all__ = ["RosRobotDriver"]

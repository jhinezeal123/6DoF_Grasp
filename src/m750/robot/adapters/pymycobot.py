"""Official pymycobot implementation of the public RobotDriver contract."""

from __future__ import annotations

import math
import time
from typing import Optional, Sequence

from scipy.spatial.transform import Rotation

from m750.robot.arm import MyArmM750
from m750.robot.control import ArmController
from m750.spec import RobotSpec

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


class PymycobotRobotDriver(RobotDriver, EmergencyStopController, PowerController):
    """Real myArm M750 backend using Elephant Robotics official pymycobot API.

    Vendor-specific units are isolated here:
    - pymycobot joints are degrees, RobotDriver joints are radians.
    - pymycobot gripper is 0..100, RobotDriver gripper is metres.
    - Cartesian commands use repository Pinocchio IK then pymycobot write_angles.
    """

    def __init__(
        self,
        arm: Optional[MyArmM750] = None,
        controller: Optional[ArmController] = None,
        spec: Optional[RobotSpec] = None,
        *,
        joint_speed: int = 20,
        gripper_speed: int = 30,
        joint_timeout_s: float = 30.0,
        tcp_tolerance_mm: float = 2.0,
    ) -> None:
        self.spec = spec or RobotSpec()
        self.arm = arm or MyArmM750(self.spec)
        self.controller = controller or ArmController(arm=self.arm, spec=self.spec)
        self.joint_speed = int(joint_speed)
        self.gripper_speed = int(gripper_speed)
        self.joint_timeout_s = float(joint_timeout_s)
        self.tcp_tolerance_mm = float(tcp_tolerance_mm)

    @property
    def joint_limits_rad(self) -> JointLimits:
        margin = float(self.spec.margin_deg)
        return tuple(
            (
                math.radians(float(lower) + margin),
                math.radians(float(upper) - margin),
            )
            for lower, upper in zip(self.spec.fw_min_deg, self.spec.fw_max_deg)
        )

    @property
    def max_gripper_opening_m(self) -> float:
        return float(self.spec.gripper_max_opening_m)

    def _gripper_opening(self):
        raw = self.arm.open().get_gripper_value()
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return None, None
        if not math.isfinite(value) or value < 0.0 or value > 100.0:
            return None, value
        return value / 100.0 * self.max_gripper_opening_m, value

    def read_state(self) -> RobotState:
        q_deg = self.arm.q_deg
        if q_deg is None:
            return RobotState(
                joints_rad=None,
                gripper_opening_m=None,
                tcp_pose=None,
                connected=False,
                ready=False,
                metadata={"backend": "pymycobot"},
            )

        joints_rad = tuple(math.radians(float(value)) for value in q_deg)
        transform = self.controller.kin.fk_tool0(q_deg)
        tcp_pose = TcpPose(
            tuple(float(value) for value in transform.translation),
            tuple(float(value) for value in Rotation.from_matrix(transform.rotation).as_quat()),
        )
        opening, gripper_value = self._gripper_opening()
        device = self.arm.open()
        powered = device.is_powered_on() == 1
        temperatures = self.arm.temperatures()

        return RobotState(
            joints_rad=joints_rad,
            gripper_opening_m=opening,
            tcp_pose=tcp_pose,
            connected=True,
            ready=powered,
            metadata={
                "backend": "pymycobot",
                "gripper_value": gripper_value,
                "temperatures_c": None if temperatures is None else tuple(temperatures),
            },
        )

    def move_joints(self, joints_rad: Sequence[float]) -> bool:
        values = _finite(joints_rad, 6)
        if values is None:
            return False
        for value, (lower, upper) in zip(values, self.joint_limits_rad):
            if value < lower or value > upper:
                return False
        target_deg = tuple(math.degrees(value) for value in values)
        return bool(
            self.arm.write_joints(
                target_deg,
                speed=self.joint_speed,
                timeout_s=self.joint_timeout_s,
            )
        )

    def move_tcp(self, pose: TcpPose) -> bool:
        euler_deg = Rotation.from_quat(pose.quaternion_xyzw).as_euler(
            "xyz",
            degrees=True,
        )
        coords = [
            *(float(value) * 1000.0 for value in pose.position_m),
            *(float(value) for value in euler_deg),
        ]
        return bool(
            self.controller.move_tcp_to(
                coords,
                speed=self.joint_speed,
                tol=self.tcp_tolerance_mm,
                timeout_s=self.joint_timeout_s,
            )
        )

    def set_gripper(self, opening_m: float) -> bool:
        try:
            opening = float(opening_m)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(opening) or opening < 0.0 or opening > self.max_gripper_opening_m:
            return False

        value = int(round(opening / self.max_gripper_opening_m * 100.0))
        result = self.arm.open().set_gripper_value(value, self.gripper_speed)
        return result != -1 and result is not False

    def emergency_stop(self) -> bool:
        result = self.arm.open().stop()
        return result != -1 and result is not False

    def power_on(self) -> bool:
        return bool(self.arm.power_on())

    def power_off(self) -> bool:
        device = self.arm.open()
        result = device.power_off()
        if result == -1 or result is False:
            return False
        time.sleep(0.2)
        return device.is_powered_on() == 0

    def close(self) -> None:
        self.arm.close()


__all__ = ["PymycobotRobotDriver"]

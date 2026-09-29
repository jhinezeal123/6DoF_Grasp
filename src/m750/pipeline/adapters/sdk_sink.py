"""Pipeline Sink backed by the public RobotDriver abstraction."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from m750.robot import RobotDriver, TcpPose

from ..interfaces import Sink
from ..types import Action, ActionSpec


class RobotSink(Sink):
    """Translate pipeline actions into backend-neutral robot commands."""

    def __init__(
        self,
        robot: RobotDriver,
        spec: ActionSpec,
        *,
        write_callback: Callable[[RobotDriver, Action], Any] | None = None,
        halt_callback: Callable[[RobotDriver, str], Any] | None = None,
    ) -> None:
        super().__init__(spec)
        self.robot = robot
        self.write_callback = write_callback
        self.halt_callback = halt_callback

    def write(self, action: Action) -> None:
        if self.write_callback is not None:
            result = self.write_callback(self.robot, action)
        elif self.spec.mode == "joint_position":
            result = self.robot.move_joints(action.values)
        elif self.spec.mode == "tcp_pose":
            if len(action.values) != 7:
                raise ValueError("tcp_pose cần [x, y, z, qx, qy, qz, qw]")
            result = self.robot.move_tcp(
                TcpPose(
                    tuple(action.values[:3]),
                    tuple(action.values[3:]),
                )
            )
        elif self.spec.mode == "gripper":
            result = self.robot.set_gripper(action.values[0])
        else:
            raise ValueError(
                f"Chưa có writer mặc định cho mode {self.spec.mode!r}; "
                "hãy truyền write_callback"
            )
        if result is False:
            raise RuntimeError("Robot từ chối action")

    def halt(self, reason: str) -> None:
        if self.halt_callback is not None:
            result = self.halt_callback(self.robot, reason)
            if result is False:
                raise RuntimeError("Controller từ chối halt")


__all__ = ["RobotSink"]

"""Sink tùy chọn nối Action với facade ``program.robot.Robot``."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..interfaces import Sink
from ..types import Action, ActionSpec


class RobotSink(Sink):
    """Chuyển các mode chuẩn sang API Robot.

    ``halt_callback`` mặc định là no-op có chủ ý: ``Robot.stop()`` là dừng khẩn
    cấp có chốt lỗi và không phù hợp để tự gọi khi reset/kết thúc episode. Có thể
    truyền callback controller-specific nếu ứng dụng cần cancel/hold.
    """

    def __init__(
        self,
        robot: Any,
        spec: ActionSpec,
        *,
        write_callback: Callable[[Any, Action], Any] | None = None,
        halt_callback: Callable[[Any, str], Any] | None = None,
    ) -> None:
        super().__init__(spec)
        self.robot = robot
        self.write_callback = write_callback
        self.halt_callback = halt_callback

    def write(self, action: Action) -> None:
        if self.write_callback is not None:
            result = self.write_callback(self.robot, action)
        elif self.spec.mode == "joint_position":
            result = self.robot.set_pose(action.values)
        elif self.spec.mode == "tcp_pose":
            if len(action.values) != 7:
                raise ValueError("tcp_pose cần [x, y, z, qx, qy, qz, qw]")
            result = self.robot.set_tcp_pose(action.values[:3], action.values[3:])
        elif self.spec.mode == "gripper":
            result = self.robot.set_opening(action.values[0])
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

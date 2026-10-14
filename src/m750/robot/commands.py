"""Thực thi lệnh robot và đổi đơn vị; không parse argv hoặc quản lý serial."""

from __future__ import annotations

import math
from dataclasses import asdict

from .application import RobotControl
from .contracts import EmergencyStopController, PowerController


class RobotCommands:
    """Giữ command contract của CLI: degree, mm và Euler XYZ ở đầu vào."""

    def __init__(self, driver):
        self._driver = driver
        self._control = RobotControl(driver)

    def execute(self, args):
        app = self._control
        driver = self._driver
        if args.command in ("state", "angles"):
            state = app.state()
            if args.command == "state":
                return state.connected, {"state": asdict(state)}
            degrees = None if state.joints_rad is None else [math.degrees(v) for v in state.joints_rad]
            return degrees is not None, {"joints_deg": degrees}
        if args.command == "joints":
            return app.move_joints(tuple(math.radians(v) for v in args.degrees)), {}
        if args.command == "joint":
            state = app.state()
            if state.joints_rad is None:
                raise RuntimeError("không đọc được góc joint; không gửi lệnh")
            target = list(state.joints_rad)
            target[args.id - 1] = math.radians(args.degree)
            return app.move_joints(target), {}
        if args.command == "gripper":
            opening_m = (
                args.opening / 100.0 * driver.max_gripper_opening_m
                if args.unit == "percent"
                else args.opening / 1000.0
            )
            return app.set_gripper(opening_m), {}
        if args.command == "tcp":
            from scipy.spatial.transform import Rotation

            xyz = [v / 1000.0 for v in args.pose[:3]]
            quaternion = Rotation.from_euler("xyz", args.pose[3:], degrees=True).as_quat()
            return app.move_tcp(xyz, quaternion), {}
        if args.command == "stop" and isinstance(driver, EmergencyStopController):
            return driver.emergency_stop(), {}
        if isinstance(driver, PowerController):
            if args.command == "power-on":
                return driver.power_on(), {}
            if args.command == "power-off":
                return driver.power_off(), {}
        raise ValueError("backend không hỗ trợ lệnh " + args.command)

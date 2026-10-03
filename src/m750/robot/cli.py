"""CLI pymycobot: chỉ chuyển input/đơn vị, mọi lệnh đi qua RobotDriver."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict

from m750.spec import RobotSpec

from .application import RobotControl
from .contracts import EmergencyStopController, PowerController


def _finite(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError("giá trị phải hữu hạn")
    return number


def _positive(value: str) -> float:
    number = _finite(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("giá trị phải lớn hơn 0")
    return number


def build_parser() -> argparse.ArgumentParser:
    spec = RobotSpec()
    parser = argparse.ArgumentParser(description="Điều khiển myArm M750 bằng pymycobot.")
    parser.add_argument("--port", default=spec.port, help="cổng serial")
    parser.add_argument("--baud", type=int, default=spec.baudrate)
    parser.add_argument("--speed", type=int, default=20, help="tốc độ joint/gripper 1..100")
    parser.add_argument("--timeout", type=_positive, default=30.0, help="timeout joint, giây")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("state", help="đọc trạng thái, không gửi lệnh chuyển động")
    commands.add_parser("angles", help="đọc sáu góc joint, đơn vị độ")
    commands.add_parser("power-on", help="bật nguồn servo")
    commands.add_parser("power-off", help="tắt servo; đỡ tay trước khi tắt")
    commands.add_parser("stop", help="stop qua serial; không thay thế hardware E-stop")
    joints = commands.add_parser("joints", help="đặt sáu joint tuyệt đối, đơn vị độ")
    joints.add_argument("degrees", nargs=6, type=_finite, metavar="DEG")
    joint = commands.add_parser("joint", help="đặt một joint, giữ năm joint còn lại")
    joint.add_argument("id", type=int, choices=range(1, 7))
    joint.add_argument("degree", type=_finite)
    gripper = commands.add_parser("gripper", help="đặt tổng độ mở hai ngón")
    gripper.add_argument("opening", type=_finite)
    gripper.add_argument("--unit", choices=("mm", "percent"), default="mm")
    tcp = commands.add_parser("tcp", help="tool0: X Y Z mm, RX RY RZ độ Euler XYZ")
    tcp.add_argument("pose", nargs=6, type=_finite, metavar="VALUE")
    return parser


def _validate(args, parser):
    spec = RobotSpec()
    if args.baud <= 0 or not 1 <= args.speed <= 100:
        parser.error("baud phải > 0; speed phải trong 1..100")
    if args.command == "joints":
        values = enumerate(args.degrees)
    elif args.command == "joint":
        values = [(args.id - 1, args.degree)]
    else:
        values = []
    for index, value in values:
        lower = spec.fw_min_deg[index] + spec.margin_deg
        upper = spec.fw_max_deg[index] - spec.margin_deg
        if not lower <= value <= upper:
            parser.error(f"joint {index + 1} phải trong [{lower:g}, {upper:g}] độ")
    if args.command == "gripper":
        maximum = 100.0 if args.unit == "percent" else spec.gripper_max_opening_m * 1000.0
        if not 0 <= args.opening <= maximum:
            parser.error(f"độ mở phải trong [0, {maximum:g}] {args.unit}")


def build_driver(args):
    """Composition duy nhất; --help và input lỗi không import driver phần cứng."""
    from .adapters.pymycobot import PymycobotRobotDriver

    return PymycobotRobotDriver(
        spec=RobotSpec(port=args.port, baudrate=args.baud),
        joint_speed=args.speed,
        gripper_speed=args.speed,
        joint_timeout_s=args.timeout,
    )


def _execute(args, driver):
    app = RobotControl(driver)
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


def main(argv=None, *, driver_factory=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate(args, parser)
    driver = None
    exit_code = 1
    try:
        driver = (build_driver if driver_factory is None else driver_factory)(args)
        ok, details = _execute(args, driver)
        print(json.dumps({"ok": bool(ok), "command": args.command, **details}, ensure_ascii=False))
        exit_code = 0 if ok else 1
    except KeyboardInterrupt:
        if isinstance(driver, EmergencyStopController):
            try:
                if not driver.emergency_stop():
                    print("Không xác nhận được lệnh stop qua serial.", file=sys.stderr)
            except Exception as exc:
                print(f"Không gửi được lệnh stop qua serial: {exc}", file=sys.stderr)
        print("Đã ngắt lệnh.", file=sys.stderr)
        exit_code = 130
    except (OSError, RuntimeError, ValueError, ImportError) as exc:
        print(f"LỖI: {exc}", file=sys.stderr)
    finally:
        if driver is not None:
            try:
                driver.close()
            except Exception as exc:
                print(f"Không đóng được driver: {exc}", file=sys.stderr)
                if exit_code != 130:
                    exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

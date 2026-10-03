"""CLI qua fake RobotDriver: kiểm tra đơn vị, validation và cleanup, không serial."""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from m750 import EmergencyStopController, PowerController
from m750.robot.cli import main
from test_feature_first_architecture import MemoryDriver


class Driver(MemoryDriver, PowerController, EmergencyStopController):
    def __init__(self):
        super().__init__("fake")
        self.events = []

    def power_on(self):
        self.events.append("power-on")
        return True

    def power_off(self):
        self.events.append("power-off")
        return True

    def emergency_stop(self):
        self.events.append("stop")
        return True


def run(args, driver):
    return main(args, driver_factory=lambda options: driver)


def test_joints_convert_degrees_and_close_driver():
    driver = Driver()
    assert run(["joints", "10", "20", "30", "40", "50", "60"], driver) == 0
    assert driver.joints == pytest.approx(tuple(math.radians(v) for v in [10, 20, 30, 40, 50, 60]))
    assert driver.closed


def test_existing_command_object_keeps_its_unit_and_lifecycle_contract():
    from m750.robot.cli import build_parser
    from m750.robot.commands import RobotCommands

    driver = Driver()
    args = build_parser().parse_args(["joints", "10", "20", "30", "40", "50", "60"])
    assert RobotCommands(driver).execute(args) == (True, {})
    assert driver.joints == pytest.approx(tuple(math.radians(v) for v in args.degrees))
    assert not driver.closed


def test_single_joint_preserves_other_five():
    driver = Driver()
    driver.joints = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    assert run(["joint", "3", "10"], driver) == 0
    assert driver.joints == pytest.approx((0.1, 0.2, math.radians(10), 0.4, 0.5, 0.6))


@pytest.mark.parametrize(
    "args, expected",
    [
        (["gripper", "30"], 0.03),
        (["gripper", "50", "--unit", "percent"], 0.04),
    ],
)
def test_gripper_has_explicit_mm_and_percent_units(args, expected):
    driver = Driver()
    assert run(args, driver) == 0
    assert driver.gripper == pytest.approx(expected)


def test_tcp_converts_mm_euler_to_metres_xyzw():
    driver = Driver()
    assert run(["tcp", "100", "200", "300", "0", "0", "90"], driver) == 0
    assert driver.pose.position_m == pytest.approx((0.1, 0.2, 0.3))
    assert driver.pose.quaternion_xyzw == pytest.approx((0.0, 0.0, 2**-0.5, 2**-0.5))


@pytest.mark.parametrize("command", ["power-on", "power-off", "stop"])
def test_capability_commands_use_the_driver(command):
    driver = Driver()
    assert run([command], driver) == 0
    assert driver.events == [command]
    assert driver.closed


@pytest.mark.parametrize(
    "args",
    [
        ["joints", "nan", "0", "0", "0", "0", "0"],
        ["joints", "165", "0", "0", "0", "0", "0"],
        ["joint", "7", "10"],
        ["gripper", "70"],
        ["gripper", "101", "--unit", "percent"],
        ["--speed", "0", "state"],
        ["--baud", "0", "state"],
        ["--timeout", "0", "state"],
    ],
)
def test_invalid_input_never_creates_hardware(args):
    def forbidden(options):
        raise AssertionError("không được mở driver khi input lỗi")

    with pytest.raises(SystemExit) as error:
        main(args, driver_factory=forbidden)
    assert error.value.code == 2


def test_state_is_json_and_has_no_motion_commands(capsys):
    driver = Driver()
    assert run(["state"], driver) == 0
    response = json.loads(capsys.readouterr().out)
    assert response["state"]["metadata"]["backend"] == "fake"
    assert driver.events == []
    assert driver.closed


def test_runtime_error_still_closes_the_driver(capsys):
    driver = Driver()

    def fail():
        raise RuntimeError("serial read failed")

    driver.read_state = fail
    assert run(["state"], driver) == 1
    assert driver.closed
    assert "serial read failed" in capsys.readouterr().err


def test_root_help_is_offline_and_does_not_import_vendor():
    root = Path(__file__).resolve().parents[1]
    code = "import sys; from m750.robot.cli import build_parser; build_parser(); assert 'pymycobot' not in sys.modules; assert 'pinocchio' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    env = {**os.environ, "M750_PYTHON": sys.executable}
    help_result = subprocess.run(
        [str(root / "robot"), "--help"], cwd="/tmp", env=env, text=True, capture_output=True
    )
    assert help_result.returncode == 0, help_result.stderr
    assert "Điều khiển myArm M750" in help_result.stdout


def test_keyboard_interrupt_stops_and_closes_driver():
    driver = Driver()

    def interrupt(values):
        raise KeyboardInterrupt()

    driver.move_joints = interrupt
    assert run(["joints", "0", "0", "0", "0", "0", "0"], driver) == 130
    assert driver.events == ["stop"]
    assert driver.closed


@pytest.mark.parametrize("stop_result", [False, RuntimeError("serial disconnected")])
def test_interrupt_stop_failure_still_closes_and_returns_130(capsys, stop_result):
    driver = Driver()

    def interrupt(values):
        raise KeyboardInterrupt()

    def failed_stop():
        driver.events.append("stop")
        if isinstance(stop_result, Exception):
            raise stop_result
        return stop_result

    driver.move_joints = interrupt
    driver.emergency_stop = failed_stop
    assert run(["joints", "0", "0", "0", "0", "0", "0"], driver) == 130
    assert driver.closed and driver.events == ["stop"]
    error = capsys.readouterr().err
    assert "stop qua serial" in error and "Đã ngắt lệnh" in error


def test_interrupt_keeps_130_when_close_also_fails(capsys):
    driver = Driver()

    def interrupt(values):
        raise KeyboardInterrupt()

    def failed_close():
        driver.closed = True
        raise OSError("close failed")

    driver.move_joints = interrupt
    driver.close = failed_close
    assert run(["joints", "0", "0", "0", "0", "0", "0"], driver) == 130
    assert driver.closed and driver.events == ["stop"]
    assert "close failed" in capsys.readouterr().err


def test_normal_command_reports_failed_cleanup(capsys):
    driver = Driver()

    def failed_close():
        raise OSError("close failed")

    driver.close = failed_close
    assert run(["angles"], driver) == 1
    assert "close failed" in capsys.readouterr().err


def test_stop_cannot_take_a_serial_port_owned_by_another_process(capsys):
    driver = Driver()

    def held_port():
        raise RuntimeError("/dev/ttyACM1 đang bị giữ: pid 123")

    driver.emergency_stop = held_port
    assert run(["stop"], driver) == 1
    assert driver.closed and driver.events == []
    assert "đang bị giữ" in capsys.readouterr().err

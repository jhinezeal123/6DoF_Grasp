"""Luồng vận hành từ root, dùng fake subprocess/camera; không mở serial."""

import io
import json
import math
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from m750.operator import camera
from m750.operator.config import ProfileStore
from m750.operator.console import main
from m750.operator.tasks import Tasks
from m750.operator.terminal import Back, CommandRunner, Terminal


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def factory(tmp_path, monkeypatch):
    monkeypatch.delenv("M750_PYTHON", raising=False)
    for name in (
        ".venv/bin/python",
        "grasp_pipeline_repo/.venv/bin/python",
        "grasp_pipeline_repo/scripts/worker.sh",
        "grasp_pipeline_repo/start",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fake command; never execute")
        path.chmod(0o755)
    store = ProfileStore(tmp_path)
    store.save(store.load())

    def create(answers="", dry=False):
        output = io.StringIO()
        view = Terminal(io.StringIO(answers), output)
        task = Tasks(tmp_path, store, view, CommandRunner(tmp_path, view, dry))
        return task, output

    return create


def test_root_catalog_works_outside_repo_without_robot_or_site_packages():
    result = subprocess.run(
        [str(ROOT / "start"), "--list"],
        cwd="/tmp",
        env={**os.environ, "M750_PYTHON": sys.executable},
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Điều khiển robot thật" in result.stdout
    assert "VLA" in result.stdout and "ground-truth" in result.stdout
    code = "import start, sys; assert not any(n in sys.modules for n in ('m750','numpy','mujoco','pinocchio','pymycobot')); start.main(['--list'])"
    cold = subprocess.run(
        [sys.executable, "-S", "-c", code], cwd=str(ROOT), capture_output=True, text=True
    )
    assert cold.returncode == 0, cold.stderr


def test_menu_eof_bad_choice_and_return_do_not_touch_robot(factory):
    task, output = factory()
    with patch("m750.operator.terminal.CommandRunner._execute") as run:
        assert (
            main(["--config", str(task.store.path)], Terminal(io.StringIO("bad\n0\n"), output)) == 0
        )
        assert main(["--config", str(task.store.path)], Terminal(io.StringIO(""), output)) == 0
    run.assert_not_called()
    assert "Chọn một số" in output.getvalue()


def test_serial_profile_is_private_and_used_again(factory):
    task, _ = factory()
    selected = replace(task.profile, port="/dev/serial/by-id/test", speed=15, camera_port=8093)
    task.save(selected)
    next_task, _ = factory()
    assert next_task.profile == selected
    assert task.store.path.stat().st_mode & 0o777 == 0o600


def test_atomic_profile_failure_keeps_previous_data(factory):
    task, _ = factory()
    before = task.store.path.read_bytes()
    with patch("m750.operator.config.os.replace", side_effect=OSError("disk error")):
        with pytest.raises(OSError):
            task.save(replace(task.profile, speed=15))
    assert task.store.path.read_bytes() == before
    assert list(task.store.path.parent.iterdir()) == [task.store.path]


@pytest.mark.parametrize(
    "changes",
    [
        {"speed": 0},
        {"speed": True},
        {"baud": 0},
        {"port": ""},
        {"camera_port": 65536},
        {"socket": ""},
    ],
)
def test_invalid_profile_is_rejected_before_execution(factory, changes):
    task, _ = factory()
    with pytest.raises(ValueError):
        task.save(replace(task.profile, **changes))


def test_motion_shows_command_before_confirm_and_can_be_cancelled(factory):
    task, output = factory("3\n30\nn\n")
    with patch("m750.operator.terminal.CommandRunner._execute") as run:
        with pytest.raises(Back):
            task.robot()
    run.assert_not_called()
    text = output.getvalue()
    assert "gripper 30" in text
    assert text.index("Lệnh:") < text.index("Gửi lệnh này")


def test_joint_command_uses_saved_serial_and_does_not_auto_power_on(factory):
    task, _ = factory("1\n3\n10\ny\n")
    task.save(replace(task.profile, port="/dev/serial/by-id/test", speed=15))
    response = subprocess.CompletedProcess([], 0, '{"ok":true,"command":"joint"}\n', "")
    with patch("m750.operator.terminal.CommandRunner._execute", return_value=response) as run:
        task.robot()
    argv = run.call_args.args[0]
    assert argv[-3:] == ["joint", "3", "10"]
    assert argv[2] == "/dev/serial/by-id/test" and argv[6] == "15"
    assert run.call_count == 1 and "power-on" not in argv


@pytest.mark.parametrize("choice,command", [("5", "power-on"), ("6", "power-off"), ("7", "stop")])
def test_power_and_serial_stop_route_through_root_cli(factory, choice, command):
    task, output = factory(choice + "\ny\n")
    response = subprocess.CompletedProcess([], 0, '{"ok":true}\n', "")
    with patch("m750.operator.terminal.CommandRunner._execute", return_value=response) as run:
        task.robot()
    assert run.call_args.args[0][-1] == command
    if command == "stop":
        assert "không phải hardware E-stop" in output.getvalue()


def test_invalid_numeric_input_never_runs_driver(factory):
    task, _ = factory("3\nnan\n")
    with patch("m750.operator.terminal.CommandRunner._execute") as run:
        with pytest.raises(ValueError, match="hữu hạn"):
            task.robot()
    run.assert_not_called()


def test_feedback_is_formatted_in_human_units(factory):
    task, output = factory()
    state = {
        "ok": True,
        "state": {
            "joints_rad": [0, math.pi / 2, 0, 0, 0, 0],
            "gripper_opening_m": 0.03,
            "tcp_pose": {"position_m": [0.1, 0.2, 0.3]},
        },
    }
    task.feedback(subprocess.CompletedProcess([], 0, "driver log\n" + json.dumps(state) + "\n", ""))
    text = output.getvalue()
    assert "90.0" in text and "30.0 mm" in text and "100.0 / 200.0 / 300.0" in text


def test_simulation_uses_scoped_preset_and_separate_interpreters(factory):
    task, output = factory("y\n")
    before_profile = task.store.path.read_bytes()
    before_env = dict(os.environ)
    with patch(
        "m750.operator.terminal.CommandRunner._execute", return_value=subprocess.CompletedProcess([], 0)
    ) as run:
        task.simulation()
    assert run.call_count == 2
    worker = run.call_args_list[0]
    assert "grasp_pipeline_repo/scripts/worker.sh" in worker.args[0][1]
    assert worker.kwargs["env"]["YOLOE_CONF"] == "0.05"
    assert worker.kwargs["env"]["GRASP_DEPTH_BACKEND"] == "da3"
    harness = run.call_args_list[1]
    assert harness.args[0][0] == str(task.root / ".venv/bin/python")
    assert harness.args[0][harness.args[0].index("--volume") + 1] == "gravity"
    assert harness.args[0][harness.args[0].index("--depth-source") + 1] == "worker"
    assert "ground-truth" in output.getvalue()
    assert task.store.path.read_bytes() == before_profile and dict(os.environ) == before_env


def test_missing_robot_environment_blocks_worker_restart(factory):
    task, _ = factory("y\n")
    (task.root / ".venv/bin/python").unlink()
    with patch("m750.operator.terminal.CommandRunner._execute") as run:
        with pytest.raises(RuntimeError, match="Thiếu Python robot"):
            task.simulation()
    run.assert_not_called()


def test_dry_run_does_not_fork_or_write_configuration(factory):
    task, output = factory("3\n30\n", dry=True)
    before = task.store.path.read_bytes()
    with patch("m750.operator.terminal.CommandRunner._execute") as run:
        task.robot()
        task.simulation()
        task.save(replace(task.profile, speed=15))
    run.assert_not_called()
    assert task.store.path.read_bytes() == before
    assert "Chỉ xem lệnh" in output.getvalue() and "gravity" in output.getvalue()


def test_perception_uses_child_launcher_socket_and_captured_photo(factory):
    task, _ = factory()
    photo = task.root / "private.jpg"
    photo.write_bytes(b"fixture")
    task.profile = replace(task.profile, last_photo=str(photo))
    with patch(
        "m750.operator.terminal.CommandRunner._execute", return_value=subprocess.CompletedProcess([], 0)
    ) as run:
        task.perception()
    argv = run.call_args.args[0]
    assert argv[0] == "bash" and argv[1] == str(Path(task.profile.pipeline_repo) / "start")
    assert argv[-2:] == ["--image", str(photo)]
    assert argv[2:4] == ["--socket", task.profile.socket]
    assert str(task.root / ".venv/bin/python") not in argv


def test_ros_launch_can_be_declined_without_powering_servo(factory):
    task, _ = factory("1\nn\n")
    with patch("m750.operator.terminal.CommandRunner._execute") as run:
        task.advanced()
    run.assert_not_called()


def test_camera_stream_always_stops_after_interrupt():
    events = []

    class Stream:
        def __init__(self, **kwargs):
            events.append(kwargs)

        def start(self):
            events.append("start")

        def stop(self):
            events.append("stop")

    module = SimpleNamespace(MjpegStream=Stream, capture=lambda **kwargs: None)
    with patch.dict(sys.modules, {"m750.camera": module}), patch(
        "m750.operator.camera.threading.Event"
    ) as event:
        event.return_value.wait.side_effect = KeyboardInterrupt()
        assert camera.main(["stream", "--device", "/dev/video9", "--port", "8093"]) == 130
    assert events == [{"device": "/dev/video9", "port": 8093}, "start", "stop"]


def test_camera_stream_start_failure_still_cleans_up():
    stream = SimpleNamespace(
        start=lambda: (_ for _ in ()).throw(RuntimeError("camera failed")),
        stop=lambda: events.append("stop"),
    )
    events = []
    with patch.dict(
        sys.modules,
        {"m750.camera": SimpleNamespace(MjpegStream=lambda **kwargs: stream, capture=None)},
    ):
        with pytest.raises(RuntimeError, match="camera failed"):
            camera.main(["stream", "--device", "/dev/video9"])
    assert events == ["stop"]


def test_camera_photo_uses_existing_capture_and_reports_failure():
    seen = []
    module = SimpleNamespace(MjpegStream=None, capture=lambda **kwargs: seen.append(kwargs))
    with patch.dict(sys.modules, {"m750.camera": module}):
        assert camera.main(["photo", "--device", "/dev/video9", "--out", "private.jpg"]) == 1
    assert seen == [{"path": "private.jpg", "device": "/dev/video9"}]


def test_child_interrupt_keeps_exit_130(factory):
    task, output = factory()
    with patch.object(Tasks, "status", side_effect=KeyboardInterrupt):
        assert (
            main(["status", "--config", str(task.store.path)], Terminal(io.StringIO(""), output))
            == 130
        )
    with patch(
        "m750.operator.terminal.CommandRunner._execute",
        return_value=subprocess.CompletedProcess([], 130, "", ""),
    ), pytest.raises(KeyboardInterrupt):
        task.runner.run(["fake"])


def test_root_back_exits_without_running_a_task(factory):
    task, output = factory()
    with patch("m750.operator.terminal.CommandRunner._execute") as run:
        assert main(
            ["--config", str(task.store.path)], Terminal(io.StringIO(":q\n"), output)
        ) == 0
    run.assert_not_called()
    assert "Đã thoát menu" in output.getvalue()


def test_readiness_labels_keep_their_priority(factory):
    task, _ = factory()
    expected = {
        "status": "cần thiết bị serial", "robot": "cần thiết bị serial",
        "camera": "cần camera", "simulation": "volume ground-truth",
        "perception": "có menu con", "advanced": "ROS cần SDK; VLA thử nghiệm",
        "configure": "có thể mở", "setup": "cần Conda",
    }
    assert {key: task.availability(key) for key in expected} == expected
    (task.root / ".venv/bin/python").unlink()
    expected.update({key: "cần môi trường" for key in ("status", "robot", "camera", "simulation")})
    assert {key: task.availability(key) for key in expected} == expected

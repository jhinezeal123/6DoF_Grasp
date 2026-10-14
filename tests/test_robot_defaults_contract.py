"""Khóa default server; menu bootstrap không được cần package robot."""

from m750 import spec
import os
import subprocess
import sys
from pathlib import Path


def test_server_defaults_are_unchanged():
    assert spec.DEFAULT_PORT == "/dev/ttyACM1"
    assert spec.DEFAULT_BAUDRATE == 1_000_000
    assert spec.DEFAULT_CAMERA == "/dev/v4l/by-id/usb-046d_Logitech_Webcam_C925e_3F4C8F2F-video-index0"


def test_defaults_are_available_without_importing_robot_api():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-S", "-c", "import sys; import m750_defaults; assert 'm750' not in sys.modules; print(m750_defaults.DEFAULT_PORT)"],
        env={**os.environ, "PYTHONPATH": str(root / "src")}, text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == spec.DEFAULT_PORT

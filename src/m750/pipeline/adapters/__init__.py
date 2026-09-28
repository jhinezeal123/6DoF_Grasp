"""Adapter mẫu nối pipeline với SDK hoặc implementation riêng."""

from .dry_run import DryRunSink
from .sdk_model import ModelPolicy
from .sdk_sink import RobotSink
from .sdk_source import CameraRobotSource

__all__ = ["CameraRobotSource", "DryRunSink", "ModelPolicy", "RobotSink"]

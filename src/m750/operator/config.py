"""Profile máy của menu; calibration và solver hiện có giữ nguyên."""

import json
import os
import tempfile
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from m750_defaults import DEFAULT_BAUDRATE, DEFAULT_CAMERA, DEFAULT_PORT


@dataclass
class Profile:
    port: str = DEFAULT_PORT
    baud: int = DEFAULT_BAUDRATE
    speed: int = 20
    camera_device: str = DEFAULT_CAMERA
    camera_port: int = 8083
    pipeline_repo: str = ""
    socket: str = ""
    last_photo: str = ""
    ros_lab: str = ""
    ros_python: str = ""
    ros_setup: str = ""

    def validate(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name not in ("baud", "speed", "camera_port") and not isinstance(value, str):
                raise ValueError("Cấu hình %s phải là chuỗi." % field.name)
        if not self.port or not self.camera_device or not self.pipeline_repo or not self.socket:
            raise ValueError("Port, camera, repo perception và socket không được để trống.")
        if type(self.baud) is not int or self.baud <= 0:
            raise ValueError("Baud phải là số nguyên dương.")
        if type(self.speed) is not int or not 1 <= self.speed <= 100:
            raise ValueError("Tốc độ phải trong 1..100.")
        if type(self.camera_port) is not int or not 1 <= self.camera_port <= 65535:
            raise ValueError("Port camera phải trong 1..65535.")

    def ros_environment(self):
        return {
            name: value
            for name, value in (
                ("MYARM_LAB", self.ros_lab),
                ("MYARM_PY", self.ros_python),
                ("MYARM_ROS_SETUP", self.ros_setup),
            )
            if value
        }


class ProfileStore:
    def __init__(self, root, path=None):
        self.root = Path(root)
        self.path = Path(path) if path else self.root / ".local_data" / "operator.json"

    def load(self):
        candidates = (self.root / "grasp_pipeline_repo", self.root.parent / "pipeline_grasppose")
        pipeline = next(
            (p for p in candidates if (p / "scripts/worker.sh").is_file()), candidates[0]
        )
        profile = Profile(
            pipeline_repo=str(pipeline),
            socket=os.environ.get("GRASP_WORKER_SOCKET", str(pipeline / ".runtime/worker.sock")),
            ros_lab=os.environ.get("MYARM_LAB", ""),
            ros_python=os.environ.get("MYARM_PY", ""),
            ros_setup=os.environ.get("MYARM_ROS_SETUP", ""),
        )
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or data.pop("version", None) != 1:
                raise ValueError("Profile không đúng version 1: %s" % self.path)
            if not set(data).issubset({f.name for f in fields(Profile)}):
                raise ValueError("Profile có trường không được hỗ trợ: %s" % self.path)
            profile = Profile(**{**asdict(profile), **data})
        profile.validate()
        return profile

    def save(self, profile):
        profile.validate()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=str(self.path.parent), delete=False
            ) as handle:
                temporary = Path(handle.name)
                json.dump({"version": 1, **asdict(profile)}, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            os.replace(str(temporary), str(self.path))
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

"""Source dùng Camera và Robot hiện có trong SDK.

Adapter này đọc một snapshot từ cache của Camera và Robot. Với camera ROS cần
timestamp đồng bộ nghiêm ngặt giữa nhiều sensor, hãy thay bằng Source riêng
dùng message_filters; pipeline không tự ghép các mẫu ``latest`` khác tuổi.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from typing import Any

from ..interfaces import Source
from ..types import Episode, Observation


class CameraRobotSource(Source):
    """Đọc ``camera.photo()`` và ``robot.qpos`` thành Observation."""

    def __init__(
        self,
        camera: Any | Mapping[str, Any],
        robot: Any,
        *,
        max_age_s: float = 1.0,
        camera_name: str = "front",
        state_reader: Any | None = None,
    ) -> None:
        if not math.isfinite(max_age_s) or max_age_s <= 0:
            raise ValueError("max_age_s phải hữu hạn và dương")
        self.cameras = camera if isinstance(camera, Mapping) else {camera_name: camera}
        self.robot = robot
        self.max_age_s = max_age_s
        self.state_reader = state_reader
        self._sequence = 0

    def reset(self, episode: Episode) -> None:
        self._sequence = 0

    def read(self, timeout_s: float) -> Observation:
        deadline = time.monotonic() + max(float(timeout_s), 0.0)
        while True:
            received = time.monotonic()
            images = {}
            for name, camera in self.cameras.items():
                images[name] = camera.photo()
            state_value = self.state_reader() if self.state_reader else self.robot.qpos
            try:
                state = tuple(float(value) for value in state_value)
            except (TypeError, ValueError) as exc:
                raise TimeoutError("Không đọc được trạng thái robot") from exc
            if state and all(math.isfinite(value) for value in state):
                self._sequence += 1
                stamp = time.time_ns()
                return Observation(
                    images=images,
                    state=state,
                    stamps_ns={name: stamp for name in (*images, "state")},
                    valid_until=received + self.max_age_s,
                    sequence=self._sequence,
                )
            if time.monotonic() >= deadline:
                raise TimeoutError("Chưa có feedback robot trong read_timeout_s")
            time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))


__all__ = ["CameraRobotSource"]

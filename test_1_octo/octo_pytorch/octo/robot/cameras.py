from __future__ import annotations

import threading
import time
from typing import Dict, Optional, Tuple

import numpy as np

from octo.robot.config import CameraConfig
from octo.robot.utils import (
    apply_image_transform,
    center_crop_resize,
    ensure_hw3_uint8,
    make_zero_image,
)


class OpenCVCameraStream:
    def __init__(self, config: CameraConfig):
        self.config = config
        self._cap = None
        self._lock = threading.Lock()
        self._thread = None
        self._stop_event = threading.Event()
        self._latest_rgb: Optional[np.ndarray] = None
        self._latest_timestamp: float = 0.0

    def start(self) -> None:
        if not self.config.enabled:
            return
        import cv2

        cap = cv2.VideoCapture(self.config.index, self.config.api_preference)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        cap.set(cv2.CAP_PROP_FPS, self.config.fps)
        if self.config.fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.config.fourcc))
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)
        except Exception:
            pass
        if not cap.isOpened():
            raise RuntimeError(
                f"Failed to open camera '{self.config.name}' on index {self.config.index}"
            )
        self._cap = cap

        for _ in range(max(self.config.warmup_frames, 0)):
            ok, frame = self._cap.read()
            if not ok:
                continue
            self._update_latest(frame)

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def _process_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        import cv2

        if self.config.convert_rgb:
            frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        else:
            frame = frame_bgr
        frame = ensure_hw3_uint8(frame)
        frame = apply_image_transform(
            frame,
            crop_top=self.config.crop_top,
            crop_bottom=self.config.crop_bottom,
            crop_left=self.config.crop_left,
            crop_right=self.config.crop_right,
            flip_horizontal=self.config.flip_horizontal,
            flip_vertical=self.config.flip_vertical,
            rotation_deg=self.config.rotation_deg,
        )
        return frame

    def _update_latest(self, frame_bgr: np.ndarray) -> None:
        frame = self._process_frame(frame_bgr)
        with self._lock:
            self._latest_rgb = frame
            self._latest_timestamp = time.time()

    def _reader_loop(self) -> None:
        while not self._stop_event.is_set():
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            self._update_latest(frame)

    def read_rgb(self) -> np.ndarray:
        frame, _ = self.read_rgb_with_timestamp()
        return frame

    def read_rgb_with_timestamp(self) -> Tuple[np.ndarray, float]:
        if not self.config.enabled:
            raise RuntimeError(f"Camera '{self.config.name}' is disabled")
        with self._lock:
            if self._latest_rgb is None:
                raise RuntimeError(f"Camera '{self.config.name}' has no frame yet")
            frame = self._latest_rgb.copy()
            ts = float(self._latest_timestamp)
        age = time.time() - ts
        if (not self.config.allow_stale_frames) and age > float(self.config.stale_timeout_s):
            raise RuntimeError(
                f"Camera '{self.config.name}' frame is stale: age={age:.3f}s > {self.config.stale_timeout_s:.3f}s"
            )
        return frame, ts

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class MultiCameraRig:
    def __init__(self, cameras: Dict[str, CameraConfig]):
        self._streams: Dict[str, OpenCVCameraStream] = {}
        for name, cfg in cameras.items():
            if cfg.enabled:
                self._streams[name] = OpenCVCameraStream(cfg)

    def start(self) -> None:
        for stream in self._streams.values():
            stream.start()

    def stop(self) -> None:
        for stream in self._streams.values():
            stream.stop()

    def snapshot(self, target_sizes: Optional[Dict[str, tuple]] = None) -> Dict[str, np.ndarray]:
        obs = {}
        for name, stream in self._streams.items():
            image = stream.read_rgb()
            key = f"image_{name}"
            if target_sizes and name in target_sizes:
                image = center_crop_resize(image, target_sizes[name])
            obs[key] = image
        return obs

    def snapshot_with_metadata(self, target_sizes: Optional[Dict[str, tuple]] = None):
        obs = {}
        meta = {}
        for name, stream in self._streams.items():
            image, ts = stream.read_rgb_with_timestamp()
            if target_sizes and name in target_sizes:
                image = center_crop_resize(image, target_sizes[name])
            obs[f"image_{name}"] = image
            meta[name] = {"timestamp": ts, "age_s": max(0.0, time.time() - ts)}
        return obs, meta

    def available_modalities(self):
        return list(self._streams.keys())

    def make_zero_snapshot(self, target_sizes: Dict[str, tuple]) -> Dict[str, np.ndarray]:
        obs = {}
        for name, size in target_sizes.items():
            h, w = int(size[0]), int(size[1])
            obs[f"image_{name}"] = make_zero_image(h, w)
        return obs

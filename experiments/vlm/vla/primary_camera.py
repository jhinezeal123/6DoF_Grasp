"""Camera primary (SPCA2650, /dev/video2) theo dung kieu cua m750.ros.camera.Camera.

Camera trong package chi phuc vu /myarm/cameras/cam01/image_raw, ma node
myarm_camera mo /dev/video0 - do la C925e gan tren co tay. Test 2 can nhin TOAN
CANH ban lam viec, tuc /dev/video2.

Module nay duck-type dung ba thu ma CameraRobotSource can: photo(), open(),
close(). Nho vay no cam thang vao adapter co san, khong phai sua pipeline.

QUAN TRONG - thu tu kenh mau: photo() tra RGB, GIONG m750.ros.camera.Camera
(khung ROS bgr8 duoc dao kenh o _photo_ros). Phia model tu doi sang BGR truoc
khi ma hoa JPEG. Doi nham o day thi do thanh xanh.
"""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np

# /dev/video2 = SPCA2650 = camera primary. Phai ep CAP_V4L2, khong thi OpenCV
# chon GStreamer va bao "Failed to open camera 'primary' on index 2".
DEFAULT_INDEX = 2
DEFAULT_WIDTH, DEFAULT_HEIGHT = 1280, 720


class PrimaryCamera:
    """Doc /dev/videoN bang OpenCV, tra khung RGB moi nhat."""

    def __init__(self, index: int = DEFAULT_INDEX,
                 width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT) -> None:
        self.index = int(index)
        self.width = int(width)
        self.height = int(height)
        self._cap: cv2.VideoCapture | None = None
        self._lock = threading.RLock()
        self._last_frame: np.ndarray | None = None
        self._last_stamp = 0.0

    def open(self) -> None:
        """Mo thiet bi. Goi nhieu lan vo hai - photo() cung tu goi khi can."""
        with self._lock:
            if self._cap is not None and self._cap.isOpened():
                return
            cap = cv2.VideoCapture(self.index, cv2.CAP_V4L2)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # dung lay anh cu trong buffer
            if not cap.isOpened():
                cap.release()
                raise RuntimeError("khong mo duoc camera primary index %d" % self.index)
            self._cap = cap

    def close(self) -> None:
        with self._lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None

    def photo(self) -> np.ndarray:
        """Khung RGB uint8 (H, W, 3). Tu mo thiet bi neu chua mo."""
        with self._lock:
            if self._cap is None:
                self.open()
            # Bo vai khung cu: camera chay 30 fps con vong lap servo chay ~0.07 fps,
            # nen buffer chac chan chua anh lich su.
            for _ in range(3):
                self._cap.grab()
            ok, frame = self._cap.read()
            if not ok or frame is None:
                if self._last_frame is not None:
                    return self._last_frame.copy()
                raise RuntimeError("khong doc duoc khung tu camera primary")
            rgb = np.ascontiguousarray(frame[:, :, ::-1])
            self._last_frame = rgb
            self._last_stamp = time.monotonic()
            return rgb

    @property
    def photo_age_s(self) -> float:
        """Tuoi cua khung gan nhat, giay. inf neu chua chup lan nao."""
        if self._last_stamp <= 0.0:
            return float("inf")
        return time.monotonic() - self._last_stamp

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)


__all__ = ["PrimaryCamera", "DEFAULT_INDEX", "DEFAULT_WIDTH", "DEFAULT_HEIGHT"]

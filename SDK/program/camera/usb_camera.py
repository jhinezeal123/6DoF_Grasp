"""
camera/usb_camera.py - Camera USB (V4L2) gan truc tiep tren Jetson. Chi de XEM.

Tach rieng khoi Camera (camera.py) co chu y: Camera lo camera ROS 2 hoac camera
trong scene MuJoCo, con lop nay lo thiet bi V4L. Khong lien quan gi toi robot.

Moi camera chay MOT THREAD RIENG de lay khung hinh va nen JPEG. Day khong phai
toi uu hoa som ma la rang buoc that: cv2.VideoCapture.read() CHAN khi camera bi
rut ra hoac cham, va neu goi no trong thread render thi ca hai khung nhin MuJoCo
dung theo. Nen JPEG luon o thread nay; thread render khong dung toi USB.
"""
from __future__ import annotations

import threading
import time
from typing import Optional

import cv2

from script.compressor import ImageCompressor

# Doi cong o DAY. (khoa, thiet bi) - khoa thanh duong dan /stream/cam_<khoa>.mjpg.
#
# video1/video3 KHONG dung duoc: do la node metadata cua UVC, mo len nhung khong
# tra frame nao.
USB_CAMERAS = (
    ("usb0", "/dev/video0"),   # Logitech Webcam C925e
    ("usb1", "/dev/video2"),   # SPCA2650 PC Camera
)


class USBCamera:
    """Mot camera USB: thread rieng lay khung hinh, cache JPEG moi nhat."""

    def __init__(self, device: str, width: int = 640, height: int = 480,
                 fps: float = 25.0, quality: int = 75) -> None:
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.quality = quality
        self._cap = None
        self._jpeg: Optional[bytes] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._loop, name="USBCam-" + self.device, daemon=True)
        self._thread.start()

    def _open(self) -> bool:
        cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            return False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        # Hang doi 1 khung: khong doc lai anh cu, luon lay khung moi nhat.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap = cap
        return True

    def _loop(self) -> None:
        while not self._stop.is_set():
            if self._cap is None and not self._open():
                # Rut ra hoac chua san sang: thu lai, khong lam chet thread.
                print(f"[USBCamera] Chua mo duoc {self.device}, thu lai sau 2s")
                self._stop.wait(2.0)
                continue

            t0 = time.time()
            ok, frame = self._cap.read()
            if not ok or frame is None:
                print(f"[USBCamera] Mat {self.device}, dong lai")
                self._close()
                self._stop.wait(0.5)
                continue

            # cv2 tra BGR; ImageCompressor doi RGB.
            jpeg = ImageCompressor.encode_jpeg(frame[:, :, ::-1], quality=self.quality)
            if jpeg:
                with self._lock:
                    self._jpeg = jpeg

            # read() da chan cho khung moi; chi ngu phan con lai cho du nhip.
            # Dung _stop.wait thay vi sleep de stop() khong phai cho het nhip.
            rest = 1.0 / max(float(self.fps), 1.0) - (time.time() - t0)
            if rest > 0:
                self._stop.wait(rest)

    @property
    def latest_jpeg(self) -> Optional[bytes]:
        """JPEG moi nhat, hoac None neu chua lay duoc khung nao."""
        with self._lock:
            return self._jpeg

    def _close(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._close()


__all__ = ["USBCamera", "USB_CAMERAS"]

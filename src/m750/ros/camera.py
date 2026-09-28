"""
ros/camera.py - Lop Camera: nguon anh ROS 2 hoac camera trong MuJoCo.

HAI NGUON, chon bang tham so duy nhat `mj_model`:
  - Camera()                     -> camera THAT qua ROS 2. Device co dinh
                                    (C925e, xem m750.ros.bridge.CAMERA_DEVICE), do
                                    node myarm_camera mo va phat len
                                    /myarm/cameras/cam01/image_raw.
  - Camera(m, d, "view_cam")     -> camera trong scene MuJoCo (chi de hien thi).

Ban cu tu mo cv2.VideoCapture: da bo. Node myarm_camera la chu duy nhat cua
thiet bi V4L, nen khong con chuyen hai tien trinh gianh nhau `/dev/video0`.
"""
from __future__ import annotations

import threading
from typing import Optional

import cv2
import mujoco
import numpy as np
from sensor_msgs.msg import Image

from m750.ros.bridge import CAMERA_DEVICE, TOPIC_CAMERA_IMAGE, get_bridge


class Camera:
    """Camera that (ROS 2) hoac camera trong scene MuJoCo."""

    def __init__(self, mj_model: Optional[mujoco.MjModel] = None,
                 mj_data: Optional[mujoco.MjData] = None,
                 camera_name: str = "wrist_cam",
                 width: int = 640, height: int = 480,
                 fov_y_deg: float = 70.0):
        """
        Args:
            mj_model, mj_data: chi truyen khi muon camera trong scene MuJoCo.
                Bo trong = camera that qua ROS 2 (device co dinh, khong tham so).
            camera_name: ten camera trong XML (vd 'wrist_cam', 'view_cam').
        """
        self.m = mj_model
        self.d = mj_data
        self.camera_name = camera_name
        self.is_simulation = mj_model is not None
        self.width = width
        self.height = height
        self.fov_y_deg = fov_y_deg
        self.device_path = CAMERA_DEVICE

        self._lock = threading.RLock()
        self._renderer = None
        self._ros_image = None
        self._last_frame: Optional[np.ndarray] = None
        self._mj_lock = threading.RLock()

        if self.is_simulation:
            self._init_simulation()
        else:
            self.cam_id = -1
            self._init_ros_camera()

    # ------------------------------------------------------------------ khoi tao
    def _init_simulation(self) -> None:
        self.cam_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name)
        # Doc fovy THAT tu model thay vi dung hang so (fovy=0 nghia la camera
        # dung 'sensorsize'/'focal').
        if self.cam_id >= 0 and 0.0 < float(self.m.cam_fovy[self.cam_id]) < 180.0:
            self.fov_y_deg = float(self.m.cam_fovy[self.cam_id])
        # KHONG tao Renderer o day: EGL context gan chat vao thread tao no, nen
        # renderer phai duoc tao o thread se render (xem _ensure_renderer).

    def _init_ros_camera(self) -> None:
        self._b = get_bridge()
        self._b.node.create_subscription(
            Image, TOPIC_CAMERA_IMAGE, self._on_image, 10
        )

    def _on_image(self, message) -> None:
        # Chi giu con tro message: giai ma o photo() de khong dot CPU 30 lan/giay.
        with self._lock:
            self._ros_image = message

    # ------------------------------------------------------------------ render sim
    def _ensure_renderer(self):
        """Tao renderer MuJoCo trong THREAD HIEN TAI (bat buoc voi EGL)."""
        if self._renderer is None and self.m is not None:
            try:
                self._renderer = mujoco.Renderer(self.m, height=self.height, width=self.width)
                self._apply_render_flags(self._renderer)
            except Exception as e:
                print(f"[Camera] Canh bao tao Renderer: {e}")

    def _apply_render_flags(self, renderer):
        """
        Tat shadow + reflection cho renderer.

        Shadow la nghen co that tren GPU tich hop: do tren Intel Iris Xe, mjr_render
        mat 1374 ms/frame khi bat shadow va 248 ms khi tat (nhanh 5,5 lan); chi phi
        khong phu thuoc do phan giai (64x64 ton ngang 640x480). Phai ap lai moi khi
        tao renderer vi flags thuoc ve tung renderer.
        """
        try:
            f = renderer._scene.flags
            f[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
            f[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        except Exception:
            pass  # API noi bo doi -> bo qua, khong lam chet render

    # ------------------------------------------------------------------ anh
    def photo(self) -> np.ndarray:
        """Frame moi nhat dang RGB uint8 (H, W, 3)."""
        if self.is_simulation:
            return self._photo_simulation()
        return self._photo_ros()

    def _photo_simulation(self) -> np.ndarray:
        with self._lock:
            self._ensure_renderer()
            if self._renderer is not None and self.d is not None:
                if self.cam_id >= 0:
                    self._renderer.update_scene(self.d, camera=self.cam_id)
                else:
                    self._renderer.update_scene(self.d)
                frame = self._renderer.render()
                self._last_frame = frame.copy()
                return frame
        return self._placeholder()

    def _photo_ros(self) -> np.ndarray:
        with self._lock:
            message = self._ros_image
        if message is None:
            return self._placeholder()

        frame = np.frombuffer(message.data, dtype=np.uint8)
        try:
            frame = frame.reshape(message.height, message.width, -1)
        except ValueError:
            return self._placeholder()
        if message.encoding in ("bgr8", "8UC3"):
            frame = frame[:, :, ::-1]
        elif message.encoding == "mono8":
            frame = np.repeat(frame, 3, axis=2)

        frame = np.ascontiguousarray(frame)
        with self._lock:
            self._last_frame = frame
        return frame

    def _placeholder(self) -> np.ndarray:
        with self._lock:
            if self._last_frame is not None:
                return self._last_frame.copy()
            placeholder = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            cv2.putText(placeholder, f"Camera: {self.camera_name}", (30, self.height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 128), 2)
            cv2.putText(placeholder, "chua co frame", (30, self.height // 2 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            self._last_frame = placeholder
            return placeholder


__all__ = ["Camera"]

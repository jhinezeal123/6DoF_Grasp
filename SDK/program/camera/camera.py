"""
camera/camera.py - Lop Camera: nguon anh ROS 2 hoac camera trong MuJoCo.

HAI NGUON, chon bang tham so duy nhat `mj_model`:
  - Camera()                     -> camera THAT qua ROS 2. Device co dinh
                                    (C925e, xem ros_bridge.CAMERA_DEVICE), do
                                    node myarm_camera mo va phat len
                                    /myarm/cameras/cam01/image_raw.
  - Camera(m, d, "view_cam")     -> camera trong scene MuJoCo (chi de hien thi).

Ban cu tu mo cv2.VideoCapture: da bo. Node myarm_camera la chu duy nhat cua
thiet bi V4L, nen khong con chuyen hai tien trinh gianh nhau `/dev/video0`.
"""
from __future__ import annotations

import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from typing import Any, Dict, Optional

import cv2
import mujoco
import numpy as np
from sensor_msgs.msg import Image

from program.ros_bridge import CAMERA_DEVICE, TOPIC_CAMERA_IMAGE, get_bridge
from script.compressor import ImageCompressor


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class CameraStreamHandler(BaseHTTPRequestHandler):
    """HTTP Handler phuc vu luong MJPEG video."""

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path not in ("/", "/stream.mjpg", "/video", "/stream"):
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.end_headers()

        cam_obj: "Camera" = self.server.camera_ref
        last_sent = None
        try:
            while cam_obj and cam_obj.is_streaming:
                # Chi DOC frame da render san (khong tu render o day):
                # renderer MuJoCo/EGL gan chat vao thread da tao no.
                with cam_obj._lock:
                    jpeg_bytes = cam_obj._stream_jpeg
                if jpeg_bytes and jpeg_bytes is not last_sent:
                    last_sent = jpeg_bytes
                    header = (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(jpeg_bytes)).encode() + b"\r\n\r\n"
                    )
                    self.wfile.write(header + jpeg_bytes + b"\r\n")
                time.sleep(1.0 / float(cam_obj.stream_fps))
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            pass


class Camera:
    """Camera that (ROS 2) hoac camera trong scene MuJoCo."""

    def __init__(self, mj_model: Optional[mujoco.MjModel] = None,
                 mj_data: Optional[mujoco.MjData] = None,
                 camera_name: str = "wrist_cam",
                 width: int = 640, height: int = 480,
                 fov_y_deg: float = 70.0,
                 external_render: bool = False):
        """
        Args:
            mj_model, mj_data: chi truyen khi muon camera trong scene MuJoCo.
                Bo trong = camera that qua ROS 2 (device co dinh, khong tham so).
            camera_name: ten camera trong XML (vd 'wrist_cam', 'view_cam').
            external_render: True neu caller tu goi update_stream_frame() trong
                vong lap cua no -> stream() khong tao thread render rieng, tranh
                hai thread cung ghi vao mjData.
        """
        self.m = mj_model
        self.d = mj_data
        self.camera_name = camera_name
        self.is_simulation = mj_model is not None
        self.width = width
        self.height = height
        self.fov_y_deg = fov_y_deg
        self.external_render = external_render
        self.device_path = CAMERA_DEVICE

        self._lock = threading.RLock()
        self._renderer = None
        self._ros_image = None
        self._last_frame: Optional[np.ndarray] = None
        self._mj_lock = threading.RLock()

        self.stream_server = None
        self.stream_thread = None
        self.is_streaming = False
        self.stream_fps = 25.0
        self.stream_port = None
        self._stream_jpeg = None
        self._render_thread = None

        if self.is_simulation:
            self._init_simulation()
        else:
            self.cam_id = -1
            self._init_ros_camera()

    # ------------------------------------------------------------------ khoi tao
    def _init_simulation(self) -> None:
        self.cam_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name)
        # Doc fovy THAT tu model thay vi dung hang so: neu lech, pixel_to_3d_base
        # se sai ti le. fovy=0 nghia la camera dung 'sensorsize'/'focal'.
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

    def reset_renderer(self):
        """
        Bo renderer de thread khac tao lai.

        Renderer MuJoCo/EGL gan chat vao thread da tao no. Neu photo() duoc goi
        lan dau o thread A roi sau do goi tu thread B, thread B se nhan frame
        RONG (khong loi, chi im lang) -> stream dung hinh.
        """
        with self._lock:
            if self._renderer is not None:
                try:
                    self._renderer.close()
                except Exception:
                    pass
            self._renderer = None

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

    # ------------------------------------------------------------------ hinh hoc
    @property
    def position(self) -> np.ndarray:
        """
        Toa do 3D camera [X, Y, Z] trong he base_link.

        Voi camera MuJoCo: doc tu model. Voi camera that: vi tri nam tren TF
        (wrist_camera_optical_frame) - chua doc TF trong giai doan nay, tra nan
        thay vi tra mot con so bia.
        """
        if not self.is_simulation:
            return np.full(3, np.nan, dtype=np.float64)
        with self._lock:
            mujoco.mj_forward(self.m, self.d)
            if self.cam_id >= 0:
                return self.d.cam_xpos[self.cam_id].copy()
            return np.array([0.25, -0.65, 0.45], dtype=np.float64)

    @property
    def rotation_matrix(self) -> np.ndarray:
        if not self.is_simulation:
            return np.full((3, 3), np.nan, dtype=np.float64)
        with self._lock:
            mujoco.mj_forward(self.m, self.d)
            if self.cam_id >= 0:
                return self.d.cam_xmat[self.cam_id].reshape(3, 3).copy()
            return np.eye(3, dtype=np.float64)

    @property
    def distribution(self) -> Dict[str, Any]:
        pos = self.position
        return {
            "position": pos,
            "matrix": self.rotation_matrix,
            "coordinates": (float(pos[0]), float(pos[1]), float(pos[2])),
            "camera_name": self.camera_name,
            "is_simulation": self.is_simulation,
            "resolution": (self.width, self.height),
            "topic": None if self.is_simulation else TOPIC_CAMERA_IMAGE,
            "device_path": None if self.is_simulation else self.device_path,
        }

    distribute = distribution

    def pixel_to_3d_base(self, u: float, v: float, depth_m: Optional[float] = None,
                         table_z: Optional[float] = None) -> np.ndarray:
        """
        Chieu nguoc diem anh (u, v) sang toa do 3D base_link. CHI dung cho camera
        MuJoCo: camera that can TF + ma tran noi tai tu camera_info (giai doan sau).

        Hai che do:
          - Co depth_m: dung do sau quang hoc doc truc nhin.
          - Khong co depth_m: ray-casting cat mat phang lam viec z = table_z.
            Chi co nghia khi camera NHIN XUONG mat ban. Voi wrist_cam (eye-in-hand
            nhin ngang ve phia truoc), tia nhin song song mat ban nen ray-casting
            KHONG dung duoc -> raise ValueError thay vi tra mot diem rac.

        Quy uoc truc (theo MuJoCo, KHAC OpenCV):
            cam_mat cot 0 = x_cam (sang phai trong anh)
            cam_mat cot 1 = y_cam (HUONG LEN trong anh)
            cam_mat cot 2 = z_cam, camera nhin theo -z_cam
        """
        if not self.is_simulation:
            raise ValueError("pixel_to_3d_base chi ho tro camera MuJoCo.")
        if table_z is None:
            from script.scene import table_z as _tz
            table_z = _tz(self.m)

        with self._lock:
            mujoco.mj_forward(self.m, self.d)
            if self.cam_id >= 0:
                cam_pos = self.d.cam_xpos[self.cam_id].copy()
                cam_mat = self.d.cam_xmat[self.cam_id].reshape(3, 3).copy()
            else:
                cam_pos = self.position.copy()
                cam_mat = self.rotation_matrix.copy()

        fy = self.height / (2.0 * np.tan(np.radians(self.fov_y_deg) / 2.0))
        fx = fy
        ray_cam = np.array([(u - self.width / 2.0) / fx,
                            (self.height / 2.0 - v) / fy, -1.0])
        ray_dir = cam_mat @ ray_cam

        if depth_m is not None and depth_m > 0:
            return cam_pos + depth_m * ray_dir

        if abs(ray_dir[2]) < 1e-6 * max(1.0, np.linalg.norm(ray_dir)):
            raise ValueError(
                "Khong the ray-cast xuong mat ban: tia nhin song song mat phang z=%.3f "
                "(ray_dir.z=%.2e). Camera nay nhin ngang, hay truyen depth_m." % (table_z, ray_dir[2])
            )
        t = (table_z - cam_pos[2]) / ray_dir[2]
        if t <= 0:
            raise ValueError(
                "Khong the ray-cast: mat phang z=%.3f nam SAU camera (t=%.3f). "
                "Camera dang nhin ra xa mat ban, hay truyen depth_m." % (table_z, t)
            )
        point = cam_pos + t * ray_dir
        if np.linalg.norm(point - cam_pos) > 10.0:
            raise ValueError(
                "Khong the ray-cast: tia cat mat phang z=%.3f tai diem cach camera %.1f m "
                "(tia gan song song mat phang). Hay truyen depth_m." % (table_z, np.linalg.norm(point - cam_pos))
            )
        return point

    # ------------------------------------------------------------------ stream
    def stream(self, port: int, fps: float = 25.0):
        """Stream frame lien tuc len network port chi dinh qua MJPEG."""
        if self.is_streaming:
            print(f"[Camera] Stream dang chay tai port {self.stream_port}.")
            return

        self.stream_port = port
        self.stream_fps = fps
        self.is_streaming = True

        # Thread render RIENG: renderer MuJoCo/EGL gan chat vao thread da tao no,
        # nen khong the goi photo() tu thread cua HTTP handler.
        def _render_loop():
            self._renderer = None
            while self.is_streaming:
                try:
                    frame = self.photo()
                    if frame is not None and frame.size > 0:
                        jpeg_bytes = ImageCompressor.encode_jpeg(frame, quality=75)
                        if jpeg_bytes:
                            with self._lock:
                                self._stream_jpeg = jpeg_bytes
                except Exception as e:
                    print(f"[Camera] Loi render stream: {e}")
                time.sleep(1.0 / max(float(self.stream_fps), 1.0))

        if not self.external_render:
            self._render_thread = threading.Thread(target=_render_loop, name="CameraRender", daemon=True)
            self._render_thread.start()

        def _run_server():
            try:
                self.stream_server = ThreadedHTTPServer(("0.0.0.0", port), CameraStreamHandler)
                self.stream_server.camera_ref = self
                print(f"[Camera] Bat dau MJPEG stream tai http://0.0.0.0:{port}/stream.mjpg (@ {fps:.1f} FPS)")
                self.stream_server.serve_forever()
            except Exception as e:
                print(f"[Camera] Loi server stream tai port {port}: {e}")
                self.is_streaming = False

        self.stream_thread = threading.Thread(target=_run_server, name=f"CameraStream-{port}", daemon=True)
        self.stream_thread.start()

    def update_stream_frame(self):
        """Render 1 frame va cap nhat cache cho stream (khi external_render=True)."""
        try:
            frame = self.photo()
            if frame is not None and frame.size > 0:
                jpeg_bytes = ImageCompressor.encode_jpeg(frame, quality=75)
                if jpeg_bytes:
                    with self._lock:
                        self._stream_jpeg = jpeg_bytes
                    return True
        except Exception as e:
            print(f"[Camera] Loi update_stream_frame: {e}")
        return False

    def stop_stream(self):
        self.is_streaming = False
        if self._render_thread is not None:
            self._render_thread.join(timeout=2.0)
            self._render_thread = None
        if self.stream_server is not None:
            try:
                self.stream_server.shutdown()
                self.stream_server.server_close()
            except Exception:
                pass
            self.stream_server = None
        print(f"[Camera] Da dung stream tai port {self.stream_port}")


__all__ = ["Camera"]

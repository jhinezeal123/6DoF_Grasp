"""
webui/web_control.py - Web UI Control Server cho robot myArm M750.
Cung cap:
  - He thong UI slider tuong ung voi ham change().
  - 2 mode hoat dong: 'control' (robot that) va 'simulate' (robot ao).
    + Chuyen sang simulate: giu nguyen pose hien tai.
    + Chuyen tu simulate sang control: ve lai pose cua robot that.
    + Trong mode simulate: hien thi option "Dua robot that ve pose nay" (goi real_robot.FollowTrajectory).
  - Tich hop Live Camera stream tu class Camera.
"""
import os
import time
import multiprocessing as mp
import threading
from typing import Optional
import numpy as np

from m750.ros import bridge as ros_bridge
from m750.ros.robot import Robot
from m750.ros.camera import Camera
from m750.ros.usb_camera import USB_CAMERAS, USBCamera
from m750.media import ImageCompressor
from m750.spec import model_dir
from m750.webui.fake_robot import FakeRobot
from m750.webui.teleop import render_html as render_teleop_page
from .http import ThreadedWebServer as ThreadedWebServer, WebControlHandler as WebControlHandler
from .streams import STREAM_KEYS as STREAM_KEYS
from .rendering import _mujoco_render_worker as _mujoco_render_worker
from .pages.control import render_html as render_control_page

# Duong dan stream -> khoa trong cache JPEG (xem get_stream_jpeg).
# Anh xa TUONG MINH thay vi suy ra tu danh tinh doi tuong camera: ban cu lam vay
# va khi them camera thu ba thi no bi gan nham nhan, tab third-person am tham
# hien anh cua USB camera.
# Camera USB: mot duong dan cho moi thiet bi trong USB_CAMERAS, de chi co MOT
# cho khai bao thiet bi (program/camera/usb_camera.py).








class WebControlApp:
    """
    Quan ly state he thong cho giao dien Web Control.
    Ket noi RealRobot, FakeRobot va cac Camera.
    """

    MODE_CONTROL = "control"      # Dieu khien truc tiep robot that
    MODE_SIMULATE = "simulate"    # Dieu khien thu nghiem tren robot ao
    MODE_OFFSET = "offset"        # Hieu chuan offset: slider CHI can robot ao

    # Teleop intentionally keeps orientation fixed while the keyboard only
    # changes XYZ.  With the SDK tool0 transform this makes the gripper point
    # vertically down in base_link.
    TELEOP_DEFAULT_QUAT = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    TELEOP_MIN_POSITION = np.array([-0.10, -0.60, 0.00], dtype=np.float64)
    TELEOP_MAX_POSITION = np.array([0.70, 0.60, 0.70], dtype=np.float64)
    TELEOP_MAX_DELTA_M = 0.05

    def __init__(self, real_robot: Robot, fake_robot: FakeRobot,
                 cam_sim: Optional[Camera] = None,
                 cam_real: Optional[Camera] = None,
                 port: int = 8080,
                 cam_third: Optional[Camera] = None):
        self.real_robot = real_robot
        self.fake_robot = fake_robot
        self.cam_sim = cam_sim
        self.cam_real = cam_real
        # Camera nhin toan canh (third-person) - goc nhin khac voi wrist_cam.
        # None thi tab third-person se bao khong co, khong lam hong cac tab khac.
        self.cam_third = cam_third
        self.port = port
        self.running = True
        self._lock = threading.RLock()
        # KHOA MUJOCO TOAN CUC: moi loi goi mj_forward/mj_step/render tren m/d phai
        # nam trong khoa nay, bat ke thread nao (render, HTTP handler). MuJoCo
        # KHONG thread-safe: 2 mj_forward chong len cung d lam hong nefc
        # ("mj_makeConstraint: nefc under-allocation") roi segfault cho ca process
        # (da do bang faulthandler: server chet im lang duoi tai browser).
        # Cac khoa chuyen mon (FakeRobot._lock, Robot._lock, Camera._lock) khong
        # du vi chung khac nhau -> van cho 2 thread vao mj_forward cung luc.
        self._mj_lock = threading.RLock()
        # THU TU KHOA DUY NHAT DUOC PHEP:  self._lock  ->  self._mj_lock.
        # Tuyet doi KHONG lam nguoc lai. Trong khoi `with self._mj_lock` khong duoc
        # goi bat cu ham nao lay self._lock: get_session_mode(), set_session_mode(),
        # active_control_object(), display_override_active().
        # Ly do: cac handler HTTP deu di chieu _lock -> _mj_lock, rieng vong render
        # truoc day goi no ben trong _mj_lock -> 2 chieu nguoc nhau
        # -> deadlock ABBA. Trieu chung: ~30 thread cung futex_wait, trang tinh van
        # tra 200 nhung /api/status treo VINH VIEN. Da xay ra 2 lan. Lan sua thu
        # nhat chi reorder mot cho trong /api/status nen benh tai phat.

        # Mode tach theo TUNG PHIEN: moi tab co mode rieng, khong dung chung.
        # Truoc day ui_mode la thuoc tinh cua app -> tab A bam Simulate thi tab B
        # cung nhay theo, rat nguy hiem vi B tuong dang xem robot that.
        self._sessions = {}           # sid -> {"mode": str, "last": float, "name": str}
        self._control_tabs = {}       # sid -> "joint" | "coordinate"
        # Per-session absolute TCP target used by keyboard teleop.  Deltas are
        # accumulated here because TCP feedback arrives slower than key repeats.
        self._teleop_targets = {}     # sid -> {"position": [x, y, z]}

        # Luu tru pose truoc khi vao mode simulate
        self._saved_real_pose = None

        # Thread dang chay FollowTrajectory (dua robot ao -> robot that).
        # Dung de chan bam trung nut va bao loi tu thread ra ngoai UI.
        self._traj_thread = None
        self._traj_error = None

        # Pose EE dang xem truoc tren ghost MuJoCo. Chi co mot FakeRobot dung
        # chung, nen luu mot target hien tai va khoa mirror_real() cho toi khi
        # user submit hoac thuc hien lenh dieu khien khac.
        self._ee_preview = None

        # Cache JPEG dung chung cho MOI tab (xem _serve_mjpeg). 1 thread render
        # duy nhat, thay vi moi ket noi tu render -> N tab khong con N lan render.
        self.stream_fps = 25.0
        self._jpeg_cache = {"sim": None, "real": None}
        self._jpeg_lock = threading.Lock()
        self._render_thread = None
        # MuJoCo/EGL lives in a separate process.  The parent keeps ownership
        # of FakeRobot for IK/status, while the worker owns its renderers.
        self._render_process = None
        self._render_cmd_send = None
        self._render_frame_recv = None
        self._render_frame_thread = None

        # Camera USB: chi de xem, khong lien quan robot. Moi camera tu chay thread
        # rieng (xem program/camera/usb_camera.py) nen vong render o duoi KHONG
        # dung toi chung - mot camera USB treo cung khong lam dung khung nhin MuJoCo.
        self.usb_cams = {}
        for key, device in USB_CAMERAS:
            cam = USBCamera(device)
            cam.start()
            self.usb_cams[key] = cam

    # ------------------------------------------------------------------
    # Mode theo tung phien
    # ------------------------------------------------------------------
    def get_session_mode(self, sid: str) -> str:
        """Mode cua 1 tab. Tab la khac nhau thi mode doc lap."""
        with self._lock:
            s = self._sessions.get(sid)
            return s["mode"] if s else self.MODE_CONTROL

    def set_session_mode(self, sid: str, mode: str) -> bool:
        with self._lock:
            mode_clean = (mode or "").lower().strip()
            if mode_clean not in (self.MODE_CONTROL, self.MODE_SIMULATE,
                                  self.MODE_OFFSET):
                return False
            self._sessions.setdefault(sid, {"mode": self.MODE_CONTROL, "last": 0.0,
                                            "name": self._default_name(sid)})
            self._sessions[sid]["mode"] = mode_clean
            return True

    def _default_name(self, sid: str) -> str:
        """Ten goi y cho nguoi dung khi chua tu dat ten."""
        return f"Nguoi-{sid[:4]}" if sid else "khach"

    def get_session_name(self, sid: str) -> str:
        with self._lock:
            s = self._sessions.get(sid)
            if not s:
                return self._default_name(sid)
            return s.get("name") or self._default_name(sid)

    def set_session_name(self, sid: str, name: str) -> str:
        with self._lock:
            self._sessions.setdefault(sid, {"mode": self.MODE_CONTROL, "last": 0.0})
            self._sessions[sid]["name"] = (name or "").strip()[:32] or self._default_name(sid)
            return self._sessions[sid]["name"]

    def get_control_tab(self, sid: str) -> str:
        with self._lock:
            return self._control_tabs.get(sid, "joint")

    def set_control_tab(self, sid: str, tab: str) -> bool:
        tab_clean = (tab or "").lower().strip()
        if tab_clean not in ("joint", "coordinate"):
            return False
        with self._lock:
            self._control_tabs[sid] = tab_clean
            # The ghost pose is shared. Leaving coordinate preview active while
            # switching back to joint sliders would make the renderer overwrite
            # the joint preview, so clear it at the tab boundary.
            if tab_clean == "joint":
                self._ee_preview = None
            return True

    def touch_session(self, sid: str):
        """Ghi nhan tab nay con hoat dong (de don phien cu)."""
        if not sid:
            return
        with self._lock:
            s = self._sessions.setdefault(sid, {"mode": self.MODE_CONTROL, "last": 0.0})
            s["last"] = time.time()
            s.setdefault("name", self._default_name(sid))
            # Don phien khong dung qua 1 ngay -> tranh phinh bo nho vo han.
            if len(self._sessions) > 64:
                old = sorted(self._sessions.items(), key=lambda kv: kv[1].get("last", 0))[:16]
                for k, _ in old:
                    if k != sid:
                        self._sessions.pop(k, None)
                        self._control_tabs.pop(k, None)
                        self._teleop_targets.pop(k, None)

    def get_stream_jpeg(self, key: str) -> Optional[bytes]:
        """Lay JPEG moi nhat tu cache (thread-safe). None neu chua render xong."""
        usb = self.usb_cams.get(key)
        if usb is not None:
            return usb.latest_jpeg
        with self._jpeg_lock:
            return self._jpeg_cache.get(key)

    def stream_available(self, key: str) -> bool:
        """Duong stream nay co camera that su khong (cam_real co the la None)."""
        if key in self.usb_cams:
            return True
        return {"sim": self.cam_sim, "third": self.cam_third,
                "real": self.cam_real}.get(key) is not None

    def render_teleop_html(self) -> bytes:
        """Render the standalone keyboard teleoperation page."""
        return render_teleop_page(USB_CAMERAS)

    @staticmethod
    def _finite_position(value):
        try:
            position = np.asarray(value, dtype=np.float64).reshape(3)
        except (TypeError, ValueError):
            return None
        return position if np.all(np.isfinite(position)) else None

    def teleop_status(self, sid: str = ""):
        """Return teleop target, measured TCP pose and safety state."""
        with self._lock:
            actual = self._finite_position(self.real_robot.tcp_pos)
            entry = self._teleop_targets.get(sid)
            if entry is not None:
                target = self._finite_position(entry.get("position"))
            else:
                target = actual.copy() if actual is not None else None
            return {
                "ok": target is not None,
                "position": None if target is None else target.tolist(),
                "quaternion": self.TELEOP_DEFAULT_QUAT.tolist(),
                "actual_position": None if actual is None else actual.tolist(),
                "safety_state": self.real_robot.safety_state,
                "is_armed": bool(self.real_robot.is_armed),
                "is_real_connected": bool(self.real_robot.is_real_connected),
            }

    def reset_teleop(self, sid: str = ""):
        """Forget the accumulated target; the next step starts at feedback TCP."""
        with self._lock:
            self._teleop_targets.pop(sid, None)
            status = self.teleop_status(sid)
            status["message"] = "Da reset target teleop theo pose robot"
            return status

    def teleop_step(self, body, sid: str = ""):
        """Apply one keyboard Cartesian delta with fixed downward orientation."""
        if not isinstance(body, dict):
            return {"ok": False, "message": "body teleop phai la JSON object"}
        try:
            delta = np.asarray([body.get("dx"), body.get("dy"), body.get("dz")],
                               dtype=np.float64)
        except (TypeError, ValueError):
            return {"ok": False, "message": "dx/dy/dz phai la so"}
        if delta.shape != (3,) or not np.all(np.isfinite(delta)):
            return {"ok": False, "message": "dx/dy/dz phai huu han"}
        if np.any(np.abs(delta) > self.TELEOP_MAX_DELTA_M):
            return {"ok": False,
                    "message": "Moi buoc teleop khong duoc vuot qua 50 mm"}
        if not np.any(np.abs(delta) > 1e-12):
            return {"ok": False, "message": "Buoc teleop phai khac 0"}

        with self._lock:
            entry = self._teleop_targets.get(sid)
            if entry is None:
                base = self._finite_position(self.real_robot.tcp_pos)
                if base is None:
                    return {"ok": False,
                            "message": "Chua co TCP feedback moi de bat dau teleop"}
            else:
                base = self._finite_position(entry.get("position"))
                if base is None:
                    self._teleop_targets.pop(sid, None)
                    return {"ok": False,
                            "message": "Target teleop khong hop le; hay reset lai"}

            target = base + delta
            if np.any(target < self.TELEOP_MIN_POSITION) or np.any(target > self.TELEOP_MAX_POSITION):
                return {
                    "ok": False,
                    "message": "Target vuot mien an toan XYZ cua teleop",
                    "position": base.tolist(),
                }

            result = self._move_ee_pose_locked(target, self.TELEOP_DEFAULT_QUAT.copy())
            if result.get("ok"):
                self._teleop_targets[sid] = {"position": target.tolist()}
                result["delta"] = delta.tolist()
                result["position"] = target.tolist()
                result["quaternion"] = self.TELEOP_DEFAULT_QUAT.tolist()
            return result

    def _start_render_thread(self):
        """
        Start the state synchronizer and the isolated MuJoCo render worker.

        The old implementation rendered in this process.  On headless EGL that
        call can block while holding the Python interpreter lock, starving the
        RosBridge callback thread.  The worker process below owns the MuJoCo
        renderers; this process only mirrors/snapshots qpos and serves JPEGs.
        """
        if self._render_thread is not None:
            return
        sim_cams = []
        for key, cam in (("sim", self.cam_sim), ("third", self.cam_third)):
            if cam is None or not getattr(cam, "is_simulation", False):
                continue
            sim_cams.append((key, cam.camera_name, cam.width, cam.height))

        # No MuJoCo camera is a valid configuration (for example a small API
        # embedding that only wants robot state).  Keep the synchronizer alive
        # for the parent FakeRobot, but do not spawn an idle worker.
        cmd_recv = cmd_send = frame_recv = frame_send = None
        process = None
        if sim_cams:
            # ``spawn`` is deliberate: fork would copy rclpy's DDS thread and
            # MuJoCo state into the child, which is unsafe for both libraries.
            ctx = mp.get_context("spawn")
            cmd_recv, cmd_send = ctx.Pipe(duplex=False)
            frame_recv, frame_send = ctx.Pipe(duplex=False)
            scene_path = getattr(self.fake_robot, "scene_path", None)
            if not scene_path:
                raise RuntimeError("FakeRobot khong luu scene_path cho render worker")
            process = ctx.Process(
                target=_mujoco_render_worker,
                args=(scene_path, sim_cams, self.stream_fps,
                      cmd_recv, frame_send),
                name="MuJoCoRenderWorker",
                daemon=True,
            )
            process.start()
            # The child owns these endpoints; parent only retains send/recv.
            cmd_recv.close()
            frame_send.close()

        self._render_process = process
        self._render_cmd_send = cmd_send
        self._render_frame_recv = frame_recv

        def receive_frames():
            while self.running or (process is not None and process.is_alive()):
                if frame_recv is None:
                    return
                try:
                    if not frame_recv.poll(0.2):
                        continue
                    key, jpeg = frame_recv.recv()
                except (EOFError, OSError):
                    return
                if jpeg:
                    with self._jpeg_lock:
                        self._jpeg_cache[key] = jpeg

        if frame_recv is not None:
            self._render_frame_thread = threading.Thread(
                target=receive_frames, name="WebRenderFrames", daemon=True)
            self._render_frame_thread.start()

        def sync_loop():
            last_mirror = 0.0
            last_sent = None
            last_preview = None
            while self.running:
                now = time.time()
                do_mirror = (now - last_mirror) >= 0.2
                # Keep lock order _lock -> _mj_lock.  The mode check must happen
                # before taking _mj_lock to avoid the old ABBA deadlock.
                display_override = self.display_override_active()
                try:
                    with self._mj_lock:
                        if do_mirror:
                            last_mirror = now
                            if not display_override:
                                self.fake_robot.mirror_real()
                        qpos = self.fake_robot.mj_qpos
                except Exception as exc:
                    print(f"[WebControl] Loi dong bo MuJoCo: {exc}")
                    qpos = None

                if qpos is not None and cmd_send is not None:
                    try:
                        with self._lock:
                            preview_active = self._ee_preview is not None
                        # Avoid filling the pipe with identical snapshots while
                        # still sending every actual pose change promptly.
                        if (last_sent is None or not np.array_equal(qpos, last_sent)
                                or preview_active != last_preview):
                            cmd_send.send((qpos, preview_active))
                            last_sent = qpos.copy()
                            last_preview = preview_active
                    except (BrokenPipeError, EOFError, OSError):
                        break

                # A ROS camera is not EGL backed, so it can remain in the parent
                # loop.  The default web app has no cam_real, but this preserves
                # the optional real camera stream API.
                if self.cam_real is not None and not getattr(self.cam_real, "is_simulation", False):
                    try:
                        frame = self.cam_real.photo()
                        if frame is not None and frame.size > 0:
                            jpeg = ImageCompressor.encode_jpeg(frame, quality=75)
                            if jpeg:
                                with self._jpeg_lock:
                                    self._jpeg_cache["real"] = jpeg
                    except Exception as exc:
                        print(f"[WebControl] Loi render 'real': {exc}")
                time.sleep(1.0 / max(self.stream_fps, 1.0))

        # Retain the historical attribute name for diagnostics/tests.  This is
        # now only a lightweight synchronizer; MuJoCo rendering is in process.
        self._render_thread = threading.Thread(
            target=sync_loop, name="WebRenderSync", daemon=True)
        self._render_thread.start()

    def _stop_render_worker(self):
        """Stop the isolated renderer and release its IPC endpoints."""
        sender = self._render_cmd_send
        process = self._render_process
        if sender is not None:
            try:
                sender.send(None)
            except (BrokenPipeError, EOFError, OSError):
                pass
            try:
                sender.close()
            except Exception:
                pass
            self._render_cmd_send = None

        if process is not None:
            process.join(timeout=3.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)
        self._render_process = None

        receiver = self._render_frame_recv
        if receiver is not None:
            try:
                receiver.close()
            except Exception:
                pass
            self._render_frame_recv = None

    def active_control_object(self, sid: str = ""):
        """Object ma tab nay dang dieu khien truc tiep (theo mode cua CHINH tab do)."""
        if self.get_session_mode(sid) == self.MODE_SIMULATE:
            return self.fake_robot
        return self.real_robot

    def display_override_active(self) -> bool:
        """Co tab nao dang keo con robot ao bang tay khong (offset hoac simulate).

        O HAI che do nay slider chi ghi vao m/d cua MuJoCo, nen vong render phai
        NGUNG keo pose that vao sim: mirror_real() chay 5 Hz se ghi de dung cai
        pose nguoi dung dang can, keo slider xong la no bat tro lai.

        Chi co MOT con robot ao dung chung cho moi tab, nen hien dien cua bat ky
        tab nao cung khoa mirror cho TAT CA - cung mot su nhuong bo nhu ban cu.
        """
        with self._lock:
            return (self._ee_preview is not None or
                    any(s.get("mode") in (self.MODE_OFFSET, self.MODE_SIMULATE)
                        for s in self._sessions.values()))

    @staticmethod
    def _parse_ee_pose(body):
        """Validate a web pose and normalize its ROS quaternion."""
        if not isinstance(body, dict):
            raise ValueError("body phai la JSON object")
        position = body.get("position")
        quaternion = body.get("quaternion")
        # Accept flat fields as well, which makes the endpoint convenient for
        # curl/scripts in addition to the web form.
        if position is None:
            position = [body.get(k) for k in ("x", "y", "z")]
        if quaternion is None:
            quaternion = [body.get(k) for k in ("qx", "qy", "qz", "qw")]
        try:
            p = np.asarray(position, dtype=np.float64).reshape(3)
            q = np.asarray(quaternion, dtype=np.float64).reshape(4)
        except (TypeError, ValueError) as exc:
            raise ValueError("position can 3 so va quaternion can 4 so") from exc
        if not np.all(np.isfinite(p)) or not np.all(np.isfinite(q)):
            raise ValueError("position/quaternion phai gom so huu han")
        q_norm = float(np.linalg.norm(q))
        if q_norm < 1e-9:
            raise ValueError("quaternion khong duoc la vector 0")
        q = q / q_norm
        return p, q

    def preview_ee_pose(self, body, sid: str = ""):
        """Solve display-only MuJoCo IK for the requested tool0 pose."""
        del sid  # The ghost model is shared by all browser tabs.
        try:
            position, quaternion = self._parse_ee_pose(body)
        except ValueError as exc:
            return {"ok": False, "message": str(exc)}
        with self._lock:
            try:
                with self._mj_lock:
                    result = self.fake_robot.preview_tcp_pose(position, quaternion)
            except Exception as exc:
                return {"ok": False, "message": str(exc)}
            if result.get("ok"):
                self._ee_preview = {
                    "position": position.tolist(),
                    "quaternion": quaternion.tolist(),
                    "actual_position": result.get("actual_position"),
                    "position_error_m": result.get("position_error_m"),
                    "orientation_error_rad": result.get("orientation_error_rad"),
                    "iterations": result.get("iterations"),
                }
            return {
                **result,
                "position": position.tolist(),
                "quaternion": quaternion.tolist(),
            }

    def set_ee_absolute(self, body, sid: str = ""):
        """Apply a coordinate slider target according to the active UI mode."""
        mode = self.get_session_mode(sid)
        if mode == self.MODE_SIMULATE:
            return self.preview_ee_pose(body, sid)
        if mode == self.MODE_CONTROL:
            return self.move_ee(body, sid)
        return {"ok": False,
                "message": "Che do OFFSET chi ho tro dieu khien khop"}

    def _move_ee_pose_locked(self, position, quaternion):
        """Publish a validated pose; caller must hold ``self._lock``."""
        if not self.real_robot.is_armed:
            return {"ok": False,
                    "message": "Robot chua armed; bam KHÔI PHỤC truoc khi Submit"}
        if not self.real_robot.is_real_connected:
            return {"ok": False,
                    "message": "Robot khong co feedback moi; kiem tra nguon va cap USB"}
        if self._traj_thread is not None and self._traj_thread.is_alive():
            return {"ok": False, "message": "Robot dang chay quy dao khac"}
        try:
            ok = bool(self.real_robot.set_tcp_pose(position, quaternion))
        except Exception as exc:
            return {"ok": False, "message": f"set_tcp_pose loi: {exc}"}
        if ok:
            # Return the display to the measured robot after submit. The
            # third-person stream will then follow the real feedback again.
            self._ee_preview = None
        return {
            "ok": ok,
            "position": position.tolist(),
            "quaternion": quaternion.tolist(),
            "message": "Da gui set_tcp_pose" if ok else "Robot tu choi set_tcp_pose",
        }

    def move_ee(self, body, sid: str = ""):
        """Publish a validated absolute TCP pose to the real robot."""
        try:
            position, quaternion = self._parse_ee_pose(body)
        except ValueError as exc:
            return {"ok": False, "message": str(exc)}
        with self._lock:
            self._teleop_targets.pop(sid, None)
            return self._move_ee_pose_locked(position, quaternion)

    def stop(self) -> bool:
        """Dung khan cap robot that (dung ngay, khong chay not quy dao)."""
        with self._lock:
            try:
                self.real_robot.stop()
                return True
            except Exception as e:
                print(f"[WebControl] stop() loi: {e}")
                return False

    def power_off(self) -> bool:
        """
        Ngat mo-men roi cat dien robot that.

        Thu tu quan trong: release_all_servos truoc de servo thoi giu vi tri.
        Chi power_off khong thi servo van giu vat va tiep tuc nong (da do: 70 do
        chi giam 2 do trong 105 giay).

        DA KIEM CHUNG TREN MAY THAT: release_all_servos() lam tay TUT THAT (do duoc
        217 do qua 4 khop, roi tu do roi xuong va tua vao foam). LUU Y: sau khi
        release, is_all_servo_enable() VAN tra ve 1 — ham nay bao trang thai KET NOI
        BUS servo, KHONG phai mo-men. Dung lay no lam bang chung release that bai.
        """
        with self._lock:
            # Robot moi: power_off() di qua service /myarm/robot/power_off cua
            # driver. Driver tu nha mo-men roi cat dien va xac nhan lai.
            ok = self.real_robot.power_off()
            print("[WebControl] Da yeu cau tat nguon robot." if ok
                  else "[WebControl] Tat nguon that bai.")
            return bool(ok)

    def power_on(self) -> bool:
        """
        Cap dien lai cho robot that.

        KHONG tu di chuyen: sau khi tat nguon servo mat mo-men nen tay co the da
        tut xuong. Chi cap dien, de nguoi dung tu keo slider dua ve pose mong muon.

        Di qua service /myarm/robot/power_on cua driver; driver xac nhan lai trang
        thai truoc khi tra ve.
        """
        with self._lock:
            ok = self.real_robot.power_on()
            print("[WebControl] Da yeu cau bat nguon robot." if ok
                  else "[WebControl] Bat nguon that bai.")
            return bool(ok)

    def rearm(self) -> bool:
        """
        Khoi phuc sau DUNG KHAN: xoa fault_latched de executor nhan quy dao lai.

        Driver chot fault khi bam stop va tu choi MOI quy dao sau do. Khong co
        buoc nay thi nut DUNG KHAN bien giao dien thanh vo dung: robot im lang mai
        va chi khoi dong lai stack moi go duoc.
        """
        with self._lock:
            ok = self.real_robot.rearm()
            print("[WebControl] Da khoi phuc (rearm)." if ok
                  else "[WebControl] Khoi phuc that bai.")
            return bool(ok)

    def set_ui_mode(self, sid: str, mode: str) -> bool:
        """
        Chuyen doi che do dieu khien CHO RIENG TAB NAY:
        - Khi sang 'simulate': GIU NGUYEN pose hien tai (fake_robot sync tu ROS).
        - Khi tu 'simulate' ve 'control': VE LAI pose cua robot that.

        LUU Y: fake_robot la TAI NGUYEN DUNG CHUNG (chi co 1 robot ao), nen 2 tab
        cung o mode simulate se dung chung 1 con robot ao. Mode thi doc lap, nhung
        pose thi khong. Xem get_session_mode().

        KHONG con mode independent/synchronized trong FakeRobot: backend that hay ao
        la viec cua launch profile (services.yaml), khong phai cua UI. O ca hai che
        do, FakeRobot deu sync() tu /myarm/state/joint_state.
        """
        with self._lock:
            mode_clean = mode.lower().strip()
            self._ee_preview = None
            self._teleop_targets.pop(sid, None)
            if mode_clean == self.MODE_SIMULATE:
                self.set_session_mode(sid, self.MODE_SIMULATE)
                with self._mj_lock:
                    self.fake_robot.sync()
                print("[WebControl] Mode chuyen sang SIMULATE: robot ao sync tu ROS.")
                return True
            elif mode_clean == self.MODE_CONTROL:
                self.set_session_mode(sid, self.MODE_CONTROL)
                with self._mj_lock:
                    self.fake_robot.sync()
                print("[WebControl] Mode chuyen sang CONTROL: quay ve pose cua robot that.")
                return True
            elif mode_clean == self.MODE_OFFSET:
                # Vao che do chinh offset: bat dau TU POSE THAT de nguoi dung chi
                # phai chinh phan chenh lech, roi bam Luu.
                self.set_session_mode(sid, self.MODE_OFFSET)
                with self._mj_lock:
                    self.fake_robot.sync()
                print("[WebControl] Mode OFFSET: slider chi can con robot ao, "
                      "KHONG gui lenh xuong tay that. Bam Luu de ghi offset.")
                return True
            return False

    def handle_change(self, param, delta: float, sid: str = "") -> bool:
        """Goi ham change tren active_control_object cua tab nay.

        Rieng gripper: ca slider lan nut bam cua UI deu dung PHAN TRAM (0..100),
        nen phai doi sang met truoc khi goi Robot.change() - ham do nhan MET.
        """
        with self._lock:
            self._ee_preview = None
            self._teleop_targets.pop(sid, None)
            target = self.active_control_object(sid)
            if isinstance(param, str) and param.lower() in ("gripper", "grip"):
                # % -> met, dung MAX_GRIPPER_M cua Robot (mot nguon su that).
                delta_m = float(delta) / 100.0 * target.MAX_GRIPPER_M
                with self._mj_lock:
                    return target.change("gripper", delta_m)
            with self._mj_lock:   # fake.change() mj_step/mj_forward o simulate
                return target.change(param, delta)

    def handle_set_absolute(self, param, target_val: float, sid: str = "") -> bool:
        """Tinh toan delta va goi ham change() de dat gia tri tuyet doi."""
        with self._lock:
            self._ee_preview = None
            self._teleop_targets.pop(sid, None)
            # Hai che do nay slider chi duoc ghi vao HINH VE MuJoCo:
            #   OFFSET   - can robot ao cho khop thuc te roi ghi lai offset
            #   SIMULATE - dung pose ao, roi bam "dua robot that ve pose nay"
            # Ca hai deu KHONG duoc publish /myarm/command/joint_goal. Thieu chan
            # nay thi keo slider o che do Simulate lam TAY THAT chay ngay trong luc
            # nguoi dung tuong chi dang chinh hinh (va mirror_real 5 Hz se keo hinh
            # ve cho cu, nen nhin nhu slider bi bat tro).
            mode = self.get_session_mode(sid)
            display_only = mode in (self.MODE_OFFSET, self.MODE_SIMULATE)
            target = self.active_control_object(sid)
            if isinstance(param, str) and param.lower() in ("gripper", "grip"):
                if display_only:
                    # Kep khong co trang thai rieng trong sim (qpos ngon tay chua bao
                    # gio duoc ghi), nen khong the hien thi -> tra False cho UI biet.
                    return False
                # target_val la PHAN TRAM 0..100 (UI gui %), khong phai met.
                with self._mj_lock:
                    return target.set_gripper_pct(float(target_val))
            else:
                # Joint.
                # LUU Y: JS gui param duoi dang CHUOI ("0".."5") vi sinh ra tu vong
                # lap. Truoc day chi nhan isinstance(param, int) -> moi chuoi deu
                # roi vao j_idx = 0, nen KE0 SLIDER NAO CUNG CHI CHAY KHOP 1.
                s = str(param).lower().strip()
                if s.startswith("j"):
                    s = s[1:]          # "j3" -> khop 3 (danh so tu 1)
                    try:
                        j_idx = int(s) - 1
                    except ValueError:
                        return False
                else:
                    try:
                        j_idx = int(s)  # "3" -> khop 3 (danh so tu 0)
                    except ValueError:
                        return False
                if not (0 <= j_idx < 6):
                    print(f"[WebControl] chi so khop ngoai pham vi: {param!r} -> {j_idx}")
                    return False
                if display_only:
                    # Chi ghi vao m/d cua MuJoCo. Khong publish, khong cho executor.
                    with self._mj_lock:
                        q = self.fake_robot.mj_qpos[:6].copy()
                        q[j_idx] = float(target_val)
                        self.fake_robot.set_display(q)
                    return True
                # Goi set_joint() TRUC TIEP thay vi doi ra delta roi goi change().
                # Slider gui gia tri TUYET DOI; doi sang delta chi de change()
                # cong nguoc lai la vong tron de sai:
                #   - neu change() lay goc tu encoder con ctrl[] lai la gia tri cu
                #     -> delta bi cong hai lan (da gap: dat -44.6 lai thanh -89.2)
                #   - neu _target_rad lech encoder thi delta sai hoan toan, va
                #     ham van tra True -> slider quay ve gia tri cu ngay.
                with self._mj_lock:
                    return target.set_joint(j_idx, target_val)

    def apply_fake_pose_to_real(self, sid: str = "", control_tab: str = None):
        """Apply the pose currently shown in Simulate to the real robot.

        The joint tab keeps its existing FollowTrajectory path.  The coordinate
        tab applies the stored TCP target through set_tcp_pose(), so the top
        ``Dua robot that ve pose nay`` button is the single submit action for
        both control styles.
        """
        with self._lock:
            if self.get_session_mode(sid) != self.MODE_SIMULATE:
                return {"ok": False, "message": "Chi Submit duoc trong che do SIMULATE"}
            tab = (control_tab or self.get_control_tab(sid)).lower().strip()

            if tab == "coordinate":
                if self._ee_preview is None:
                    # Selecting the coordinate tab and pressing Submit without
                    # touching a slider should still apply the pose currently
                    # shown by the ghost model.
                    with self._mj_lock:
                        position, quaternion = self.fake_robot.tcp_pose()
                    position = np.asarray(position, dtype=np.float64)
                    quaternion = np.asarray(quaternion, dtype=np.float64)
                else:
                    position = np.asarray(self._ee_preview["position"], dtype=np.float64)
                    quaternion = np.asarray(self._ee_preview["quaternion"], dtype=np.float64)
                result = self._move_ee_pose_locked(position, quaternion)
                if result.get("ok"):
                    result["message"] = "Da gui set_tcp_pose tu pose EE mo phong"
                return result

            if tab != "joint":
                return {"ok": False, "message": "Tab dieu khien khong hop le"}
            self._ee_preview = None
            # Chan bam trung: FollowTrajectory chay o thread rieng va mat vai giay.
            # Bam 3 lan nhanh -> 3 thread cung dieu khien 1 tay -> lenh tron lan
            # (da do duoc: 3 thread cung chay). Doi xong moi cho bam tiep.
            if self._traj_thread is not None and self._traj_thread.is_alive():
                print("[WebControl] Dang di chuyen robot that, bo qua lenh trung.")
                return {"ok": False, "message": "Robot dang di chuyen"}
            with self._mj_lock:
                # mj_qpos la pose MuJoCo DANG VE, tuc "sim" = real - offset.
                target_sim = self.fake_robot.mj_qpos[:6].copy()
                target_grip = self.fake_robot.gripper
            # Muon tay that ve dung cho dang NHIN THAY thi phai cong offset tra lai
            # (raw_deg chinh la chieu nguoc cua model_rad ma sync() dung).
            target_qpos = np.radians(ros_bridge.raw_deg(target_sim))
            print(f"[WebControl] Thuc thi FollowTrajectory dua robot that ve pose mo phong: {np.degrees(target_qpos)}")

            # Tao chuoi waypoint ket hop 6 khop va gripper
            traj_data = np.concatenate([target_qpos, [target_grip]])

            # Chay trong thread rieng de khong block HTTP request
            def _async_exec():
                try:
                    self.real_robot.FollowTrajectory(traj_data, max_step_rad=0.08)
                    print("[WebControl] Hoan tat di chuyen robot that den pose mo phong!")
                except Exception as e:
                    # Khong bat loi thi thread chet im lang, con UI thi da bao
                    # "thanh cong" tu truoc -> nguoi dung tuong robot da di chuyen.
                    print(f"[WebControl] LOI FollowTrajectory: {e}")
                    self._traj_error = str(e)

            self._traj_error = None
            th = threading.Thread(target=_async_exec, name="FollowTrajectory-Worker", daemon=True)
            th.start()
            self._traj_thread = th
            return {"ok": True, "message": "Da bat dau dua robot that ve pose mo phong"}

    def apply_preset(self, name: str, sid: str = "") -> bool:
        """Dat pose preset."""
        presets = {
            "home": ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.0),
            # Grip tinh bang MET do mo TONG (0..0.08), khong phai toa do mot ngon.
            "pre_grasp": ([0.0, -0.3386, 0.8755, -0.0091, 0.7515, 0.0075], 0.069),
            "rest": ([0.0, 1.0472, 0.0, 0.0, -1.0472, 0.0], 0.0),
            # Pose trong anh nguoi dung chup (tay duoi thang truoc mat, gripper
            # chuc thang xuong tren vat). Do bang RAW encoder luc chup:
            #   [-10.1, 96.08, -100.65, 2.1, 92.46, 2.54]  (test1/verify_pose.py)
            # model = raw - OFFSETS_DEG, vi duong ghi la raw = model + OFFSETS_DEG.
            "photo": ([0.0, 1.1781, -1.6272, 0.0, 1.9836, 0.0], 0.0),
        }
        if name not in presets:
            return False
        q_rad, grip = presets[name]
        with self._lock:
            self._teleop_targets.pop(sid, None)
        target = self.active_control_object(sid)
        # Gui CA 6 khop trong MOT lenh (set_pose), cho ca robot that lan fake.
        # Voi 6 lan set_joint noi tiep, moi lan dong bo lai tu encoder nen khop sau
        # THU khop truoc ve gia tri encoder -> preset chi con dung khop cuoi cung
        # (da do). FakeRobot ke thua set_pose tu Robot, cung di qua joint_goal.
        with self._lock:
            self._ee_preview = None
            with self._mj_lock:
                return target.set_pose(q_rad, grip)

    def reset_scene(self) -> bool:
        """Reset scene mo phong MuJoCo ve trang thai mac dinh."""
        with self._lock:
            self._ee_preview = None
            self._teleop_targets.clear()
            if hasattr(self.fake_robot, "reset_scene"):
                with self._mj_lock:
                    return self.fake_robot.reset_scene()
            return False

    def render_html(self) -> bytes:
        """Trả đúng trang HTML hiện tại; logic template nằm trong pages/."""
        return render_control_page(self.real_robot, USB_CAMERAS)

    def run(self):
        """Khoi dong HTTP server phuc vu Web Control."""
        # Render truoc 1 lan de tab dau tien co anh ngay, khong cho trong.
        self._start_render_thread()
        server_address = ("0.0.0.0", self.port)
        httpd = ThreadedWebServer(server_address, WebControlHandler)
        httpd.app_context = self
        print(f"[WebControl] Giao dien Web Control da khoi dong tai: http://localhost:{self.port}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("[WebControl] Dung server Web Control.")
        finally:
            self.running = False
            httpd.server_close()
            self._stop_render_worker()
            for cam in self.usb_cams.values():
                cam.stop()


def start_web_server(real_robot: Robot, fake_robot: FakeRobot,
                     cam_sim: Optional[Camera] = None,
                     cam_real: Optional[Camera] = None,
                     port: int = 8080,
                     cam_third: Optional[Camera] = None):
    """Ham tien ich khoi chay web control server."""
    app = WebControlApp(real_robot, fake_robot, cam_sim, cam_real, port, cam_third)
    app.run()


def main() -> None:
    """Khoi chay web control voi moi truong mac dinh.

    Chay: python -m m750.webui.web_control  (hoac entry point m750-web)
    Yeu cau stack ROS 2 dang chay (xem run_web.sh): Robot/FakeRobot/Camera
    deu la mat na ROS, khong tu mo serial hay thiet bi camera nua.
    """
    print("[web_control.py] Khoi dong moi truong mac dinh...")
    # Scene trong package: m750/model/scene_vla.xml
    SCENE_PATH = os.path.join(str(model_dir()), "scene_vla.xml")

    robot = Robot()
    fake = FakeRobot(SCENE_PATH)
    # Camera PHAI dung CHUNG m/d voi FakeRobot.
    #
    # FakeRobot tu tao m/d rieng, va vong render goi sync() ghi qpos do duoc tu ROS
    # vao fake.d. Truoc day cho nay tu tao them mot m/d nua roi dua cho camera, nen
    # camera render mot scene chi duoc mj_forward DUNG MOT LAN luc khoi dong: anh
    # dung yen o pose mac dinh (qpos = 0) trong khi sim_deg van bao dung pose that.
    cam = Camera(fake.m, fake.d, camera_name="wrist_cam")
    cam_third = Camera(fake.m, fake.d, camera_name="view_cam")
    start_web_server(robot, fake, cam_sim=cam, port=8080, cam_third=cam_third)


if __name__ == "__main__":
    main()

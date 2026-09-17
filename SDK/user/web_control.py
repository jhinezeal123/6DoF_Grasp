"""
web_control.py - Web UI Control Server cho robot myArm M750.
Cung cap:
  - Lay distribute tu object control de render truc quan tren web port.
  - He thong UI slider tuong ung voi ham change().
  - 2 mode hoat dong: 'control' (robot that) va 'simulate' (robot ao).
    + Chuyen sang simulate: giu nguyen pose hien tai.
    + Chuyen tu simulate sang control: ve lai pose cua robot that.
    + Trong mode simulate: hien thi option "Dua robot that ve pose nay" (goi real_robot.FollowTrajectory).
  - Tich hop Live Camera stream tu class Camera.
"""
import os
import sys
import time
import json
import multiprocessing as mp
import threading
from urllib.parse import urlparse
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from typing import Optional
import numpy as np

from program import ros_bridge
from program.robot.robot import Robot
from program.camera.camera import Camera
from program.camera.usb_camera import USB_CAMERAS, USBCamera
from script.compressor import ImageCompressor
from user.fake_robot.fake_robot import FakeRobot
from user.teleop import render_html as render_teleop_page

# Duong dan stream -> khoa trong cache JPEG (xem get_stream_jpeg).
# Anh xa TUONG MINH thay vi suy ra tu danh tinh doi tuong camera: ban cu lam vay
# va khi them camera thu ba thi no bi gan nham nhan, tab third-person am tham
# hien anh cua USB camera.
STREAM_KEYS = {
    "/stream/cam_sim.mjpg": "sim",
    "/stream/cam_third.mjpg": "third",
    "/stream/cam_real.mjpg": "real",
}
# Camera USB: mot duong dan cho moi thiet bi trong USB_CAMERAS, de chi co MOT
# cho khai bao thiet bi (program/camera/usb_camera.py).
STREAM_KEYS.update({"/stream/cam_%s.mjpg" % key: key for key, _ in USB_CAMERAS})


def _mujoco_render_worker(scene_path, camera_specs, fps, command_recv,
                          frame_send):
    """Render MuJoCo frames outside the HTTP/ROS process.

    On the headless Jetson, the first MuJoCo/EGL render can block while holding
    the Python interpreter lock.  Keeping it in the web process prevents the
    RosBridge spin thread from running, so the UI reports NaN even though the
    driver is publishing feedback.  This worker owns its own model, data and
    EGL contexts; the parent only sends qpos snapshots and receives JPEGs.
    """
    renderers = []
    try:
        # Imports must happen in the spawned process after it has its own EGL
        # context.  Importing MuJoCo in the parent is still needed by FakeRobot.
        import mujoco

        model = mujoco.MjModel.from_xml_path(scene_path)
        data = mujoco.MjData(model)
        for key, camera_name, width, height in camera_specs:
            renderer = mujoco.Renderer(model, height=int(height), width=int(width))
            try:
                flags = renderer._scene.flags
                flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
                flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
            except Exception:
                pass
            cam_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
            renderers.append((key, renderer, cam_id))

        latest_qpos = None
        preview_active = False
        preview_site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "tool0_preview")
        interval = 1.0 / max(float(fps), 1.0)
        next_frame = time.monotonic()
        while True:
            # Drain the pipe so a burst of slider updates cannot make the
            # worker render stale poses.
            while command_recv.poll():
                packet = command_recv.recv()
                if packet is None:
                    return
                if isinstance(packet, tuple):
                    packet, preview_active = packet
                latest_qpos = np.asarray(packet, dtype=np.float64).reshape(-1)

            if latest_qpos is not None and latest_qpos.size == model.nq:
                data.qpos[:] = latest_qpos
                mujoco.mj_forward(model, data)
            if preview_site_id >= 0:
                model.site_rgba[preview_site_id, 3] = 0.95 if preview_active else 0.0

            for key, renderer, cam_id in renderers:
                if cam_id >= 0:
                    renderer.update_scene(data, camera=cam_id)
                else:
                    renderer.update_scene(data)
                frame = renderer.render()
                jpeg = ImageCompressor.encode_jpeg(frame, quality=75)
                if jpeg:
                    try:
                        frame_send.send((key, jpeg))
                    except (BrokenPipeError, EOFError, OSError):
                        return

            next_frame += interval
            wait = next_frame - time.monotonic()
            if wait > 0:
                # poll() both sleeps without a busy loop and lets the parent
                # stop the worker promptly by sending the None sentinel.
                if command_recv.poll(wait):
                    packet = command_recv.recv()
                    if packet is None:
                        return
                    if isinstance(packet, tuple):
                        packet, preview_active = packet
                    latest_qpos = np.asarray(packet, dtype=np.float64).reshape(-1)
            else:
                next_frame = time.monotonic()
    except (EOFError, OSError, BrokenPipeError):
        pass
    except Exception as exc:
        print(f"[WebRenderWorker] Loi render: {exc}", flush=True)
    finally:
        for _, renderer, _ in renderers:
            try:
                renderer.close()
            except Exception:
                pass


class ThreadedWebServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class WebControlHandler(BaseHTTPRequestHandler):
    """HTTP Handler phuc vu giao dien Web UI va cac API dieu khien."""

    def log_message(self, format, *args):
        # Tat log console HTTP thong thuong de terminal sach se
        pass

    @property
    def server_ctx(self):
        return self.server.app_context

    def _sid(self) -> str:
        """Ma phien cua tab nay (tu cookie). Rong neu client chua co cookie."""
        raw = self.headers.get("Cookie", "")
        for part in raw.split(";"):
            k, _, v = part.strip().partition("=")
            if k == "sid" and v:
                return v
        return ""

    def _sid_or_new(self) -> str:
        """
        Lay ma phien, chua co thi SINH MOI va gui lai bang cookie.

        Quan trong: phai that su gui cookie, khong chi tra ma tam. Neu chi tra ma
        tam thi client khong luu -> request sau lai la nguoi khac, khong bao gio
        giu duoc quyen dieu khien va moi request lai tao 1 phien rac.
        Phai goi TRUOC khi send_response (header chua duoc gui).
        """
        sid = self._sid()
        if sid:
            return sid
        import uuid
        sid = uuid.uuid4().hex[:12]
        self._pending_sid = sid
        return sid

    def _send_json_header(self):
        pass

    def _flush_sid_cookie(self):
        """Gui Set-Cookie cho phien moi (goi ngay sau send_response)."""
        sid = getattr(self, "_pending_sid", "")
        if sid:
            self._set_sid_cookie(sid)
            self._pending_sid = ""

    def _set_sid_cookie(self, sid: str):
        self.send_header("Set-Cookie", f"sid={sid}; Path=/; SameSite=Lax")

    def do_GET(self):
        p = urlparse(self.path).path
        ctx = self.server_ctx

        if p in ("/", "/index.html"):
            sid = self._sid_or_new()
            html = ctx.render_html()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            # Cap ma phien neu chua co -> moi tab co mode rieng, khong dung chung.
            self._flush_sid_cookie()
            self.end_headers()
            self.wfile.write(html)

        elif p in ("/teleop", "/teleop.html"):
            sid = self._sid_or_new()
            ctx.touch_session(sid)
            html = ctx.render_teleop_html()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self._flush_sid_cookie()
            self.end_headers()
            self.wfile.write(html)

        elif p == "/api/whoami":
            sid = self._sid_or_new()
            name = ctx.get_session_name(sid)
            self._send_json({"sid": sid, "name": name, "mode": ctx.get_session_mode(sid)})

        elif p == "/api/status":
            sid = self._sid_or_new()
            # Chon doi tuong TRUOC khi lay _mj_lock.
            #
            # LOI THAT (nguoi dung bao "keo slider khong hoat dong"): truoc day goi
            # get_active_distribution() BEN TRONG _mj_lock, ma ham do di qua
            # get_session_mode() -> self._lock. The la:
            #   - vong poll /api/status (400ms) giu _mj_lock roi xin self._lock
            #   - keo slider (POST set_absolute) giu self._lock roi xin _mj_lock
            # Hai thu tu nguoc nhau -> DEADLOCK. Da do bang gdb: ~30 thread cung
            # cho 1 semaphore, UI treo VINH VIEN trong khi trang tinh van tra 200.
            target = ctx.active_control_object(sid)
            # distribution cham mj_forward (tcp_pos) -> phai trong khoa MuJoCo.
            with ctx._mj_lock:
                dist = target.distribution
            # Convert numpy arrays to list for JSON serialization
            serialized = {}
            for k, v in dist.items():
                if isinstance(v, np.ndarray):
                    serialized[k] = v.tolist()
                elif isinstance(v, (np.float32, np.float64)):
                    serialized[k] = float(v)
                elif isinstance(v, (np.int32, np.int64)):
                    serialized[k] = int(v)
                else:
                    serialized[k] = v

            # Trang thai an toan cua driver. Sau khi bam DUNG KHAN, driver chot
            # "fault_latched" va TU CHOI moi quy dao cho toi khi rearm - UI phai
            # hien thi duoc, neu khong nguoi dung chi thay robot im lang vo co.
            serialized["safety_state"] = ctx.real_robot.safety_state
            serialized["is_armed"] = ctx.real_robot.is_armed

            # Bo sung thong tin mode hien tai (mode cua CHINH tab nay).
            # Dung qpos_cached: distribution() o tren da doc encoder roi, goi lai
            # qpos se ton them 1 round-trip serial vo ich.
            serialized["ui_mode"] = ctx.get_session_mode(sid)
            # Offset hieu chuan dang dung. Doc qua MODULE chu khong import ten:
            # save_offsets_deg() gan lai bien do, mot ban sao import se thanh so cu.
            serialized["offsets_deg"] = list(ros_bridge.OFFSETS_DEG)
            # Robot that co dang chay quy dao khong (de UI khoa nut va bao loi).
            serialized["trajectory"] = {
                "running": bool(ctx._traj_thread is not None and ctx._traj_thread.is_alive()),
                "error": ctx._traj_error,
            }
            # Pose TCP dang preview tren robot ao (neu co). Doc truoc _mj_lock de
            # giu thu tu khoa duy nhat: _lock -> _mj_lock.
            with ctx._lock:
                ee_preview = (dict(ctx._ee_preview)
                              if ctx._ee_preview is not None else None)
                serialized["ee_preview"] = ee_preview
                serialized["control_tab"] = ctx.get_control_tab(sid)
            # The coordinate tab needs a pose to initialize/track its sliders.
            # In Simulate use the ghost FK; while coordinate preview is active,
            # use the requested target so polling cannot snap sliders backward.
            if ee_preview is not None:
                ee_position = ee_preview.get("position")
                ee_quaternion = ee_preview.get("quaternion")
            elif serialized["ui_mode"] == ctx.MODE_SIMULATE:
                with ctx._mj_lock:
                    _p, _q = ctx.fake_robot.tcp_pose()
                ee_position = _p.tolist()
                ee_quaternion = _q.tolist()
            else:
                ee_position = dist.get("tcp_pos")
                ee_quaternion = dist.get("tcp_quat")
                if ee_position is not None and not np.all(np.isfinite(ee_position)):
                    ee_position = None
                elif isinstance(ee_position, np.ndarray):
                    ee_position = ee_position.tolist()
                if ee_quaternion is not None and not np.all(np.isfinite(ee_quaternion)):
                    ee_quaternion = None
                elif isinstance(ee_quaternion, np.ndarray):
                    ee_quaternion = ee_quaternion.tolist()
            serialized["ee_control_pose"] = {
                "position": ee_position,
                "quaternion": ee_quaternion,
            }
            real_q = ctx.real_robot.qpos_cached
            serialized["real_qpos"] = real_q.tolist()
            serialized["real_deg"] = np.degrees(real_q).tolist()
            # Robot mat nguon/mat ket noi -> UI phai bao ro, khong de nguoi dung
            # tuong slider hong. Robot moi lay tu ROS: driver phat is_connected
            # trong /myarm/robot/diagnostics; is_real_connected doc tu do.
            serialized["hardware_offline"] = not ctx.real_robot.is_real_connected
            serialized["active_deg"] = np.degrees(dist["qpos"]).tolist()
            # Pose THAT cua model MuJoCo (digital twin). Co so nay moi phat hien duoc
            # sim bi dong bang (view thu ba dung yen) thay vi chi tin active_deg.
            # mj_qpos la TOAN BO qpos scene (43 phan tu: 6 khop + vat the tu do),
            # chi lay 6 khop dau. Doc trong _mj_lock: doc .qpos tu thread HTTP trong
            # khi thread render dang photo() tren d -> segfault.
            with ctx._mj_lock:
                sim_deg = np.degrees(ctx.fake_robot.mj_qpos[:6])
            serialized["sim_deg"] = sim_deg.tolist()
            # Che do OFFSET: slider dieu khien CHINH con robot ao, nen vi tri slider
            # phai lay tu m/d MuJoCo. Lay tu Robot.qpos (goc topic, chua hieu chuan)
            # thi slider lech khoi con robot render dung bang OFFSETS_DEG, va cu keo
            # dau tien la nhay mot cu.
            if serialized["ui_mode"] == ctx.MODE_OFFSET:
                serialized["active_deg"] = sim_deg.tolist()
            # Nhiet do servo: servo qua nhiet se tu ngat va TU CHOI lenh, nen phai
            # hien thi de nguoi dung biet tai sao robot khong nhuc nhich.
            # nan -> None de JSON hop le (UI hien '--' thay vi 'NaN').
            _temps = ctx.real_robot.temperatures()
            serialized["temperatures"] = [
                None if not np.isfinite(t) else float(t) for t in _temps
            ]

            resp = json.dumps(serialized).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        elif p == "/api/teleop/status":
            sid = self._sid_or_new()
            ctx.touch_session(sid)
            self._send_json(ctx.teleop_status(sid))

        elif p in STREAM_KEYS:
            key = STREAM_KEYS[p]
            if not ctx.stream_available(key):
                # Camera khong duoc cau hinh (vd chay chi voi cam_sim). Phai tra
                # loi ngay: _serve_mjpeg gui 200 roi lap vo han cho anh, nen
                # trinh duyet se treo spinner mai ma khong bao gi.
                self.send_error(503, "Camera khong duoc cau hinh")
            else:
                self._serve_mjpeg(key)

        else:
            self.send_error(404)

    def do_POST(self):
        p = urlparse(self.path).path
        ctx = self.server_ctx

        content_len = int(self.headers.get("Content-Length", 0))
        body = {}
        if content_len > 0:
            try:
                body = json.loads(self.rfile.read(content_len).decode("utf-8"))
            except Exception:
                pass

        sid = self._sid_or_new()
        name = ctx.get_session_name(sid)
        ctx.touch_session(sid)

        if p == "/api/change":
            # Goi ham change(p, delta) tren object control hien tai
            param = body.get("param")
            delta = float(body.get("delta", 0.0))
            success = ctx.handle_change(param, delta, sid)
            self._send_json({"status": "ok" if success else "error"})

        elif p == "/api/set_absolute":
            # Dat gia tri tuyet doi cho khop (tinh delta tuong ung)
            param = body.get("param")
            target_val = float(body.get("value", 0.0))  # rad hoac grip
            success = ctx.handle_set_absolute(param, target_val, sid)
            self._send_json({"status": "ok" if success else "error"})

        elif p == "/api/ee_preview":
            # IK display-only tren MuJoCo; tuyet doi khong publish lenh robot that.
            result = ctx.preview_ee_pose(body, sid)
            self._send_json({"status": "ok" if result.get("ok") else "error",
                             **result})

        elif p == "/api/move_ee":
            # Lenh that: goi Robot.set_tcp_pose() voi pose da validate.
            result = ctx.move_ee(body, sid)
            self._send_json({"status": "ok" if result.get("ok") else "error",
                             **result})

        elif p == "/api/set_ee_absolute":
            # Giong slider khop: control -> robot that, simulate -> ghost IK.
            result = ctx.set_ee_absolute(body, sid)
            self._send_json({"status": "ok" if result.get("ok") else "error",
                             **result})

        elif p == "/api/teleop/step":
            result = ctx.teleop_step(body, sid)
            self._send_json({"status": "ok" if result.get("ok") else "error",
                             **result})

        elif p == "/api/teleop/reset":
            result = ctx.reset_teleop(sid)
            self._send_json({"status": "ok" if result.get("ok") else "error",
                             **result})

        elif p == "/api/set_control_tab":
            tab = body.get("tab", "joint")
            ok = ctx.set_control_tab(sid, tab)
            self._send_json({"status": "ok" if ok else "error",
                             "tab": ctx.get_control_tab(sid)})

        elif p == "/api/set_mode":
            # Chuyen doi mode: 'control', 'simulate' hoac 'offset' (chi doi cho tab nay)
            new_mode = body.get("mode", "control")
            success = ctx.set_ui_mode(sid, new_mode)
            self._send_json({"status": "ok" if success else "error",
                             "mode": ctx.get_session_mode(sid)})

        elif p == "/api/offsets/save":
            # Hieu chuan offset.
            #
            # Che do OFFSET: nguoi dung keo slider dua con robot ao ve dung tu the
            # thuc te. Con robot ao dang duoc ve qua model_rad() (xem
            # FakeRobot.sync), tuc la:
            #
            #     sim = real - offset
            #
            # Nguoi dung keo sim toi S roi bam Luu. Muon con robot ao DUNG YEN tai
            # dung S sau khi luu thi offset moi phai thoa sim = real - offset_moi:
            #
            #     offset_moi = real - sim
            #
            # Cong thuc nay IDEMPOTENT: bam Luu lai ma khong keo gi thi sim van
            # = real - offset_moi, nen offset khong doi.
            #
            # Don vi la DO: hai ve deu da qua np.degrees truoc khi tru.
            cu = np.array(ros_bridge.OFFSETS_DEG)
            with ctx._mj_lock:
                sim_deg = np.degrees(ctx.fake_robot.mj_qpos[:6]).copy()
            real_deg = np.degrees(ctx.real_robot.qpos_cached).copy()
            try:
                new = ros_bridge.save_offsets_deg(real_deg - sim_deg)
            except ValueError as e:
                self._send_json({"status": "error", "message": str(e)})
                return
            doi = np.round(np.array(new) - cu, 3)
            print("[WebControl] Da luu offset moi (deg): %s  (doi: %s)"
                  % ([round(v, 2) for v in new], list(doi)))
            self._send_json({"status": "ok", "offsets_deg": list(new),
                             "hieu_chinh_deg": list(doi)})

        elif p == "/api/apply_to_real":
            # Option trong simulate: 'Dua robot that ve pose nay' -> goi FollowTrajectory
            result = ctx.apply_fake_pose_to_real(sid, body.get("control_tab"))
            self._send_json({"status": "ok" if result.get("ok") else "error",
                             **result})

        elif p == "/api/preset":
            # Ap dung cac pose preset mac dinh
            preset_name = body.get("name", "home")
            success = ctx.apply_preset(preset_name, sid)
            self._send_json({"status": "ok" if success else "error"})

        elif p == "/api/reset_scene":
            # Reset toan bo scene mo phong
            success = ctx.reset_scene()
            self._send_json({"status": "ok" if success else "error", "message": "Da reset scene ve vi tri ban dau!"})

        elif p == "/api/stop":
            # Dung khan cap: KHONG doi quyen, ai cung bam duoc. Dung ngay tai cho.
            ctx.stop()
            print(f"[WebControl] DUNG KHAN CAP boi '{name}'.")
            self._send_json({"status": "ok", "message": "Da dung robot!"})

        elif p == "/api/power_off":
            # Ngat mo-men roi cat dien. release truoc de servo thoi giu vi tri
            # (chi power_off thi servo van giu vat va tiep tuc nong).
            ok = ctx.power_off()
            self._send_json({"status": "ok" if ok else "error",
                             "message": "Da tat nguon robot!" if ok else "Tat nguon that bai."})

        elif p == "/api/power_on":
            # Bat lai dien + mo-men. Sau khi tat nguon, servo mat mo-men nen tay co
            # the da tut xuong; KHONG tu di chuyen, chi cap dien lai thoi.
            ok = ctx.power_on()
            self._send_json({"status": "ok" if ok else "error",
                             "message": "Da bat nguon robot!" if ok else "Bat nguon that bai."})

        elif p == "/api/rearm":
            # Khoi phuc sau DUNG KHAN. Driver chot fault_latched va tu choi moi
            # quy dao; khong co buoc nay thi bam DUNG KHAN xong la het duong lam
            # tiep, phai khoi dong lai ca stack.
            ok = ctx.rearm()
            self._send_json({"status": "ok" if ok else "error",
                             "message": "Da khoi phuc (armed)!" if ok else "Khoi phuc that bai.",
                             "safety_state": ctx.real_robot.safety_state})

        else:
            self.send_error(404)

    def _send_json(self, data: dict, code: int = 200):
        payload = json.dumps(data).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        # Client goi API bang curl/script khong co cookie -> cap ma phien de lan
        # sau ho giu duoc quyen dieu khien (neu khong moi request la 1 nguoi moi).
        self._flush_sid_cookie()
        self.end_headers()
        self.wfile.write(payload)

    def _serve_mjpeg(self, key: str):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.end_headers()

        ctx = self.server_ctx
        try:
            while ctx.running:
                # Doc JPEG tu cache dung chung THAY VI tu render.
                # Truoc day moi tab tu goi photo() -> N tab = N lan render cung mot
                # khung hinh (render MuJoCo ~24 ms + JPEG). Gio chi 1 thread render,
                # moi tab chi doc lai bytes co san.
                jpeg = ctx.get_stream_jpeg(key)
                if jpeg:
                    hdr = (
                        f"--frame\r\n"
                        f"Content-Type: image/jpeg\r\n"
                        f"Content-Length: {len(jpeg)}\r\n\r\n"
                    ).encode("utf-8")
                    self.wfile.write(hdr + jpeg + b"\r\n")
                time.sleep(1.0 / 25.0)
        except Exception:
            pass


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
        """Tao giao dien Web HTML/CSS/JS hien dai voi slider, camera stream, va buttons."""
        joint_names = self.real_robot.JOINT_NAMES
        rad_min = self.real_robot.rad_min
        rad_max = self.real_robot.rad_max
        deg_min = np.degrees(rad_min).tolist()
        deg_max = np.degrees(rad_max).tolist()

        # Khung camera USB: sinh tu USB_CAMERAS de danh sach thiet bi chi duoc
        # khai bao o MOT cho (program/camera/usb_camera.py).
        usb_windows = "".join(
            '<div class="usb-window">'
            '<img src="/stream/cam_%s.mjpg" alt="%s">'
            '<div class="usb-label">%s</div>'
            "</div>" % (key, key, device)
            for key, device in USB_CAMERAS
        )

        html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <title>myArm M750 VLA - Robot Web Control Interface</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: #0d1117;
      color: #c9d1d9;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      padding: 16px;
    }}
    header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-bottom: 12px;
      margin-bottom: 16px;
      border-bottom: 1px solid #30363d;
    }}
    .title-box h1 {{
      font-size: 1.3rem;
      font-weight: 600;
      color: #58a6ff;
    }}
    .title-box p {{
      font-size: 0.8rem;
      color: #8b949e;
      margin-top: 2px;
    }}
    .mode-bar {{
      display: flex;
      gap: 10px;
      align-items: center;
    }}
    .btn {{
      padding: 8px 16px;
      border-radius: 6px;
      font-weight: 600;
      font-size: 0.85rem;
      cursor: pointer;
      border: 1px solid transparent;
      transition: all 0.2s;
    }}
    .btn-mode {{
      background: #21262d;
      color: #8b949e;
      border-color: #30363d;
    }}
    .btn-mode.active-control {{
      background: #238636 !important;
      color: #fff !important;
      border-color: #2ea043 !important;
      box-shadow: 0 0 10px rgba(46, 160, 67, 0.4);
    }}
    .btn-mode.active-simulate {{
      background: #d29922 !important;
      color: #000 !important;
      border-color: #bb8009 !important;
      box-shadow: 0 0 10px rgba(210, 153, 34, 0.4);
    }}
    .btn-apply {{
      background: #8957e5;
      color: #fff;
      border-color: #a371f7;
    }}
    /* Nut "Dua robot that ve pose nay" chi hien khi vao mode simulate. */
    #btnApply {{
      display: none;
    }}
    .btn-apply:hover {{
      background: #a371f7;
    }}
    .main-grid {{
      display: grid;
      grid-template-columns: 640px 1fr;
      gap: 16px;
    }}
    .viewport-card, .control-card, .usb-card {{
      background: #161b22;
      border: 1px solid #30363d;
      border-radius: 8px;
      padding: 14px;
    }}
    .usb-card {{
      grid-column: 1 / -1;
    }}
    .usb-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }}
    .usb-window {{
      background: #000;
      border: 1px solid #21262d;
      border-radius: 6px;
      overflow: hidden;
    }}
    .usb-window img {{
      width: 100%;
      height: 300px;
      /* contain chu KHONG cover: day la anh that de doi chieu, cat bot la mat goc nhin */
      object-fit: contain;
      display: block;
    }}
    .usb-label {{
      padding: 4px 8px;
      font-size: 0.7rem;
      color: #8b949e;
      background: #0d1117;
    }}
    .stream-container {{
      width: 640px;
      height: 480px;
      background: #000;
      border-radius: 6px;
      overflow: hidden;
      position: relative;
      border: 1px solid #21262d;
    }}
    .stream-container img {{
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
    }}
    .cam-nav {{
      display: flex;
      gap: 8px;
      margin-top: 10px;
    }}
    .cam-nav button {{
      flex: 1;
      padding: 6px;
      background: #21262d;
      color: #c9d1d9;
      border: 1px solid #30363d;
      border-radius: 4px;
      font-size: 0.75rem;
      cursor: pointer;
    }}
    .cam-nav button.active {{
      background: #1f6feb;
      border-color: #388bfd;
      color: #fff;
    }}
    .slider-group {{
      margin-bottom: 12px;
      padding-bottom: 8px;
      border-bottom: 1px solid #21262d;
    }}
    .slider-header {{
      display: flex;
      justify-content: space-between;
      font-size: 0.8rem;
      font-weight: 600;
      margin-bottom: 4px;
    }}
    .slider-header span.val {{
      color: #58a6ff;
      font-family: monospace;
    }}
    .slider-controls {{
      display: flex;
      gap: 10px;
      align-items: center;
    }}
    input[type=range] {{
      flex: 1;
      accent-color: #1f6feb;
      cursor: pointer;
    }}
    .btn-step {{
      background: #21262d;
      color: #c9d1d9;
      border: 1px solid #30363d;
      border-radius: 4px;
      width: 28px;
      height: 24px;
      font-size: 0.8rem;
      cursor: pointer;
    }}
    .btn-step:hover {{
      background: #30363d;
    }}
    .badge {{
      display: inline-block;
      padding: 2px 8px;
      border-radius: 12px;
      font-size: 0.75rem;
      font-weight: 600;
    }}
    .badge-sim {{ background: rgba(210, 153, 34, 0.2); color: #e3b341; border: 1px solid #d29922; }}
    .badge-real {{ background: rgba(46, 160, 67, 0.2); color: #3fb950; border: 1px solid #2ea043; }}
    .preset-bar {{
      display: flex;
      gap: 8px;
      margin-top: 14px;
    }}
    .preset-bar button {{
      flex: 1;
      background: #21262d;
      color: #c9d1d9;
      border: 1px solid #30363d;
      padding: 6px;
      border-radius: 4px;
      font-size: 0.78rem;
      cursor: pointer;
    }}
    .preset-bar button:hover {{
      background: #30363d;
      color: #58a6ff;
    }}
    .control-tabs {{
      display: flex;
      gap: 6px;
      margin: -2px -2px 14px;
      border-bottom: 1px solid #30363d;
    }}
    .control-tab {{
      flex: 1;
      padding: 9px 10px;
      color: #8b949e;
      background: #21262d;
      border: 1px solid #30363d;
      border-bottom: 0;
      border-radius: 5px 5px 0 0;
      font-size: 0.78rem;
      font-weight: 600;
      cursor: pointer;
    }}
    .control-tab.active {{
      color: #fff;
      background: #1f6feb;
      border-color: #388bfd;
    }}
    .control-panel {{
      display: none;
    }}
    .control-panel.active {{
      display: block;
    }}
    .ee-card h3 {{
      font-size: 0.95rem;
      margin-bottom: 6px;
      color: #f0f6fc;
    }}
    .ee-help {{
      color: #8b949e;
      font-size: 0.75rem;
      line-height: 1.4;
      margin-bottom: 10px;
    }}
    .ee-slider-group {{
      margin-bottom: 12px;
      padding-bottom: 8px;
      border-bottom: 1px solid #21262d;
    }}
    .ee-slider-header {{
      display: flex;
      justify-content: space-between;
      margin-bottom: 4px;
      color: #c9d1d9;
      font-size: 0.8rem;
      font-weight: 600;
    }}
    .ee-slider-header .val {{
      color: #58a6ff;
      font-family: monospace;
    }}
    .ee-slider-controls {{
      display: flex;
      gap: 10px;
      align-items: center;
    }}
    .ee-slider-controls input[type=range] {{
      flex: 1;
    }}
    .ee-field {{
      display: flex;
      flex-direction: column;
      gap: 4px;
      color: #8b949e;
      font-size: 0.72rem;
    }}
    .ee-info {{
      min-height: 18px;
      margin-top: 9px;
      color: #8b949e;
      font-size: 0.75rem;
      font-family: monospace;
    }}
    @media (max-width: 900px) {{
      .main-grid {{ grid-template-columns: 1fr; }}
      .stream-container {{ width: 100%; height: auto; aspect-ratio: 4 / 3; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="title-box">
      <h1>myArm M750 - Hệ Thống Điều Khiển Robot VLA</h1>
      <p>Cấu trúc framework tích hợp Real Robot, Ghost Simulator & Camera</p>
    </div>
    <a href="/teleop" style="color:#c9d1d9; border:1px solid #30363d; border-radius:6px;
       padding:8px 12px; text-decoration:none; font-size:0.8rem; white-space:nowrap;">TELEOP</a>
    <div class="mode-bar">
      <span id="badgeStatus" class="badge badge-real">MODE: CONTROL</span>
      <button id="btnControl" class="btn btn-mode active-control" onclick="setMode('control')">CONTROL (Robot Thật)</button>
      <button id="btnSimulate" class="btn btn-mode" onclick="setMode('simulate')">SIMULATE (Robot Ảo)</button>
      <button id="btnOffset" class="btn btn-mode" onclick="setMode('offset')">OFFSET (Hiệu Chuẩn)</button>
      <button id="btnApply" class="btn btn-apply" onclick="applyToReal()">🚀 Đưa robot thật về pose này</button>
      <button id="btnSaveOffset" class="btn btn-apply" onclick="saveOffsets()" style="display:none">💾 LƯU OFFSET</button>
    </div>
  </header>

  <div class="main-grid">
    <!-- Camera Viewport Card -->
    <div class="viewport-card">
      <div class="stream-container">
        <img id="streamView" src="/stream/cam_sim.mjpg" alt="Video Feed">
      </div>
      <div class="cam-nav">
        <button id="camSimBtn" class="active" onclick="switchStream('sim')">🎥 Wrist Cam (Sim)</button>
        <button id="camThirdBtn" onclick="switchStream('third')">🎬 Third-Person (Sim)</button>
        <button id="camRealBtn" onclick="switchStream('real')">📷 USB Real Camera</button>
      </div>
      <div style="margin-top: 12px; font-size: 0.75rem; color: #8b949e; line-height: 1.4;">
        <div><b>TCP Position:</b> <span id="tcpCoord" style="color: #58a6ff;">[0.0, 0.0, 0.0]</span></div>
        <div><b>Trạng thái phần cứng:</b> <span id="hwStatus" style="color: #3fb950;">Checking...</span></div>
        <div><b>Nhiệt độ servo:</b> <span id="tempStatus" style="color: #8b949e;">--</span></div>
      </div>
      <!-- Dung khan cap: luon hien, khong phu thuoc mode -->
      <div class="cam-nav" style="margin-top: 10px;">
        <button onclick="stopRobot()" style="color: #ff7b72; border-color: #f85149; font-weight: 700;">⛔ DỪNG KHẨN</button>
        <button onclick="rearmRobot()" style="color: #d29922; border-color: #d29922;">🔓 KHÔI PHỤC</button>
        <button onclick="powerOn()" style="color: #3fb950; border-color: #3fb950;">⚡ Bật nguồn</button>
        <button onclick="powerOff()" style="color: #ff7b72; border-color: #f85149;">🔌 Tắt nguồn</button>
      </div>
      <div style="margin-top: 6px;">
        <b>An toàn:</b> <span id="safetyBadge" style="color:#8b949e;">--</span>
      </div>
      <div id="errBox" style="display:none; margin-top: 8px; padding: 8px 10px; border-radius: 6px;
           background: #3d1418; border: 1px solid #f85149; color: #ff7b72; font-size: 0.75rem;"></div>
    </div>

    <!-- Joint and coordinate controllers share one tabbed control card. -->
    <div class="control-card">
      <div class="control-tabs">
        <button id="tabJoint" class="control-tab active" onclick="setControlTab('joint')">
          Bảng Điều Khiển Khớp (change() API)
        </button>
        <button id="tabCoordinate" class="control-tab" onclick="setControlTab('coordinate')">
          Move EE with Coord
        </button>
      </div>

      <div id="jointPanel" class="control-panel active">
        <h3 style="font-size: 0.95rem; margin-bottom: 12px; color: #f0f6fc;">Điều Khiển Khớp</h3>
        <div id="slidersContainer"></div>

        <!-- Gripper Slider -->
        <div class="slider-group">
          <div class="slider-header">
            <span>Tay kẹp (Gripper)</span>
            <span class="val" id="val_grip">0.0%</span>
          </div>
          <div class="slider-controls">
            <button class="btn-step" onclick="stepChange('gripper', -5.0)">-</button>
            <input type="range" id="slider_grip" min="0" max="100" step="1" value="0"
                   oninput="onSliderDrag('gripper', this.value)" onchange="onSliderCommit('gripper', this.value)">
            <button class="btn-step" onclick="stepChange('gripper', 5.0)">+</button>
          </div>
        </div>

        <!-- Preset Actions -->
        <div class="preset-bar">
          <button onclick="applyPreset('home')">Home Pose</button>
          <button onclick="applyPreset('pre_grasp')">Pre-Grasp</button>
          <button onclick="applyPreset('rest')">Rest Pose</button>
          <button onclick="applyPreset('photo')">📷 Pose ảnh</button>
          <button onclick="stepChange('gripper', 100)">Mở Kẹp</button>
          <button onclick="stepChange('gripper', -100)">Đóng Kẹp</button>
          <button onclick="resetScene()" style="color: #ff7b72; border-color: #f85149;">🔄 Reset Scene</button>
        </div>
      </div>

      <div id="coordinatePanel" class="control-panel">
        <h3 style="font-size: 0.95rem; margin-bottom: 6px; color: #f0f6fc;">Điều Khiển EE bằng Tọa Độ</h3>
        <div class="ee-help">
          Pose tool0 trong frame <code>base_link</code>. Kéo slider để thay đổi.
          XYZ dùng mét; quaternion theo thứ tự <code>qx, qy, qz, qw</code>.
          Ở SIMULATE, ghost robot cập nhật ngay sau khi thả slider; ở CONTROL,
          slider gửi <code>set_tcp_pose()</code> tới robot thật.
          Nút Submit dùng chung ở thanh trên.
        </div>
        <div class="ee-slider-group">
          <div class="ee-slider-header"><span>X (m)</span><span class="val" id="val_ee_x">0.300</span></div>
          <div class="ee-slider-controls">
            <button class="btn-step" onclick="stepEESlider('x', -0.005)">-</button>
            <input type="range" id="ee_x" min="-0.10" max="0.70" step="0.001" value="0.300"
                   oninput="onEESliderInput('x', this.value)" onchange="onEESliderCommit()">
            <button class="btn-step" onclick="stepEESlider('x', 0.005)">+</button>
          </div>
        </div>
        <div class="ee-slider-group">
          <div class="ee-slider-header"><span>Y (m)</span><span class="val" id="val_ee_y">0.000</span></div>
          <div class="ee-slider-controls">
            <button class="btn-step" onclick="stepEESlider('y', -0.005)">-</button>
            <input type="range" id="ee_y" min="-0.60" max="0.60" step="0.001" value="0.000"
                   oninput="onEESliderInput('y', this.value)" onchange="onEESliderCommit()">
            <button class="btn-step" onclick="stepEESlider('y', 0.005)">+</button>
          </div>
        </div>
        <div class="ee-slider-group">
          <div class="ee-slider-header"><span>Z (m)</span><span class="val" id="val_ee_z">0.250</span></div>
          <div class="ee-slider-controls">
            <button class="btn-step" onclick="stepEESlider('z', -0.005)">-</button>
            <input type="range" id="ee_z" min="0.00" max="0.70" step="0.001" value="0.250"
                   oninput="onEESliderInput('z', this.value)" onchange="onEESliderCommit()">
            <button class="btn-step" onclick="stepEESlider('z', 0.005)">+</button>
          </div>
        </div>
        <div class="ee-slider-group">
          <div class="ee-slider-header"><span>qx</span><span class="val" id="val_ee_qx">0.000</span></div>
          <div class="ee-slider-controls">
            <button class="btn-step" onclick="stepEESlider('qx', -0.01)">-</button>
            <input type="range" id="ee_qx" min="-1.00" max="1.00" step="0.01" value="0.000"
                   oninput="onEESliderInput('qx', this.value)" onchange="onEESliderCommit()">
            <button class="btn-step" onclick="stepEESlider('qx', 0.01)">+</button>
          </div>
        </div>
        <div class="ee-slider-group">
          <div class="ee-slider-header"><span>qy</span><span class="val" id="val_ee_qy">0.000</span></div>
          <div class="ee-slider-controls">
            <button class="btn-step" onclick="stepEESlider('qy', -0.01)">-</button>
            <input type="range" id="ee_qy" min="-1.00" max="1.00" step="0.01" value="0.000"
                   oninput="onEESliderInput('qy', this.value)" onchange="onEESliderCommit()">
            <button class="btn-step" onclick="stepEESlider('qy', 0.01)">+</button>
          </div>
        </div>
        <div class="ee-slider-group">
          <div class="ee-slider-header"><span>qz</span><span class="val" id="val_ee_qz">0.000</span></div>
          <div class="ee-slider-controls">
            <button class="btn-step" onclick="stepEESlider('qz', -0.01)">-</button>
            <input type="range" id="ee_qz" min="-1.00" max="1.00" step="0.01" value="0.000"
                   oninput="onEESliderInput('qz', this.value)" onchange="onEESliderCommit()">
            <button class="btn-step" onclick="stepEESlider('qz', 0.01)">+</button>
          </div>
        </div>
        <div class="ee-slider-group">
          <div class="ee-slider-header"><span>qw</span><span class="val" id="val_ee_qw">1.000</span></div>
          <div class="ee-slider-controls">
            <button class="btn-step" onclick="stepEESlider('qw', -0.01)">-</button>
            <input type="range" id="ee_qw" min="-1.00" max="1.00" step="0.01" value="1.000"
                   oninput="onEESliderInput('qw', this.value)" onchange="onEESliderCommit()">
            <button class="btn-step" onclick="stepEESlider('qw', 0.01)">+</button>
          </div>
        </div>
        <div id="eePreviewInfo" class="ee-info">Chưa có pose tọa độ.</div>
      </div>
    </div>

    <!-- Camera USB: 2 thiet bi V4L, luon hien, doc lap hoan toan voi robot -->
    <div class="usb-card">
      <h3 style="font-size: 0.95rem; margin-bottom: 12px; color: #f0f6fc;">Camera USB</h3>
      <div class="usb-grid">
        {usb_windows}
      </div>
    </div>
  </div>

  <script>
    const JOINT_NAMES = {json.dumps(joint_names)};
    const DEG_MIN = {json.dumps(deg_min)};
    const DEG_MAX = {json.dumps(deg_max)};
    let currentMode = "control";
    let currentControlTab = "joint";
    let currentStream = "sim";
    let isDragging = false;
    // Slider vua tha nhung tay robot chua toi: giu hien gia tri nguoi dung chon
    // cho toi khi encoder duoi theo kip (xem onSliderCommit).
    let pendingSlider = {{}};
    // Coordinate slider vua tha nhung feedback/IK chua theo kip.
    let pendingEE = null;

    // Render 6 Joint Sliders
    function initSliders() {{
      const container = document.getElementById("slidersContainer");
      let html = "";
      for (let i = 0; i < 6; i++) {{
        html += `
          <div class="slider-group">
            <div class="slider-header">
              <span>J${{i+1}}: ${{JOINT_NAMES[i]}}</span>
              <span class="val" id="val_j${{i}}">0.0° (0.00 rad)</span>
            </div>
            <div class="slider-controls">
              <button class="btn-step" onclick="stepChange(${{i}}, -0.0087266)">-</button>
              <input type="range" id="slider_j${{i}}"
                     min="${{DEG_MIN[i]}}" max="${{DEG_MAX[i]}}" step="0.5" value="0"
                     oninput="onSliderDrag(${{i}}, this.value)"
                     onchange="onSliderCommit(${{i}}, this.value)">
              <button class="btn-step" onclick="stepChange(${{i}}, 0.0087266)">+</button>
            </div>
          </div>
        `;
      }}
      container.innerHTML = html;
    }}

    function showErr(msg) {{
      const box = document.getElementById("errBox");
      box.style.display = "block";
      box.innerText = msg;
    }}

    function clearErr() {{
      document.getElementById("errBox").style.display = "none";
    }}

    // Helper POST dung chung: MOI lenh deu kiem tra status tra ve.
    // Truoc day cac nut bo qua ket qua nen lenh that bai (vi pham gioi han,
    // servo qua nhiet) im lang khong bao gi.
    function post(url, body) {{
      return fetch(url, {{
        method: "POST",
        headers: {{"Content-Type": "application/json"}},
        body: body ? JSON.stringify(body) : undefined
      }})
        .then(r => r.json())
        .then(d => {{
          if (d && d.status === "error") {{
            showErr("Lệnh thất bại: " + (d.message || "robot từ chối (kiểm tra giới hạn khớp / nhiệt độ servo)"));
          }} else {{
            clearErr();
          }}
          return d;
        }})
        .catch(e => {{ showErr("Lỗi kết nối: " + e); }});
    }}

    function stopRobot() {{
      post("/api/stop").then(() => fetchStatus());
    }}

    function rearmRobot() {{
      // Sau DUNG KHAN, driver chot fault_latched va TU CHOI moi quy dao. Khong
      // bam nut nay thi robot im lang mai du moi thu khac van bao binh thuong.
      post("/api/rearm").then(d => {{
        if (d && d.status === "ok") {{
          clearErr();
        }} else {{
          showErr("Khôi phục thất bại: " + ((d && d.message) || "driver từ chối"));
        }}
        fetchStatus();
      }});
    }}

    // Trang thai robot that dang chay quy dao, cap nhat tu /api/status.
    let trajRunning = false;

    function updateApplyButton() {{
      const b = document.getElementById("btnApply");
      if (!b) return;
      b.disabled = trajRunning;
      b.style.opacity = trajRunning ? "0.45" : "1";
      b.innerText = trajRunning ? "⏳ Đang di chuyển robot thật..."
                                : "🚀 Đưa robot thật về pose này";
    }}

    function powerOff() {{
      if (confirm("Tắt nguồn robot? Servo sẽ mất mô-men và tay có thể rơi xuống.")) {{
        post("/api/power_off").then(d => {{
          if (d && d.status === "ok") alert(d.message || "Đã tắt nguồn.");
          fetchStatus();
        }});
      }}
    }}

    function switchStream(type) {{
      const img = document.getElementById("streamView");
      const btns = {{
        sim:   document.getElementById("camSimBtn"),
        third: document.getElementById("camThirdBtn"),
        real:  document.getElementById("camRealBtn"),
      }};
      const srcs = {{
        sim:   "/stream/cam_sim.mjpg",
        third: "/stream/cam_third.mjpg",
        real:  "/stream/cam_real.mjpg",
      }};
      if (!srcs[type]) type = "sim";
      // Doi src cua <img> khong tu huy ket noi MJPEG cu -> phai ep tai lai
      img.src = "about:blank";
      img.src = srcs[type];
      Object.keys(btns).forEach(k => {{
        if (btns[k]) btns[k].classList.toggle("active", k === type);
      }});
      currentStream = type;
    }}

    function powerOn() {{
      if (confirm("Bật nguồn robot? Servo sẽ giữ lại vị trí hiện tại (tay có thể đã tụt sau khi tắt nguồn).")) {{
        post("/api/power_on").then(d => {{
          if (d && d.status === "ok") alert(d.message || "Đã bật nguồn.");
          else alert("Lỗi: " + ((d && d.message) || "không bật được nguồn"));
        }});
      }}
    }}

    function setMode(mode) {{
      // A mode switch establishes a new pose source. Do not keep a coordinate
      // target from the previous mode while the server syncs the ghost.
      if (mode !== currentMode) {{
        pendingEE = null;
        isDragging = false;
      }}
      post("/api/set_mode", {{mode: mode}}).then(data => {{
        if (data && data.mode) updateModeUI(data.mode);
        fetchStatus();
      }});
    }}

    function renderControlTab(tab) {{
      if (tab !== "joint" && tab !== "coordinate") return;
      currentControlTab = tab;
      const jointTab = document.getElementById("tabJoint");
      const coordTab = document.getElementById("tabCoordinate");
      const jointPanel = document.getElementById("jointPanel");
      const coordPanel = document.getElementById("coordinatePanel");
      jointTab.classList.toggle("active", tab === "joint");
      coordTab.classList.toggle("active", tab === "coordinate");
      jointPanel.classList.toggle("active", tab === "joint");
      coordPanel.classList.toggle("active", tab === "coordinate");
    }}

    function setControlTab(tab) {{
      if (tab !== "joint" && tab !== "coordinate") return;
      renderControlTab(tab);
      post("/api/set_control_tab", {{tab: tab}}).then(() => fetchStatus());
    }}

    function updateModeUI(mode) {{
      currentMode = mode;
      const bControl = document.getElementById("btnControl");
      const bSim = document.getElementById("btnSimulate");
      const bOff = document.getElementById("btnOffset");
      const bApply = document.getElementById("btnApply");
      const bSave = document.getElementById("btnSaveOffset");
      const badge = document.getElementById("badgeStatus");

      bControl.className = "btn btn-mode";
      bSim.className = "btn btn-mode";
      bOff.className = "btn btn-mode";
      bApply.style.display = "none";
      bSave.style.display = "none";

      if (mode === "simulate") {{
        bSim.className = "btn btn-mode active-simulate";
        bApply.style.display = "inline-block";
        badge.className = "badge badge-sim";
        badge.innerText = "MODE: SIMULATE (Ghost Robot)";
      }} else if (mode === "offset") {{
        bOff.className = "btn btn-mode active-simulate";
        bSave.style.display = "inline-block";
        badge.className = "badge badge-sim";
        badge.innerText = "MODE: OFFSET - kéo slider căn robot ảo cho khớp tay thật, rồi LƯU";
      }} else {{
        bControl.className = "btn btn-mode active-control";
        badge.className = "badge badge-real";
        badge.innerText = "MODE: CONTROL (Real Robot)";
      }}
    }}

    function saveOffsets() {{
      if (!confirm("Lưu offset hiệu chuẩn?\\n\\nOffset = góc RAW của tay thật - góc model của robot ảo.\\nGhi vào offsets.json và áp dụng ngay cho cả tiến trình.")) return;
      post("/api/offsets/save", {{}}).then(d => {{
        if (d && d.status === "ok") {{
          alert("Đã lưu offset (deg):\\n" + (d.offsets_deg || []).map(v => v.toFixed(2)).join(", "));
        }} else {{
          alert("Lỗi: " + ((d && d.message) || "không lưu được offset"));
        }}
      }});
    }}

    function onSliderDrag(param, val) {{
      isDragging = true;
      // Nguoi dung tu keo lai -> bo gia tri dang cho, de khong tranh nhau.
      delete pendingSlider[param];
      if (param === "gripper") {{
        document.getElementById("val_grip").innerText = parseFloat(val).toFixed(1) + "%";
      }} else {{
        const deg = parseFloat(val);
        const rad = deg * Math.PI / 180.0;
        document.getElementById("val_j" + param).innerText = deg.toFixed(1) + "° (" + rad.toFixed(2) + " rad)";
      }}
    }}

    function onSliderCommit(param, val) {{
      // Thu tu QUAN TRONG: ghi pendingSlider TRUOC, roi moi isDragging = false.
      //
      // LOI THAT (nguoi dung bao: "dang dieu khien duoc thi keo slider no lai
      // quay ve gia tri cu"): tha tay -> onchange -> gui lenh. Nhung tay robot
      // di mat vai GIAY, trong khi vong poll /api/status chay moi 400ms. Lan poll
      // ke tiep tra ve active_deg CU (tay chua toi) va ghi de slider -> slider
      // nhay ve cho cu. Sau do tay moi di, nhung slider da sai.
      //
      // Cach sua: giu gia tri vua chon trong pendingSlider[] cho toi khi encoder
      // duoi theo kip, hoac qua 8s thi bo cuoc de khong ket vinh vien.
      let target_val = 0;
      let target_display = 0;
      if (param === "gripper") {{
        // Gui PHAN TRAM (0..100), khong phai met. Server tu quy doi qua
        // Robot.set_gripper_pct() -> mot nguon su that cho do mo toi da.
        // Truoc day JS tu nhan voi 0.0345 (quy uoc MOT ngon cua SDK cu), lech
        // han voi quy uoc TONG 0..0.08 cua stack ROS hien tai.
        target_val = parseFloat(val);
        target_display = parseFloat(val);
      }} else {{
        target_val = parseFloat(val) * Math.PI / 180.0;
        target_display = parseFloat(val);
      }}
      pendingSlider[param] = {{display: target_display, until: Date.now() + 8000}};
      isDragging = false;   // sau khi ghi pending -> khong con bi ghi de
      post("/api/set_absolute", {{param: param, value: target_val}}).then(() => fetchStatus());
    }}

    function stepChange(param, delta) {{
      // Gripper: ca slider lan nut bam deu dung PHAN TRAM (0..100).
      // Truoc day JS tu doi sang met bang 0.0345 -> lech quy uoc (0..0.08).
      post("/api/change", {{param: param, delta: delta}}).then(() => fetchStatus());
    }}

    function applyToReal() {{
      if (confirm("Xác nhận đưa robot thật di chuyển về pose mô phỏng hiện tại?")) {{
        post("/api/apply_to_real", {{control_tab: currentControlTab}}).then(d => {{
          if (d && d.status !== "error") alert(d.message);
          fetchStatus();
        }});
      }}
    }}

    function applyPreset(name) {{
      post("/api/preset", {{name: name}}).then(() => fetchStatus());
    }}

    function resetScene() {{
      if (confirm("Xác nhận reset toàn bộ scene mô phỏng MuJoCo và vật thể về vị trí gốc?")) {{
        post("/api/reset_scene").then(() => fetchStatus());
      }}
    }}

    function setEESliderValue(name, value) {{
      const input = document.getElementById("ee_" + name);
      if (!input || !Number.isFinite(Number(value))) return;
      input.value = Number(value);
      onEESliderInput(name, input.value, false);
    }}

    function onEESliderInput(name, value, userInput = true) {{
      if (userInput) isDragging = true;
      const numeric = Number(value);
      const label = document.getElementById("val_ee_" + name);
      if (label && Number.isFinite(numeric)) label.innerText = numeric.toFixed(3);
    }}

    function stepEESlider(name, delta) {{
      const input = document.getElementById("ee_" + name);
      if (!input) return;
      const min = Number(input.min);
      const max = Number(input.max);
      const value = Math.min(max, Math.max(min, Number(input.value) + delta));
      input.value = value.toFixed(name === "x" || name === "y" || name === "z" ? 3 : 2);
      onEESliderInput(name, input.value);
      onEESliderCommit();
    }}

    function readEEPose() {{
      const ids = ["ee_x", "ee_y", "ee_z", "ee_qx", "ee_qy", "ee_qz", "ee_qw"];
      const values = ids.map(id => parseFloat(document.getElementById(id).value));
      if (values.some(v => !Number.isFinite(v))) {{
        showErr("Pose EE phai gom 7 so huu han.");
        return null;
      }}
      const qNorm = Math.hypot(values[3], values[4], values[5], values[6]);
      if (qNorm < 1e-9) {{
        showErr("Quaternion khong duoc la vector 0.");
        return null;
      }}
      return {{
        position: values.slice(0, 3),
        quaternion: values.slice(3).map(v => v / qNorm)
      }};
    }}

    function setEEPoseInputs(position, quaternion) {{
      if (!position || position.length !== 3 || !quaternion || quaternion.length !== 4) return;
      ["x", "y", "z"].forEach((name, i) => setEESliderValue(name, position[i]));
      ["qx", "qy", "qz", "qw"].forEach((name, i) => setEESliderValue(name, quaternion[i]));
    }}

    function updateEEPreviewInfo(data, prefix) {{
      const el = document.getElementById("eePreviewInfo");
      if (!el || !data) return;
      const p = (data.actual_position || data.position || []).map(v => Number(v).toFixed(3));
      const err = Number(data.position_error_m);
      el.innerText = prefix + " | EE: [" + p.join(", ") + "] m" +
                     (Number.isFinite(err) ? " | loi " + (err * 1000).toFixed(1) + " mm" : "");
    }}

    function onEESliderCommit() {{
      const pose = readEEPose();
      if (!pose) {{
        isDragging = false;
        return;
      }}
      pendingEE = {{pose: pose, until: Date.now() + 8000}};
      isDragging = false;
      post("/api/set_ee_absolute", pose).then(d => {{
        if (d && d.status === "ok") {{
          if (currentMode === "simulate") switchStream("third");
          updateEEPreviewInfo(d, currentMode === "simulate" ? "Ghost da cap nhat" : "Da gui set_tcp_pose");
        }} else {{
          pendingEE = null;
        }}
        fetchStatus();
      }});
    }}

    function fetchStatus() {{
      fetch("/api/status")
        .then(r => r.json())
        .then(data => {{
          if (data.ui_mode) {{
            if (data.ui_mode !== currentMode) {{
              pendingEE = null;
              isDragging = false;
            }}
            updateModeUI(data.ui_mode);
          }}
          if (data.control_tab && data.control_tab !== currentControlTab) {{
            // Sync the server session without posting again on every poll.
            renderControlTab(data.control_tab);
          }}

          // Robot that dang chay quy dao -> khoa nut "Dua robot that ve pose nay"
          // de tranh bam trung lam nhieu lenh chong nhau.
          const tj = data.trajectory || {{}};
          trajRunning = !!tj.running;
          updateApplyButton();
          // Loi tu thread FollowTrajectory (neu co) phai hien ra, khong im lang.
          if (tj.error) showErr("Lỗi di chuyển robot thật: " + tj.error);
          if (data.ee_preview) {{
            updateEEPreviewInfo(data.ee_preview, "Preview đang hiển thị");
          }}

          if (currentControlTab === "coordinate" && !isDragging) {{
            const ep = data.ee_control_pose || {{}};
            const p = ep.position;
            const q = ep.quaternion;
            if (p && p.length === 3 && q && q.length === 4) {{
              let holdPending = false;
              if (pendingEE) {{
                const target = pendingEE.pose.position;
                const targetQ = pendingEE.pose.quaternion;
                const error = Math.hypot(
                  Number(p[0]) - target[0],
                  Number(p[1]) - target[1],
                  Number(p[2]) - target[2]
                );
                const qError = Math.min(
                  Math.hypot(Number(q[0]) - targetQ[0], Number(q[1]) - targetQ[1],
                             Number(q[2]) - targetQ[2], Number(q[3]) - targetQ[3]),
                  Math.hypot(Number(q[0]) + targetQ[0], Number(q[1]) + targetQ[1],
                             Number(q[2]) + targetQ[2], Number(q[3]) + targetQ[3])
                );
                if (Date.now() <= pendingEE.until && (error > 0.005 || qError > 0.04)) {{
                  holdPending = true;
                }} else {{
                  pendingEE = null;
                }}
              }}
              if (holdPending) {{
                setEEPoseInputs(pendingEE.pose.position, pendingEE.pose.quaternion);
              }} else {{
                setEEPoseInputs(p, q);
              }}
            }}
          }}

          // Cap nhat TCP
          if (data.tcp_pos) {{
            document.getElementById("tcpCoord").innerText =
              "[" + data.tcp_pos.map(v => v.toFixed(3)).join(", ") + "] m";
          }}
          // Cap nhat Hardware status
          document.getElementById("hwStatus").innerText =
            data.is_real_connected ? "Robot Thật Online (COM)" : "Mô phỏng (MuJoCo Offline Hardware)";
          document.getElementById("hwStatus").style.color =
            data.is_real_connected ? "#3fb950" : "#8b949e";

          // Robot mat nguon/mat ket noi: bao RO rang thay vi de nguoi dung tuong
          // slider hong. Truoc day server còn TREO han khi gap truong hop nay.
          if (data.hardware_offline) {{
            showErr("⚠️ Robot KHÔNG PHẢN HỒI — kiểm tra nguồn điện và cáp USB. " +
                    "Sliders sẽ không tác dụng cho tới khi robot trả lời lại.");
          }}

          // Nhiet do servo: >60 do la servo sap tu ngat (da gap: 72 do -> dung het).
          const tEl = document.getElementById("tempStatus");
          const temps = data.temperatures;
          // Server gui null cho khop chua doc duoc (driver cu / backend fake).
          const known = (temps || []).filter(v => typeof v === "number" && isFinite(v));
          if (known.length) {{
            const hot = Math.max(...known);
            tEl.innerText = "[" + known.join(", ") + "] °C";
            tEl.style.color = hot >= 60 ? "#f85149" : (hot >= 50 ? "#d29922" : "#3fb950");
          }} else {{
            tEl.innerText = "không đọc được (chưa hỗ trợ / mất nguồn)";
            tEl.style.color = "#8b949e";
          }}

          // Trang thai an toan: sau DUNG KHAN la "fault_latched" va driver TU CHOI
          // moi quy dao. Phai hien ro, neu khong nguoi dung chi thay robot im lang.
          const sEl = document.getElementById("safetyBadge");
          if (sEl) {{
            const st = data.safety_state || "unknown";
            const armed = !!data.is_armed;
            sEl.innerText = armed ? (st + " — sẵn sàng") : (st + " — CẦN KHÔI PHỤC");
            sEl.style.color = armed ? "#3fb950" : "#d29922";
          }}

          if (!isDragging) {{
            const degs = data.active_deg || [];
            const nowMs = Date.now();
            for (let i = 0; i < 6; i++) {{
              if (degs[i] !== undefined) {{
                const sl = document.getElementById("slider_j" + i);
                const vl = document.getElementById("val_j" + i);
                const pend = pendingSlider[i];
                if (pend) {{
                  // Tay robot chua toi noi -> giu gia tri nguoi dung vua chon,
                  // neu khong slider se nhay ve cho cu.
                  if (Math.abs(degs[i] - pend.display) <= 3.0 || nowMs > pend.until) {{
                    delete pendingSlider[i];   // da toi (hoac qua lau) -> theo encoder
                  }} else {{
                    if (vl) vl.innerText = pend.display.toFixed(1) + "° (" +
                      (pend.display * Math.PI / 180.0).toFixed(2) + " rad)";
                    continue;
                  }}
                }}
                if (sl) sl.value = degs[i].toFixed(1);
                if (vl) vl.innerText = degs[i].toFixed(1) + "° (" + (degs[i] * Math.PI / 180.0).toFixed(2) + " rad)";
              }}
            }}
            const gPct = data.gripper_pct !== undefined ? data.gripper_pct : 0;
            const slGrip = document.getElementById("slider_grip");
            const vlGrip = document.getElementById("val_grip");
            const pendG = pendingSlider["gripper"];
            let gripHeld = false;
            if (pendG) {{
              if (Math.abs(gPct - pendG.display) <= 5.0 || nowMs > pendG.until) {{
                delete pendingSlider["gripper"];
              }} else {{
                gripHeld = true;
                if (vlGrip) vlGrip.innerText = pendG.display.toFixed(1) + "%";
                if (slGrip) slGrip.value = pendG.display.toFixed(0);
              }}
            }}
            if (!gripHeld) {{
              if (slGrip) slGrip.value = gPct.toFixed(0);
              if (vlGrip) vlGrip.innerText = gPct.toFixed(1) + "%";
            }}
          }}
        }});
    }}

    initSliders();
    fetchStatus();
    setInterval(fetchStatus, 400);
  </script>
</body>
</html>
"""
        return html.encode("utf-8")

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


if __name__ == "__main__":
    # Chay truc tiep: python web_control.py
    # Yeu cau stack ROS 2 dang chay (xem README): Robot/FakeRobot/Camera deu la
    # mat na ROS, khong tu mo serial hay thiet bi camera nua.
    print("[web_control.py] Khoi dong moi truong mac dinh...")
    # Scene nam canh file nay: user/robot_model/scene_vla.xml
    SCENE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "robot_model", "scene_vla.xml")

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

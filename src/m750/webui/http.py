"""HTTP/cookie/JSON/stream; nghiệp vụ được gọi qua server.app_context."""

import json
import time
from urllib.parse import urlparse
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

import numpy as np

from m750.ros import bridge as ros_bridge
from .streams import STREAM_KEYS


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
                ee_preview = dict(ctx._ee_preview) if ctx._ee_preview is not None else None
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
            serialized["temperatures"] = [None if not np.isfinite(t) else float(t) for t in _temps]

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
            self._send_json({"status": "ok" if result.get("ok") else "error", **result})

        elif p == "/api/move_ee":
            # Lenh that: goi Robot.set_tcp_pose() voi pose da validate.
            result = ctx.move_ee(body, sid)
            self._send_json({"status": "ok" if result.get("ok") else "error", **result})

        elif p == "/api/set_ee_absolute":
            # Giong slider khop: control -> robot that, simulate -> ghost IK.
            result = ctx.set_ee_absolute(body, sid)
            self._send_json({"status": "ok" if result.get("ok") else "error", **result})

        elif p == "/api/teleop/step":
            result = ctx.teleop_step(body, sid)
            self._send_json({"status": "ok" if result.get("ok") else "error", **result})

        elif p == "/api/teleop/reset":
            result = ctx.reset_teleop(sid)
            self._send_json({"status": "ok" if result.get("ok") else "error", **result})

        elif p == "/api/set_control_tab":
            tab = body.get("tab", "joint")
            ok = ctx.set_control_tab(sid, tab)
            self._send_json({"status": "ok" if ok else "error", "tab": ctx.get_control_tab(sid)})

        elif p == "/api/set_mode":
            # Chuyen doi mode: 'control', 'simulate' hoac 'offset' (chi doi cho tab nay)
            new_mode = body.get("mode", "control")
            success = ctx.set_ui_mode(sid, new_mode)
            self._send_json(
                {"status": "ok" if success else "error", "mode": ctx.get_session_mode(sid)}
            )

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
            print(
                "[WebControl] Da luu offset moi (deg): %s  (doi: %s)"
                % ([round(v, 2) for v in new], list(doi))
            )
            self._send_json({"status": "ok", "offsets_deg": list(new), "hieu_chinh_deg": list(doi)})

        elif p == "/api/apply_to_real":
            # Option trong simulate: 'Dua robot that ve pose nay' -> goi FollowTrajectory
            result = ctx.apply_fake_pose_to_real(sid, body.get("control_tab"))
            self._send_json({"status": "ok" if result.get("ok") else "error", **result})

        elif p == "/api/preset":
            # Ap dung cac pose preset mac dinh
            preset_name = body.get("name", "home")
            success = ctx.apply_preset(preset_name, sid)
            self._send_json({"status": "ok" if success else "error"})

        elif p == "/api/reset_scene":
            # Reset toan bo scene mo phong
            success = ctx.reset_scene()
            self._send_json(
                {
                    "status": "ok" if success else "error",
                    "message": "Da reset scene ve vi tri ban dau!",
                }
            )

        elif p == "/api/stop":
            # Dung khan cap: KHONG doi quyen, ai cung bam duoc. Dung ngay tai cho.
            ctx.stop()
            print(f"[WebControl] DUNG KHAN CAP boi '{name}'.")
            self._send_json({"status": "ok", "message": "Da dung robot!"})

        elif p == "/api/power_off":
            # Ngat mo-men roi cat dien. release truoc de servo thoi giu vi tri
            # (chi power_off thi servo van giu vat va tiep tuc nong).
            ok = ctx.power_off()
            self._send_json(
                {
                    "status": "ok" if ok else "error",
                    "message": "Da tat nguon robot!" if ok else "Tat nguon that bai.",
                }
            )

        elif p == "/api/power_on":
            # Bat lai dien + mo-men. Sau khi tat nguon, servo mat mo-men nen tay co
            # the da tut xuong; KHONG tu di chuyen, chi cap dien lai thoi.
            ok = ctx.power_on()
            self._send_json(
                {
                    "status": "ok" if ok else "error",
                    "message": "Da bat nguon robot!" if ok else "Bat nguon that bai.",
                }
            )

        elif p == "/api/rearm":
            # Khoi phuc sau DUNG KHAN. Driver chot fault_latched va tu choi moi
            # quy dao; khong co buoc nay thi bam DUNG KHAN xong la het duong lam
            # tiep, phai khoi dong lai ca stack.
            ok = ctx.rearm()
            self._send_json(
                {
                    "status": "ok" if ok else "error",
                    "message": "Da khoi phuc (armed)!" if ok else "Khoi phuc that bai.",
                    "safety_state": ctx.real_robot.safety_state,
                }
            )

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

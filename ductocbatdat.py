"""Interface ca nhan myArm M750 - goi thang API pymycobot, khong viet lai.

Chay tren server ktmt (port /dev/ttyACM1, baud 1000000 - mac dinh 115200 sai).
"""
from glob import glob
from pymycobot import MyArmMControl
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
import threading
import time

PORT, BAUDRATE = "/dev/ttyACM1", 1000000  # baud mac dinh 115200 -> khong noi duoc voi tay
CAM = "/dev/v4l/by-id/usb-046d_Logitech_Webcam_C925e_3F4C8F2F-video-index0"  # by-id: so video* doi theo thu tu cam
FW_MIN = (-165, -80, -100, -160, -90, -180)  # gioi han firmware, khac URDF (q2=100, q3=-100)
FW_MAX = (165, 100, 80, 160, 120, 180)       # firmware chan ca goi lenh neu vuot, khong bao loi
MARGIN = 1.5  # kep sat gioi han -> servo gong, dung yen (do that: q2 kep 100 dung, 99 chay)

arm = None  # mo o _open() ben duoi, sau khi da kiem tra port

URDF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "robot model", "myarm_m750_full.urdf")
_ARM = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_flex_joint",
        "forearm_roll_joint", "wrist_flex_joint", "wrist_roll_joint")


def _model():
    """Nap URDF 1 lan (Pinocchio nang): tra (model, data, idx, gioi_han_duoi, gioi_han_tren).
    URDF la nguon gioi han chinh - tai lieu lab muc 7.5 (get_joint_max() khong dang tin)."""
    import numpy as np
    import pinocchio as pin
    if "m" not in _model.__dict__:
        m = pin.buildModelFromUrdf(URDF)
        idx = [int(m.joints[m.getJointId(n)].idx_q) for n in _ARM]
        _model.m = m
        _model.d = m.createData()
        _model.idx = idx
        _model.lo = [float(np.degrees(m.lowerPositionLimit[i])) for i in idx]
        _model.hi = [float(np.degrees(m.upperPositionLimit[i])) for i in idx]
    return _model.m, _model.d, _model.idx, _model.lo, _model.hi


def _held(port=PORT):
    out, port = [], os.path.realpath(port)
    for fd in glob("/proc/[0-9]*/fd/*"):
        try:
            if os.readlink(fd) == port:
                pid = fd.split("/")[2]
                with open(f"/proc/{pid}/cmdline", "rb") as f:
                    cmd = f.read().replace(b"\0", b" ").decode(errors="replace")
                out.append(f"pid {pid}  {cmd[:70]}")
        except OSError:  # tiến trình khác user hoặc đã thoát
            pass
    return out

def _open(port=PORT):
    global arm
    if arm is not None:
        return arm  # da mo roi: bo qua chot chan, neu khong chinh minh bi coi la ke chiem port
    if held := _held(port):
        # pymycobot KHONG bat exclusive: mo chong len van 'thanh cong' nhung 2 ben cung ghi -> lenh hong
        raise RuntimeError(f"{port} đang bị giữ: {', '.join(held)}. Tắt tiến trình/ROS trước.")
    arm = MyArmMControl(port, baudrate=BAUDRATE)
    return arm



def power_on(wait_s=2.5):
    """Bat dien servo, tra True/False that. power_on() tra -1 KE CA khi thanh cong -> phai doc lai is_powered_on()."""
    _open().power_on()
    time.sleep(wait_s)
    return arm.is_powered_on() == 1


def release_all_servos(data=None):
    """CAT MO-MEN -> tay ROT theo trong luc. Do tay/ke chan truoc. Co mo-men lai: power_on()."""
    _open().release_all_servos(data)


def temperature():
    """Nhiet do 6 servo (do C, 0..255) - xem truoc khi chay lien tuc. Tra None neu loi/doc khong ra."""
    t = _open().get_servo_temps()
    return t if isinstance(t, list) and len(t) == 6 else None


def _fk(qd):
    """FK toi tool0: 6 goc khop (do) -> pin.SE3."""
    import numpy as np
    import pinocchio as pin
    m, d, idx, _, _ = _model()
    q = pin.neutral(m)
    for i, ix in enumerate(idx):
        q[ix] = float(np.radians(qd[i]))
    pin.forwardKinematics(m, d, q); pin.updateFramePlacements(m, d)
    return d.oMf[m.getFrameId("tool0")]


def _solve_ik(T_target, q_now, n_restart=15):
    """IK 6 DOF thuan toan hoc (khong mo port -> test duoc). Tra (q_goal_do, ep_mm, eo_do).

    ep/eo la sai so THAT sau khi giai - phai kiem tra ca hai, vi bo giai co the khop
    vi tri nhung KHONG xoay duoc co tay theo huong yeu cau (pose ngoai vung lam viec).
    """
    import numpy as np
    import pinocchio as pin
    from scipy.spatial.transform import Rotation as R
    from scipy.optimize import least_squares
    _, _, _, LO, HI = _model()
    Rg = T_target.rotation

    def res(qd):
        # e[:3] la MET, e[3:] la RADIAN -> nhan 1000 de cung thang do voi nhau,
        # neu khong sai so vi tri (m) bi be so voi sai so huong (rad).
        e = pin.log6(_fk(qd).actInv(T_target)).vector
        return np.concatenate([e[:3] * 1000.0, e[3:]])

    rng = np.random.default_rng(0)
    best = (None, 1e18, 1e18, 1e18)
    for k in range(n_restart):
        seed = np.array(q_now, dtype=float) if k == 0 else rng.uniform(LO, HI)
        if k == 0 and not all(LO[i] <= seed[i] <= HI[i] for i in range(6)):
            seed = rng.uniform(LO, HI)
        try:
            r = least_squares(res, seed, bounds=(LO, HI), max_nfev=3000)
        except Exception:
            continue
        qs = [float(v) for v in r.x]
        Tf = _fk(qs)
        ep = float(np.linalg.norm(Tf.translation * 1000.0 - T_target.translation * 1000.0))
        eo = float(np.degrees(np.linalg.norm(
            R.from_matrix(Tf.rotation.T @ Rg).as_rotvec())))
        if ep + eo < best[1]:
            best = (qs, ep + eo, ep, eo)
        if ep < 0.05 and eo < 0.05:
            break
    return best[0], best[2], best[3]


GRIP_L = 45.74  # mm: tu tool0 toi diem gripper, do thuc nghiem (lech chuan 0.75mm qua 9 tu the)


def set_tcp_pose(coords, speed, tol=2.0, timeout_s=30):
    """TCP [x,y,z mm, rx,ry,rz do Euler] -> IK (Pinocchio+URDF) -> write_angles. Tra True neu toi noi.

    KHONG dung write_coords: firmware IK cua vendor lech URDF 50mm, sai so do that 17-106mm,
    tai lieu lab (myarm_m750_pymycobot_api.md muc 8.1) loai API Cartesian khoi driver.
    write_angles do that bam chinh xac e_max < 0.6 do.
    IK khong giai duoc (ngoai tam voi / vuot gioi han khop) -> tra False, KHONG chay gi.
    """
    import numpy as np
    import pinocchio as pin
    from scipy.spatial.transform import Rotation as R
    a = _open()
    q_now = a.get_angles()
    if not isinstance(q_now, list) or len(q_now) != 6 or -1 in q_now:
        print("doc goc khop loi:", q_now, "-> khong chay gi")
        return False
    q_goal, ep, eo = _solve_ik(
        pin.SE3(R.from_euler("xyz", coords[3:], degrees=True).as_matrix(),
                np.array(coords[:3], dtype=float) / 1000.0), q_now)
    if q_goal is None or ep > tol or eo > tol:
        print("POSE KHONG TOI DUOC: vi tri lech %.2f mm, huong lech %.2f do -> khong chay gi"
              % (ep, eo))
        return False
    a.write_angles(q_goal, speed)
    deadline = time.time() + timeout_s
    while time.time() < deadline and a.is_moving() == 1:
        time.sleep(0.05)
    time.sleep(0.5)
    e = max(abs(x - y) for x, y in zip(a.get_angles(), q_goal))
    print("e_max khop %.2f do (cho phep %.1f): %s" % (e, tol, "TOI" if e <= tol else "LECH"))
    state()
    return e <= tol


def set_gripper_pose(x, y, z, rx=180.0, ry=0.0, rz=0.0, speed=20,
                     tol_pos=1.0, tol_rot=1.0, timeout_s=30, drop_max=30.0):
    """Dua GRIPPER toi pose [x,y,z mm | rx,ry,rz do Euler XYZ] trong he URDF. Tra True/False.

    Gripper = tool0 + 45.74mm doc truc z. rx=180 -> chuc thang xuong.
    NHIEU POSE KHONG TOI DUOC: chuc thang xuong gan de thuong vuot wrist_flex [-90..120].
    Giai khong ra (vi tri HOAC huong lech qua tol) -> tra False, KHONG chay gi.
    Kiem tra them duong di: noi suy goc khop, neu tut qua drop_max so voi ca hai dau -> chan.
    """
    import numpy as np
    import pinocchio as pin
    from scipy.spatial.transform import Rotation as R

    a = _open()
    q_now = a.get_angles()
    if not isinstance(q_now, list) or len(q_now) != 6 or -1 in q_now:
        print("doc goc khop loi:", q_now, "-> khong chay gi")
        return False

    Rg = R.from_euler("xyz", [rx, ry, rz], degrees=True).as_matrix()
    # dich cho gripper -> quy ve tool0 (bo offset 45.74mm doc truc z cua tool0)
    p_tool = np.array([x, y, z], dtype=float) - Rg @ np.array([0.0, 0.0, GRIP_L])

    q_goal, ep, eo = _solve_ik(pin.SE3(Rg, p_tool / 1000.0), q_now)
    if q_goal is None or ep > tol_pos or eo > tol_rot:
        print("POSE KHONG TOI DUOC: vi tri lech %.2f mm (cho phep %.1f),"
              " huong lech %.2f do (cho phep %.1f) -> khong chay gi"
              % (ep, tol_pos, eo, tol_rot))
        return False

    # duong di: tay noi suy theo GOC KHOP, kiem tra co tut qua sau khong
    off = np.array([0.0, 0.0, GRIP_L])
    zs = []
    for t in np.linspace(0.0, 1.0, 11):
        Mt = _fk([p + t * (g - p) for p, g in zip(q_now, q_goal)])
        zs.append(float((Mt.translation * 1000.0 + Mt.rotation @ off)[2]))
    z_min_ok = min(zs[0], zs[-1]) - drop_max
    if min(zs) < z_min_ok:
        print("DUONG DI TUT QUA: xuong %.1f mm, gioi han %.1f -> khong chay gi"
              % (min(zs), z_min_ok))
        return False

    a.write_angles(q_goal, speed)
    deadline = time.time() + timeout_s
    while time.time() < deadline and a.is_moving() == 1:
        time.sleep(0.05)
    time.sleep(0.5)
    Mt = _fk(a.get_angles())
    p_end = Mt.translation * 1000.0 + Mt.rotation @ off
    err = float(np.linalg.norm(p_end - np.array([x, y, z], dtype=float)))
    e = max(abs(p - g) for p, g in zip(a.get_angles(), q_goal))
    print("gripper (URDF): %s  | dich: %s" % (np.round(p_end, 1), [round(v, 1) for v in (x, y, z)]))
    print("e_max khop %.2f do | sai so gripper %.2f mm" % (e, err))
    return e <= 2.0


def set_joints(angles, speed=30, wait=True, timeout_s=30):
    """Dat 6 khop [q1..q6] do, DI CHUYEN THAT. write_angles tu chan ngoai FW_MIN/FW_MAX. wait=True cho toi khi dung han."""
    a = _open()
    a.write_angles(list(angles), speed)
    if not wait:
        return True
    deadline = time.time() + timeout_s
    while time.time() < deadline and a.is_moving() == 1:  # -1 = loi doc, thoat ngay thay vi treo het timeout
        time.sleep(0.05)
    return a.is_moving() == 0


def set_joint(joint_id, degree, speed=30, wait=True, timeout_s=30):
    """Doi 1 khop (1..6) sang goc moi, 5 khop kia GIU NGUYEN. DI CHUYEN THAT. joint_id ngoai 1..6 -> loi ngay."""
    a = _open()
    a.write_angle(int(joint_id), degree, speed)
    if not wait:
        return True
    deadline = time.time() + timeout_s
    while time.time() < deadline and a.is_moving() == 1:  # -1 = loi doc, thoat ngay thay vi treo het timeout
        time.sleep(0.05)
    return a.is_moving() == 0


def state():
    """In 6 khop (do), do mo gripper (0..100), TCP [x,y,z mm + rx,ry,rz do Euler]. -1 = chua bat dien/loi doc."""
    a = _open()
    print("khop  :", [round(x, 2) for x in a.get_angles()])
    print("grip  :", a.get_gripper_value())
    print("tcp   :", [round(x, 2) for x in a.get_coords()])


def port_free(port=PORT):
    """True neu khong ai giu port - kiem tra TRUOC khi mo, vi mo chong len khong bao loi nhung lenh se hong."""
    return not _held(port)


def camera_stream(port=8080, device=CAM, fps=25, quality=75, width=1280, height=720):
    """MJPEG tai http://<ip>:<port>/. PHAI goi .stop() truoc khi thoat, khong thi core dump."""
    import cv2
    jpg, lock, run = [b""], threading.Lock(), threading.Event()

    def grab():  # thread rieng: cap.read() CHAN khi camera rut ra, goi chung thread render la treo ca stream
        c = cv2.VideoCapture(device, cv2.CAP_V4L2)
        c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # YUYV ton CPU gap 3
        c.set(cv2.CAP_PROP_FRAME_WIDTH, width); c.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        c.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # hang doi 1 khung: khong doc lai anh cu
        while not run.is_set():
            ok, im = c.read()
            if not ok:
                time.sleep(0.5); continue  # rut camera: cho, khong quay CPU
            with lock:
                jpg[0] = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()
        c.release()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass  # tat log tung request, khong thi stdout ngap

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            last = None
            while True:
                with lock: j = jpg[0]
                if j and j is not last:  # khung moi moi gui, khung cu bo qua
                    last = j
                    try: self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + j + b"\r\n")
                    except (BrokenPipeError, ConnectionResetError): return  # nguoi xem dong tab
                time.sleep(1.0 / fps)

    srv = ThreadingHTTPServer(("0.0.0.0", port), H)
    srv.daemon_threads = True
    t = threading.Thread(target=grab, daemon=True); t.start()
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    # stop() phai JOIN thread grab: no con song luc Python thoat, OpenCV dang chan trong cap.read()
    # (C++) -> "terminate called without an active exception" -> core dump. Da gap that.
    srv.stop = lambda: (run.set(), t.join(3), srv.shutdown(), srv.server_close())
    print("camera: http://0.0.0.0:%d/" % port)
    return srv


"""Interface ca nhan myArm M750 - goi thang API pymycobot, khong viet lai.

Chay tren server ktmt (port /dev/ttyACM1, baud 1000000 - mac dinh 115200 sai).
"""
from glob import glob
from pymycobot import MyArmMControl
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import atexit
import os
import threading
import time

PORT, BAUDRATE = "/dev/ttyACM1", 1000000  # baud mac dinh 115200 -> khong noi duoc voi tay
os.environ.setdefault("MUJOCO_GL", "egl")  # Jetson khong X -> EGL; phai dat TRUOC khi nap mujoco
# aarch64: libgomp phai vao static TLS block TRUOC cac thu vien khac, neu khong se bao
# "libgomp.so.1: cannot allocate memory in static TLS block" khi nap cv2/mujoco. Da gap that.
try:
    import ctypes
    ctypes.CDLL("libgomp.so.1", mode=ctypes.RTLD_GLOBAL)
except OSError:
    pass
try:  # nap mujoco NGAY tai day: _solve_ik -> scipy.optimize keo theo 1 ban EGL khac,
    import mujoco  # nap sau se hong (eglQueryString = None). Da gap that, xem preview().
except Exception:  # may khong co mujoco thi preview() bao loi ro rang, phan con lai van chay
    mujoco = None
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


def _grip(M):
    """pin.SE3 cua tool0 -> vi tri gripper (mm, he URDF)."""
    import numpy as np
    return M.translation * 1000.0 + M.rotation @ np.array([0.0, 0.0, GRIP_L])


def _ok6(q):
    """True neu doc ra du 6 goc khop. Doc serial thinh thoang tra ve int -> phai kiem tra truoc khi dung."""
    return isinstance(q, list) and len(q) == 6 and -1 not in q


def _q_now():
    """Doc 6 goc khop hien tai (do). Tra None neu loi doc."""
    q = _open().get_angles()
    if not _ok6(q):
        print("doc goc khop loi:", q, "-> khong chay gi")
        return None
    return q


def _wait_stop(timeout_s=30, quiet_s=0.5, start_s=2.0):
    """Cho tay dung HAN that: True neu dung yen, False neu qua timeout hoac doc goc loi.

    KHONG dung is_moving(): no tra 0 CA KHI tay dang quay -> lenh ke tiep de len lenh truoc
    -> servo qua tai. Da gap that: gripper over-current, firmware chot loi va chan MOI
    chuyen dong cho toi khi cycle nguon (power_off + power_on moi xoa duoc, stop() khong xoa).
    Phai cho qua start_s: lan doc dau (tay chua kip khoi dong) trong nhu da dung yen.
    """
    a, t0, last, quiet, started = _open(), time.time(), None, 0.0, False
    while time.time() - t0 < timeout_s:
        q = a.get_angles()
        if not _ok6(q):
            return False
        if last is not None:
            if max(abs(x - y) for x, y in zip(q, last)) > 0.05:
                started, quiet = True, 0.0
            elif started:
                quiet += 0.1
                if quiet >= quiet_s:
                    return True
            elif time.time() - t0 > start_s:
                return True  # khong nhuc nhich: da o dich, hoac bi chan
        last = q
        time.sleep(0.1)
    return False


def _move(target, speed=20, timeout_s=30, tol=1.0, tries=4):
    """Ghi 6 goc khop roi DOC LAI kiem tra. Tra True neu moi khop |q - target| <= tol.

    Phai kiem tra lai vi do that tren tay nay (khong phai loi doan):
      (1) firmware THINH THOANG NUOT LENH - write_angles tra ack=1 nhung khop dung yen, ~1/10 lenh;
      (2) servo DUNG NON ~1 do so voi lenh -> o tam voi 470mm thanh ~20mm sai so vi tri.
    Lenh bi nuot -> gui lai y nguyen. Dung non -> bu dan, nhung chan bu trong +/-3 do
    (dang ti vao vat thi dung day mai).
    """
    a, target = _open(), [float(v) for v in target]
    cmd = list(target)
    for _ in range(tries):
        q0 = a.get_angles()  # doc truoc khi ghi: de biet lenh co bi nuot khong
        a.write_angles(cmd, speed)
        _wait_stop(timeout_s)
        q = a.get_angles()
        if not _ok6(q):
            continue
        err = [t - v for t, v in zip(target, q)]
        if max(abs(e) for e in err) <= tol:
            return True
        if _ok6(q0) and max(abs(v - w) for v, w in zip(q, q0)) < 0.05:
            continue  # khong nhuc nhich: lenh bi nuot, gui lai y nguyen
        cmd = [t + max(-3.0, min(3.0, c + e - t)) for t, c, e in zip(target, cmd, err)]
    q = a.get_angles()
    return _ok6(q) and max(abs(t - v) for t, v in zip(target, q)) <= tol


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


# mm: tool0 -> tam gap (gripper_base_link), lay tu URDF chu khong do firmware.
# Huong ra ngoai cua tool la -z (flange o +118, tool0 o 0, dau ngon o +74.5..+99.5).
# test_gripper_pose.py kiem tra lai hang so nay voi URDF.
GRIP_L = 87.0


def get_gripper_pose():
    """Pose gripper hien tai [x,y,z mm | rx,ry,rz do Euler XYZ], he URDF.

    CUNG quy uoc voi set_gripper_pose -> doc roi ghi lai duoc:
        set_gripper_pose(*get_gripper_pose())   # khong di chuyen
    rpy=[0,0,0] nghia la gripper CHUC THANG XUONG. Tra None neu loi doc goc khop.
    """
    import numpy as np
    from scipy.spatial.transform import Rotation as R
    q = _q_now()
    if q is None:
        return None
    M = _fk(q)
    return ([round(float(v), 2) for v in _grip(M)]
            + [round(float(v), 2) for v in R.from_matrix(M.rotation).as_euler("xyz", degrees=True)])


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
    q_now = _q_now()
    if q_now is None:
        return False
    q_goal, ep, eo = _solve_ik(
        pin.SE3(R.from_euler("xyz", coords[3:], degrees=True).as_matrix(),
                np.array(coords[:3], dtype=float) / 1000.0), q_now)
    if q_goal is None or ep > tol or eo > tol:
        print("POSE KHONG TOI DUOC: vi tri lech %.2f mm, huong lech %.2f do -> khong chay gi"
              % (ep, eo))
        return False
    a = _open()
    _move(q_goal, speed, timeout_s, tol=tol)
    q_end = a.get_angles()
    if not _ok6(q_end):
        print("doc goc khop loi sau khi chay")
        return False
    M = _fk(q_end)
    ep = float(np.linalg.norm(M.translation * 1000.0 - np.array(coords[:3], dtype=float)))
    eo = float(np.degrees(np.linalg.norm(R.from_matrix(
        M.rotation.T @ R.from_euler("xyz", coords[3:], degrees=True).as_matrix()).as_rotvec())))
    print("tool0 lech %.2f mm / %.2f do (cho phep %.1f): %s"
          % (ep, eo, tol, "TOI" if ep <= tol and eo <= tol else "LECH"))
    state()
    return ep <= tol and eo <= tol


def set_gripper_pose(x, y, z, rx=0.0, ry=0.0, rz=0.0, speed=20,
                     tol_pos=1.0, tol_rot=1.0, timeout_s=30, drop_max=30.0):
    """Dua GRIPPER toi pose [x,y,z mm | rx,ry,rz do Euler XYZ] trong he URDF. Tra True/False.

    Gripper = tam gap (gripper_base_link), cach tool0 87mm. Huong ra ngoai cua tool la
    -z cua tool0, nen rpy=[0,0,0] la CHUC THANG XUONG (da kiem chung 2 cach: truc -z cua
    tool0 va vector flange->gripper_base deu = [0,0,-1] khi rpy=0).
    rpy=[180,0,0] la chuc thang LEN.
    Nghieng: rx=180 khong phai chuc xuong. Muon nghieng thi doi ry (vd ry=30 -> nghieng 30 do).
    Giai khong ra (vi tri HOAC huong lech qua tol) -> tra False, KHONG chay gi.
    Kiem tra them duong di: noi suy goc khop, neu tut qua drop_max so voi ca hai dau -> chan.
    """
    import numpy as np
    import pinocchio as pin
    from scipy.spatial.transform import Rotation as R

    q_now = _q_now()
    if q_now is None:
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
    zs = [float(_grip(_fk([p + t * (g - p) for p, g in zip(q_now, q_goal)]))[2])
          for t in np.linspace(0.0, 1.0, 11)]
    z_min_ok = min(zs[0], zs[-1]) - drop_max
    if min(zs) < z_min_ok:
        print("DUONG DI TUT QUA: xuong %.1f mm, gioi han %.1f -> khong chay gi"
              % (min(zs), z_min_ok))
        return False

    _move(q_goal, speed, timeout_s, tol=0.15, tries=6)  # 0.15 do x 470mm ~ 1.2mm: phai chat moi duoi tol_pos
    q_end = _open().get_angles()
    if not _ok6(q_end):
        print("doc goc khop loi sau khi chay")
        return False
    M = _fk(q_end)
    p_end = _grip(M)
    err = float(np.linalg.norm(p_end - np.array([x, y, z], dtype=float)))
    eo = float(np.degrees(np.linalg.norm(
        R.from_matrix(M.rotation.T @ Rg).as_rotvec())))
    print("gripper (URDF): %s  | dich: %s" % (np.round(p_end, 1), [round(v, 1) for v in (x, y, z)]))
    print("sai so gripper %.2f mm (cho phep %.1f) | huong lech %.2f do (cho phep %.1f): %s"
          % (err, tol_pos, eo, tol_rot, "TOI" if err <= tol_pos and eo <= tol_rot else "LECH"))
    return err <= tol_pos and eo <= tol_rot


def views_6(r=150, theta=30.0, center=None):
    """6 pose quanh tam, theo GraspNeRF muc IV-B: ban cau ban kinh r, goc cuc theta,
    6 phuong vi chia deu 60 do, moi pose chuc thang vao tam. Tra list 6 pose
    [x,y,z,rx,ry,rz] dung quy uoc set_gripper_pose -> truyen thang vao duoc.

    center mac dinh = gripper hien tai. r la mm tu TAM toi TAM GAP (gripper_base):
    camera khong nam dung tam gap thi tru offset cua no vao r.
    Tran DO THAT tren tay nay voi tam x=449: theta=20 -> r<=225, theta=30 -> r<=175,
    theta=40 -> r<150. Paper dung r=500m KHONG toi duoc (khuyu phai ra ~715mm > ~580mm).
    Tra None neu doc goc khop loi.
    """
    import numpy as np
    from scipy.spatial.transform import Rotation as R
    if center is None:
        p = get_gripper_pose()
        if p is None:
            return None
        center = p[:3]
    center = np.array(center, dtype=float)
    up, th = np.array([0.0, 0.0, 1.0]), np.radians(theta)
    out = []
    for k in range(6):
        ph = np.radians(60.0 * k)
        u = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])  # tam -> camera
        yv = up - u * (up @ u)  # tool0 +y gan voi phuong dung nhat; roll quanh truc nhin la tu do
        yv = yv / np.linalg.norm(yv)
        M = np.column_stack([np.cross(yv, u), yv, u])  # cot 3 = u = -huong nhin (truc +z cua tool0)
        out.append([round(float(v), 2) for v in center + r * u]
                   + [round(float(v), 2) for v in R.from_matrix(M).as_euler("xyz", degrees=True)])
    return out


def set_joints(angles, speed=30, wait=True, timeout_s=30):
    """Dat 6 khop [q1..q6] do, DI CHUYEN THAT. write_angles tu chan ngoai FW_MIN/FW_MAX.
    wait=True: cho dung han roi DOC LAI, gui lai neu bi nuot lenh, bu neu dung non."""
    if not wait:
        _open().write_angles(list(angles), speed)
        return True
    return _move(angles, speed, timeout_s)


def set_joint(joint_id, degree, speed=30, wait=True, timeout_s=30):
    """Doi 1 khop (1..6) sang goc moi, 5 khop kia GIU NGUYEN. DI CHUYEN THAT. joint_id ngoai 1..6 -> loi ngay."""
    if not 1 <= int(joint_id) <= 6:
        raise ValueError("joint_id phai trong 1..6, nhan duoc %r" % (joint_id,))
    q = _q_now()
    if q is None:
        return False
    q[int(joint_id) - 1] = float(degree)
    if not wait:
        _open().write_angles(q, speed)
        return True
    return _move(q, speed, timeout_s)


def state():
    """In 6 khop (do), do mo gripper (0..100), pose gripper URDF va toa do firmware.

    'gripper' la con so dung duoc: cung quy uoc voi get/set_gripper_pose -> truyen thang
    duoc vao set_gripper_pose. 'fw' la toa do firmware vendor, KHONG dung cho IK
    (khac he quy chieu). -1 = chua bat dien/loi doc.
    """
    a = _open()
    print("khop   :", [round(x, 2) for x in a.get_angles()])
    print("grip   :", a.get_gripper_value())
    print("gripper:", get_gripper_pose(), " <- he URDF, dung duoc")
    print("fw     :", [round(x, 2) for x in a.get_coords()], " <- firmware vendor")


def port_free(port=PORT):
    """True neu khong ai giu port - kiem tra TRUOC khi mo, vi mo chong len khong bao loi nhung lenh se hong."""
    return not _held(port)


def capture(path="cap_img.jpg", device=CAM, width=1280, height=720, tries=5):
    """Chup 1 khung MOI NHAT, ghi DE len file (mac dinh cap_img.jpg). Tra duong dan hoac None.

    Mo camera roi dong ngay: de tuoi khung phai bo vai khung dau (V4L2 giu buffer cu),
    nen doc bo tries khung va lay khung cuoi. Khong dung chung voi camera_stream().
    """
    import cv2
    c = cv2.VideoCapture(device, cv2.CAP_V4L2)
    c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    c.set(cv2.CAP_PROP_FRAME_WIDTH, width); c.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    try:
        for _ in range(tries):
            ok, im = c.read()
        if not ok:
            print("khong doc duoc khung nao tu", device)
            return None
        cv2.imwrite(path, im)
        print("ghi", os.path.abspath(path), im.shape[1], "x", im.shape[0])
        return path
    finally:
        c.release()  # phai tra camera, khong thi lan goi sau khong mo duoc


_PAGE = """<!doctype html><meta charset=utf-8><title>myArm M750 preview</title>
<style>
 body{margin:0;background:#111;color:#ddd;font:13px system-ui;overflow:hidden}
 #v{display:block;width:100vw;height:100vh;cursor:grab}
 #v.drag{cursor:grabbing}
 #ui{position:fixed;top:10px;left:10px;background:#000b;padding:10px 12px;border-radius:8px;width:250px}
 #ui h3{margin:0 0 8px;font-size:13px;font-weight:600}
 .row{display:flex;align-items:center;gap:6px;margin:3px 0}
 .row label{width:22px;color:#8ac}
 .row input[type=range]{flex:1;min-width:0}
 .row span{width:58px;text-align:right;font-variant-numeric:tabular-nums;color:#eee}
 .num{width:62px;background:#1a222a;color:#eee;border:1px solid #456;border-radius:4px;
      padding:2px 4px;font:12px ui-monospace,monospace;text-align:right}
 .num:focus{outline:1px solid #4caf50;border-color:#4caf50}
 .num.bad{border-color:#c62828;color:#ff8a80}
 button{background:#2a3a4a;color:#ddd;border:1px solid #456;border-radius:5px;
        padding:5px 10px;cursor:pointer;margin-right:6px;font-size:12px}
 button.on{background:#2e7d32;border-color:#4caf50;color:#fff}
 button:disabled{opacity:.45;cursor:default}
 #st{position:fixed;bottom:10px;left:10px;background:#000b;padding:6px 10px;
     border-radius:6px;font-variant-numeric:tabular-nums;max-width:70vw}
</style>
<canvas id=v></canvas>
<div id=ui>
  <h3>myArm M750</h3>
  <div id=sl></div>
  <div style="margin-top:8px">
    <button id=rt>realtime</button><button id=sync>sync</button><button id=home>ve 0</button>
  </div>
</div>
<div id=st>dang tai...</div>
<script>
const AX=[['x',.3,700,'mm'],['y',-500,500,'mm'],['z',-100,700,'mm'],
          ['rx',-180,180,'do'],['ry',-180,180,'do'],['rz',-180,180,'do']];
const sl=document.getElementById('sl'),st=document.getElementById('st'),cv=document.getElementById('v');
const vals=[500,0,250,0,0,0];
let realtime=false,busy=false,cam={az:135,el:-20,d:1.1,pan:[0,0,0]};
AX.forEach(([n,lo,hi,u],i)=>{
  sl.insertAdjacentHTML('beforeend',
    `<div class=row><label>${n}</label><input type=range id=s${i} min=${lo} max=${hi} step=1 value=${vals[i]}>`
   +`<input type=number id=t${i} class=num step=1 value=${vals[i]}></div>`);
});

function draw(im){                           // im phai la Image DA DECODE, khong phai Blob:
  cv.width=im.width;cv.height=im.height;     // Blob -> drawImage ve ra canvas TRONG SUOT ->
  cv.getContext('2d').drawImage(im,0,0)      // chi thay mau nen #111 = den thui. Da gap that.
}
function decode(b){return new Promise((ok,no)=>{  // Blob -> Image
  const u=URL.createObjectURL(b),i=new Image();
  i.onload=()=>{URL.revokeObjectURL(u);ok(i)};i.onerror=no;i.src=u})}
async function frame(){                      // lay 1 khung tu server, tra Image hoac null
  const r=await fetch('render',{method:'POST',body:JSON.stringify({pos:vals,cam:cam})});
  if(!r.ok)return null;                      // 422 = ngoai tam voi, giu khung cu
  return decode(await r.blob());
}

async function tick(){                       // vong lap 1 khung/lan: khong chong len nhau
  if(busy){setTimeout(tick,20);return}
  busy=true;
  let note=msg();
  try{
    if(realtime){                            // realtime: hoi thang goc khop that tu driver
      const r=await fetch('state');
      if(r.ok){const j=await r.json();
        if(j.grip){for(let i=0;i<6;i++)setVal(i,j.grip[i])}
        else note='REALTIME KHONG DOC DUOC: '+(j.err||'?')}   // giu nguyen, khong bi msg() de
    }
    if(dirty||realtime){                     // simulate: chi ve khi slider doi
      const im=await frame();if(im)draw(im);dirty=false
    }
  }catch(e){note='loi: '+e.message}
  st.textContent=note;
  busy=false;setTimeout(tick,realtime?50:200);
}
function msg(){return `gripper  x=${vals[0].toFixed(1)} y=${vals[1].toFixed(1)} z=${vals[2].toFixed(1)}`
  +`  |  rpy=${vals[3].toFixed(1)},${vals[4].toFixed(1)},${vals[5].toFixed(1)}`
  +`  |  ${realtime?'REALTIME':'simulate'}`}

let dirty=true;
function setVal(i,v,syncSlider){              // 1 cho duy nhat ghi gia tri -> slider va o nhap luon khop
  const [n,lo,hi]=AX[i];
  v=Math.max(lo,Math.min(hi,Number(v)));
  if(!isFinite(v))return false;
  vals[i]=v;
  document.getElementById('s'+i).value=v;
  document.getElementById('t'+i).value=+v.toFixed(2);   // +..toFixed bo so 0 thua
  if(syncSlider)document.getElementById('t'+i).classList.remove('bad');
  return true;
}
AX.forEach((_,i)=>{
  document.getElementById('s'+i).oninput=e=>{
    setVal(i,e.target.value);dirty=true;if(realtime)rt.classList.remove('on'),realtime=false};
  const t=document.getElementById('t'+i);
  t.oninput=e=>{                              // go tới đâu vẽ tới đó, khong doi Enter
    const v=Number(e.target.value);
    if(e.target.value===''||!isFinite(v)){t.classList.add('bad');return}  // dang go do/dau tru
    t.classList.remove('bad');setVal(i,v);dirty=true;
    if(realtime)rt.classList.remove('on'),realtime=false};
  t.onchange=e=>{                             // roi o: ep ve gia tri hop le trong [lo,hi]
    const v=Number(e.target.value);
    if(!isFinite(v)){setVal(i,vals[i]);return}   // go chu -> tra ve so cu
    setVal(i,v);t.classList.remove('bad');dirty=true};
  t.onkeydown=e=>{if(e.key==='Enter')t.blur()};
});

document.getElementById('rt').onclick=e=>{  // bat/tat realtime
  realtime=!realtime;e.target.classList.toggle('on',realtime);dirty=true;
  if(realtime)fetch('free');               // nha port de doc state
};
document.getElementById('sync').onclick=async e=>{  // chay THAT tren tay
  e.target.disabled=true;
  try{const r=await fetch('sync',{method:'POST',body:JSON.stringify({pos:vals})});
      st.textContent=await r.text()}
  catch(err){st.textContent='sync loi: '+err.message}
  e.target.disabled=false;
};
document.getElementById('home').onclick=()=>{
  const d0=[500,0,250,0,0,0];for(let i=0;i<6;i++)setVal(i,d0[i]);dirty=true};

// chuot: keo trai = quay, lan = zoom, keo phai = tinh tien
let drag=null;
cv.oncontextmenu=e=>e.preventDefault();
cv.onpointerdown=e=>{drag={x:e.clientX,y:e.clientY,b:e.button};cv.classList.add('drag');
                     cv.setPointerCapture(e.pointerId)};
cv.onpointerup=e=>{drag=null;cv.classList.remove('drag')};
cv.onpointermove=e=>{
  if(!drag)return;
  const dx=e.clientX-drag.x,dy=e.clientY-drag.y;drag={x:e.clientX,y:e.clientY,b:drag.b};
  if(drag.b===0){cam.az=(cam.az-dx*0.4)%360;cam.el=Math.max(-89,Math.min(89,cam.el+dy*0.4))}
  else{cam.pan[0]-=dx*cam.d*0.0015;cam.pan[2]+=dy*cam.d*0.0015}
  dirty=true;
};
cv.onwheel=e=>{e.preventDefault();cam.d=Math.max(.15,Math.min(6,cam.d*(1+Math.sign(e.deltaY)*0.1)));
               dirty=true};
tick();
</script>"""


def preview(port=8081, w=960, h=720, read_hz=8.0):
    """Mo web xem truoc mo phong myArm: http://<ip>:<port>/ . Tra server; goi .stop() de tat.

    6 slider x/y/z/rx/ry/rz -> mo phong cap nhat NGAY khi keo (khong doi tha chuot).
    Nut realtime: doc goc khop THAT tu driver roi ve lai (bam lai -> ve simulate don thuan).
    Nut sync: chay set_gripper_pose THAT voi so tren slider.
    Chuot: keo trai = quay, lan = zoom, keo phai = tinh tien.
    """
    import os
    os.environ.setdefault("MUJOCO_GL", "egl")  # Jetson khong X -> EGL, phai dat truoc khi nap
    if mujoco is None:
        print("khong nap duoc mujoco -> khong mo duoc preview")
        return None
    import cv2
    import json
    import numpy as np
    import threading

    import pinocchio as pin
    from scipy.spatial.transform import Rotation as R

    import queue

    m = mujoco.MjModel.from_xml_path(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "simu", "assets", "myarm_m750.xml"))
    d = mujoco.MjData(m)

    # EGL context gan chat voi THREAD tao ra no. ThreadingHTTPServer phuc vu moi request o
    # thread khac -> eglMakeCurrent bao EGL_BAD_ACCESS. Nen: 1 thread rieng giu renderer,
    # request chi gui viec vao hang doi. Bat buoc, khong phai cho dep.
    jobs = queue.Queue()
    out = {}
    cam = mujoco.MjvCamera()
    box = {}

    def worker():
        r = mujoco.Renderer(m, h, w)  # tao TRONG thread nay -> context thuoc thread nay
        mujoco.mjv_defaultFreeCamera(m, cam)
        base = d.xpos[m.body("base_link").id].copy()
        tip = _grip(_fk([0, 30, -20, 0, 80, 0])) / 1000.0
        cam.lookat[:] = (base + base + tip) / 2.0
        cam.distance, cam.azimuth, cam.elevation = 1.1, 135.0, -20.0
        box["ready"] = True
        while True:
            job = jobs.get()
            if job is None:
                break
            key, q, c = job
            try:
                if c:
                    cam.azimuth = float(c.get("az", cam.azimuth))
                    cam.elevation = max(-89.0, min(89.0, float(c.get("el", cam.elevation))))
                    cam.distance = max(0.15, min(6.0, float(c.get("d", cam.distance))))
                    if c.get("pan"):
                        cam.lookat[:] = np.array(cam.lookat, float) + np.array(c["pan"], float) * 0.5
                d.qpos[:] = 0
                for i, n in enumerate(_ARM):
                    d.qpos[m.jnt_qposadr[m.joint(n).id]] = np.radians(q[i])
                for n in ("left_gripper_joint", "right_gripper_joint"):
                    d.qpos[m.jnt_qposadr[m.joint(n).id]] = 0.0173
                mujoco.mj_forward(m, d)
                r.update_scene(d, cam)
                out[key] = cv2.imencode(".jpg", cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR),
                                        [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
            except Exception as e:
                out[key] = e
        r.close()

    threading.Thread(target=worker, daemon=True).start()
    while not box.get("ready"):
        time.sleep(0.05)
    seed = [[0.0, 30.0, -20.0, 0.0, 80.0, 0.0]]  # nghiem IK truoc do, lam diem khoi dau cho lan sau

    def snap(q, c=None):
        """Nho thread render ve 1 khung (goc khop do) -> bytes JPEG. Nem loi neu render hong."""
        key = object()
        jobs.put((key, q, c))
        while key not in out:
            time.sleep(0.005)
        v = out.pop(key)
        if isinstance(v, Exception):
            raise v
        return v

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass

        def _body(self):
            return json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")

        def _ok(self, ctype, data, code=200):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.startswith("/state"):
                # Mo port NGAY tai day neu dang ranh: preview() khong mo luc khoi dong (de khong
                # chiem tay khi nguoi dung chi muon xem mo phong), nhung realtime thi can.
                if arm is None:
                    try:
                        _open()
                    except Exception as e:
                        return self._ok("application/json",
                                        json.dumps({"err": str(e)[:200]}).encode())
                q = _q_now()
                if q is None:
                    return self._ok("application/json", b'{"err":"doc goc loi"}')
                return self._ok("application/json",
                                json.dumps({"grip": get_gripper_pose()}).encode())
            if self.path.startswith("/free"):
                return self._ok("text/plain", b"ok")  # goc khop di qua fetch(), khong chiem port o day
            self._ok("text/html; charset=utf-8", _PAGE.encode())

        def do_POST(self):
            b = self._body()
            if self.path.startswith("/render"):
                c = b.get("cam") or {}
                pos = [float(v) for v in b["pos"]]
                Rg = R.from_euler("xyz", pos[3:], degrees=True).as_matrix()
                p_tool = np.array(pos[:3]) - Rg @ np.array([0.0, 0.0, GRIP_L])
                # Seed IK lay tu chinh khung hinh cuoi (khong goi _q_now(): ham do MO PORT
                # serial -> preview se chet neu tien trinh khac dang giu tay). Preview chi ve.
                q, ep, eo = _solve_ik(pin.SE3(Rg, p_tool / 1000.0), seed[0], n_restart=3)
                # Phai kiem tra sai so: _solve_ik luon tra ve 1 nghiem nao do, ke ca khi
                # q5 dung tran (ry lon) -> ve ra tu the SAI ma nhin khong biet. Da gap that.
                if q is None or ep > 1.0 or eo > 1.0:
                    return self._ok("text/plain", b"", 422)  # ngoai tam voi: giu khung cu
                seed[0] = q
                return self._ok("image/jpeg", snap(q, c))
            if self.path.startswith("/sync"):
                pos = [float(v) for v in b["pos"]]
                ok = set_gripper_pose(*pos)
                return self._ok("text/plain", ("sync: %s" % ("TOI" if ok else "KHONG TOI")).encode())
            self._ok("text/plain", b"", 404)

    srv = ThreadingHTTPServer(("0.0.0.0", port), H)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    srv.stop = lambda: (srv.shutdown(), srv.server_close())
    atexit.register(srv.stop)
    print("preview: http://0.0.0.0:%d/   (server KHONG tu mo port serial; /sync chi chay khi da mo)" % port)
    return srv


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
    # atexit: script ket thuc tu nhien (khong kip goi .stop()) VAN duoc don -> camera.py het core dump.
    srv.stop = lambda: (run.set(), t.join(3), srv.shutdown(), srv.server_close())
    atexit.register(srv.stop)
    print("camera: http://0.0.0.0:%d/" % port)
    return srv


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


def set_tcp_pose(coords, speed, tol=20.0, settle_s=0.6, timeout_s=60):
    """Gui TCP [x,y,z,rx,ry,rz] mm+do, tra True khi sai so <= tol.
    is_moving() KHONG dung duoc de biet da toi: no tra 0 ngay khi tay con dang di -> phai cho
    vi tri DUNG YEN that (settle_s). tol=20 vi sai so vong ho cua tay nay co chuc mm, khong phai mm."""
    a = _open()
    a.write_coords(coords, speed)
    last, still, deadline = None, 0.0, time.time() + timeout_s
    while time.time() < deadline:
        time.sleep(0.05)
        now = a.get_coords()
        if now == last:
            still += 0.05
            if still >= settle_s:
                break  # dung yen du lau -> coi nhu xong
        else:
            still = 0.0  # con doi -> dem lai tu dau
        last = now
    now = a.get_coords()
    err = max(abs(n - t) for n, t in zip(now[:3], coords[:3]))
    print("sai so %.1f mm (cho phep %.1f): %s" % (err, tol, "TOI" if err <= tol else "KHONG TOI"))
    state()
    return err <= tol


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


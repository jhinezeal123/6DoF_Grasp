"""P2 - Do ma tran Jacobian command -> image cho servo anh.

Ma tran J (2x3, pixel/met) tra loi: khi ra lenh dich TCP mot doan (dx, dy, dz)
met trong he base_link, thi tam ngon kep trong anh dich bao nhieu pixel.

    e_px = J @ [dx, dy, dz]

BA truc, khong phai hai. Lan do dau chi do (x, y) va cho ra ket qua vo dung:
no doi 346 mm dich theo y cho mot sai so 104 px. Do rieng tung truc thi ro
nguyen nhan - do nhay anh chenh nhau 5.3 lan:

    truc x : 1.92 px/mm      truc z : 1.47 px/mm      truc y : 0.36 px/mm

Camera nhin cheo tu truoc-trai, nen truc y (tien/ lui theo ban) gan nhu nam
doc theo huong nhin va hau nhu khong dich anh. Ep ca sai so vao no la sai.
Do ca ba truc roi dung gia nghich dao (pinv): no tu chon to hop hieu qua nhat.

CALIBRATION PHAI CHIA CHO DELTA DA RA LENH, khong phai delta do duoc. J o day mo
ta anh huong cua LENH, nen sai so bam theo cua tay duoc bu tru thay vi thoi
phong J len ~65%. (Da tung ket luan sai cho nay o test 1.)

Cach do: tai tu the dau P, hoi model ngon kep o dau -> g0. Lan luot ra lenh
+2cm x, +2cm y, -2cm z (moi lan ve P truoc), hoi lai -> g1, g2, g3. Ba cot cua J
la (g_i - g0)/step. Ve P.

Nhan dien ngon kep: HOI MODEL, khong do mau. Do that tren canh lam viec dem duoc
11 vat do (ghe do, ao do, do tren ban, nguoi di lai) va khong vat nao trong so do
la ngon kep - loc mau don thuan khong the phan biet. Moi lan hoi ~15 s.

An toan: buoc 2cm la nho; huong giu nguyen (lay tu tcp_quat thuc te, khong doan);
luon ve tu the dau trong finally.

Chay:  python scripts/calibrate_jacobian.py [--step 0.02] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

SDK_DIR = Path("/workspace/6DoF_Grasp/htc/SDK")
sys.path.insert(0, str(SDK_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from program.robot.robot import Robot  # noqa: E402
from program.ros_bridge import get_bridge  # noqa: E402
import program.ros_bridge as ros_bridge  # noqa: E402

from vla.physbrain_model import PhysBrainClient, PhysBrainError  # noqa: E402
from vla.prompts import GRIPPER_QUESTION, POINT_SUFFIX  # noqa: E402

# /dev/video2 = SPCA2650 = camera primary. Phai ep CAP_V4L2, khong thi OpenCV
# chon GStreamer va bao "Failed to open camera 'primary' on index 2".
CAM_INDEX = 2
CAM_W, CAM_H = 1280, 720

GRIPPER_Q = GRIPPER_QUESTION

_CLIENT = PhysBrainClient("http://127.0.0.1:8081")

OUT = Path(__file__).resolve().parents[1] / "configs" / "servo_jacobian.json"
ART = Path(__file__).resolve().parents[1] / "artifacts"
ART.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Thi giac: hoi model ngon kep nam o dau
# --------------------------------------------------------------------------- #
def model_point(frame: np.ndarray, question: str):
    """Hoi model mot diem trong khung. Tra ((px, py), text_tho) hoac (None, text).

    Dung CHUNG PhysBrainClient voi run_pipeline.py: bo parse o do da biet chiu ca
    kieu JSON cua tac gia lan kieu tuple Python, va da tung sua mot lan vi model
    tra sai khuon. Nhan doi cho nay ra la de hai ban troi lech nhau.
    """
    try:
        point = _CLIENT.point(frame, question + "\n" + POINT_SUFFIX)
    except PhysBrainError as exc:
        return None, "loi goi model: %s" % exc
    return point, _CLIENT.last_text


def capture_frame(cap: cv2.VideoCapture, tag: str) -> np.ndarray:
    """Chup mot khung da on dinh (bo qua anh cu con nam trong buffer)."""
    for _ in range(12):
        cap.grab()
    frame = None
    for _ in range(10):
        ok, frame = cap.read()
        if ok:
            break
    if frame is None:
        raise RuntimeError("khong doc duoc khung tu camera")
    cv2.imwrite(str(ART / ("p2_%s.jpg" % tag)), frame)
    return frame


# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description="Do Jacobian command -> image.")
    parser.add_argument("--step", type=float, default=0.02,
                        help="do dich moi buoc, met (mac dinh 0.02 = 2cm)")
    parser.add_argument("--dry-run", action="store_true",
                        help="do thu, KHONG ra lenh chuyen dong")
    args = parser.parse_args()

    robot = Robot()
    print("=== TRANG THAI DAU ===")
    print("  OFFSETS_DEG      :", np.round(ros_bridge.OFFSETS_DEG, 2))
    print("  MODEL_MIN_RAD    :", np.round(ros_bridge.MODEL_MIN_RAD, 3))
    print("  MODEL_MAX_RAD    :", np.round(ros_bridge.MODEL_MAX_RAD, 3))

    # Phai cho: bridge moi tao chua kip khop publisher nao ca. Doc ngay thi
    # tcp_pos/tcp_quat con NaN va ta tu choi chay - dung, nhung oan uong.
    print("\n  cho stack phat feedback (toi da 25s)...")
    if not robot.wait_until_online(25.0):
        print("LOI: stack khong phat feedback sau 25s.")
        return 1
    time.sleep(0.5)

    bridge = get_bridge()
    p0 = np.array(robot.tcp_pos, dtype=float)
    q0 = np.array(robot.tcp_quat, dtype=float)
    print("  tcp_pos  :", np.round(p0, 4))
    print("  tcp_quat :", np.round(q0, 4))
    if not np.all(np.isfinite(p0)) or not np.all(np.isfinite(q0)):
        print("LOI: tcp_pos/tcp_quat khong hop le (NaN). Khong the hieu chuan.")
        return 1

    # BAT BUOC: executor TU CHOI moi lenh khi cong an toan cua driver dang disarmed,
    # nhung send_tcp_pose VAN tra True. Khong rearm thi tay dung im, anh khong
    # nhuc nhich, va J do duoc chi la nhieu - dung cai bay da lam hong lan dau.
    if not robot.is_armed:
        state, detail, _ = bridge.motion_state
        print("  dang disarmed: %s" % (detail or state))
        print("  goi rearm()...")
        if not robot.rearm():
            print("LOI: rearm() that bai.")
            return 1
        time.sleep(1.5)
    if not robot.is_armed:
        print("LOI: van chua armed sau rearm. safety_state =", robot.safety_state)
        return 1
    print("  safety_state:", robot.safety_state, " is_armed:", robot.is_armed)

    def wait_settled(commanded=None, timeout_s: float = 25.0) -> bool:
        """Cho tay dung yen THAT SU sau mot lenh. Tra False neu het thoi gian.

        Rat de viet sai ham nay, va da viet sai hai lan:

        Lan 1 - chi cho trang thai "idle". Executor bao "succeeded" (khong phai
        "idle"), nen quay vong vo ich 15 s roi bao loi oan cho con tay.

        Lan 2 - cho trang thai khac "executing" roi doi TCP dung yen. Nhung ngay
        sau khi ra lenh, trang thai VAN CON la "succeeded" cua lenh TRUOC (executor
        chua kip chuyen sang "executing"), nen ham tra ve ngay lap tuc va phep do
        duoc lay tren canh tay dang di chuyen. Do that: tai tu the goc z = 0.195
        nhung diem goc do duoc z = 0.1822 - lech 12.8 mm, tay chua toi noi.

        Lan nay chot ba tang: nghi toi thieu de executor nhan lenh, doi thoat khoi
        executing/pending, roi doi dung yen 2 s lien tuc. Kem theo doi chieu TCP
        voi dich da ra lenh de con nhin thay sai so bam.
        """
        time.sleep(1.0)                       # de executor chuyen sang "executing"
        t_end = time.time() + timeout_s
        last, stable = None, 0
        while time.time() < t_end:
            time.sleep(0.25)
            if bridge.motion_state[0] in ("executing", "pending"):
                stable, last = 0, None
                continue
            if bridge.motion_state[0] == "" and time.time() - (t_end - timeout_s) < 2.0:
                continue                       # chua nhan duoc lenh, cho them
            cur = np.array(robot.tcp_pos, dtype=float)
            if last is not None and np.linalg.norm(cur - last) < 5e-4:
                stable += 1
                if stable >= 8:                # dung yen ~2 s
                    if commanded is not None:
                        residual = float(np.linalg.norm(cur - commanded)) * 1000
                        if residual > 10.0:
                            print("     (bam theo lech %.1f mm so voi dich da ra lenh)"
                                  % residual)
                    return True
            else:
                stable = 0
            last = cur
        return False

    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print("LOI: khong mo duoc camera index", CAM_INDEX)
        return 1

    # Ba truc va dau cua phep do. z di XUONG (-) de khong bao gio vuot tran;
    # day la phep do ngan nhat va cung la huong servo se dung that.
    AXES = (("x", np.array([1.0, 0.0, 0.0])),
            ("y", np.array([0.0, 1.0, 0.0])),
            ("z", np.array([0.0, 0.0, -1.0])))

    def goto(offset_xyz, tag: str):
        """Ra lenh toi p0 + offset, cho dung han, hoi model ngon kep o dau.

        Tra ve (diem_anh, tcp_do_duoc, text_model).
        """
        target = p0 + np.asarray(offset_xyz, dtype=float)
        if not args.dry_run:
            ok = robot.set_tcp_pose(target, q0)
            if not ok:
                raise RuntimeError("set_tcp_pose tra False (lenh bi tu choi)")
            if not wait_settled(commanded=target):
                raise RuntimeError("tay khong dung yen sau 25s (tag=%s)" % tag)
        frame = capture_frame(cap, tag)
        tcp = np.array(robot.tcp_pos, dtype=float)
        print("  [%s] hoi model (co the mat ~15s)..." % tag)
        t_start = time.time()
        point, text = model_point(frame, GRIPPER_Q)
        print("  [%s] model tra loi sau %.1fs: %s" % (tag, time.time() - t_start, text.strip()))
        return point, tcp, text

    points: dict[str, tuple[float, float]] = {}
    moved: dict[str, np.ndarray] = {}
    origin = None
    try:
        print("\n=== DIEM GOC ===")
        g0, t0, _ = goto(np.zeros(3), "g0")
        if g0 is None:
            raise RuntimeError("model khong chi duoc ngon kep o tu the goc")
        origin, points["0"] = t0, g0
        print("  g0 = (%.1f, %.1f) px   tcp = %s" % (g0[0], g0[1], np.round(t0, 4)))

        for name, direction in AXES:
            print("\n=== BUOC %s: %+.0f mm theo %s ==="
                  % (name, direction[np.argmax(np.abs(direction))] * args.step * 1000, name))
            point, tcp, _ = goto(direction * args.step, "g_" + name)
            if point is None:
                raise RuntimeError("model khong chi duoc ngon kep sau buoc %s" % name)
            points[name], moved[name] = point, tcp - t0
            delta_px = np.array(point) - np.array(g0)
            print("  g_%s = (%.1f, %.1f) px   dich anh = (%.2f, %.2f) px   |d| = %.1f px"
                  % (name, point[0], point[1], delta_px[0], delta_px[1],
                     float(np.linalg.norm(delta_px))))
            print("  dich TCP thuc = %s m" % np.round(tcp - t0, 4))

            # Chot chan: TCP khong doi thi moi con so J deu la nhieu. Lan chay dau da
            # cho ra J "dep" (cond=2.1) tu mot con tay dung im - phai chan tu day.
            if not args.dry_run and np.linalg.norm(tcp - t0) < args.step * 0.2:
                raise RuntimeError(
                    "tay KHONG di chuyen o buoc %s: lenh %.0f mm nhung TCP chi doi %.2f mm. "
                    "Kiem tra is_armed / motion_state truoc khi tin vao J."
                    % (name, args.step * 1000, np.linalg.norm(tcp - t0) * 1000))

            print("\n=== VE TU THE DAU ===")
            _, tb, _ = goto(np.zeros(3), "gb_" + name)
            print("  lech TCP so voi goc = %s m" % np.round(tb - t0, 4))

    finally:
        print("\n=== VE TU THE DAU (finally) ===")
        if not args.dry_run:
            robot.set_tcp_pose(p0, q0)
            wait_settled()
            back = np.array(robot.tcp_pos, dtype=float)
            print("  tcp sau khi ve = %s   lech %.1f mm so voi goc"
                  % (np.round(back, 4), np.linalg.norm(back - p0) * 1000))
        cap.release()

    missing = [n for n, _ in AXES if n not in points]
    if "0" not in points or missing:
        print("\nLOI: thieu diem do cho truc %s. Khong tinh duoc J." % (missing or "goc"))
        return 1

    # Chia cho DO DICH CO DAU, khong phai args.step. Truc z do bang offset -0.02
    # (di xuong), nen chia cho +0.02 se lat nguoc dau cot z: J bao "ha xuong thi
    # anh di len", va servo se lenh tay BAY LEN thay vi ha xuong. Da dinh dung
    # loi nay mot lan - log in ra buoc z = +20.0 mm trong khi phai la -142 mm.
    columns = [(np.array(points[name]) - np.array(points["0"])) / (direction[axis] * args.step)
               for axis, (name, direction) in enumerate(AXES)]
    J = np.column_stack(columns)

    print("\n" + "=" * 66)
    print("MA TRAN J (pixel/met)   [cot x, y, z]")
    for row in range(2):
        print("  [%s]" % "  ".join("%9.1f" % J[row, c] for c in range(J.shape[1])))

    # Do nhay tung truc: day moi la con so quyet dinh servo dung duoc hay khong.
    # Lan do dau chi co (x, y) va truc y yeu gap 5.3 lan truc x, khien servo doi
    # 346 mm cho mot sai so 104 px. Nhin bang nay la thay ngay.
    print("\n  Do nhay tung truc (px/mm) va dau:")
    for c, (name, _) in enumerate(AXES):
        px_per_mm = float(np.linalg.norm(J[:, c])) / 1000.0
        print("    %s : %5.2f px/mm   (1 px sai <=> %.2f mm)"
              % (name, px_per_mm, 1.0 / max(px_per_mm, 1e-9)))

    # Chot chan vat ly: ha xuong (dz < 0) PHAI lam ngon kep di XUONG trong anh
    # (image-y tang), tuc J[1][2] < 0. Sai dau o day thi servo lenh tay bay len
    # thay vi ha xuong - dung loi da dinh mot lan do chia sai dau phep do.
    if J[1, 2] >= 0:
        print("\n  LOI: cot z sai dau (J[1][2] = %.0f >= 0). Ha xuong phai lam anh di"
              " xuong. Kiem tra lai phep do truoc khi dung J nay." % J[1, 2])
        return 1
    print("  dau cot z OK (ha xuong -> anh di xuong)")

    cond = float(np.linalg.cond(J))
    print("\n  cond(J) = %.1f   %s" % (cond, "OK" if cond < 20 else "CAO - J gan suy bien"))

    weakest = min(range(len(AXES)), key=lambda c: np.linalg.norm(J[:, c]))
    weak_px_per_mm = np.linalg.norm(J[:, weakest]) / 1000.0
    print("  truc yeu nhat: %s (%.2f px/mm)" % (AXES[weakest][0], weak_px_per_mm))
    if weak_px_per_mm < 0.2:
        print("  CANH BAO: truc '%s' gan nhu khong lam anh nhuc nhich. Cot J bang 0"
              % AXES[weakest][0])
        print("            thuong nghia la co vat gi CHAN chuyen dong do, khong phai")
        print("            J dep. Lan do dau, kep ti xuong mat ban lam cot z ra dung 0.0.")

    if args.dry_run:
        print("\n(DRY RUN - khong ghi file)")
        return 0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "J": J.tolist(),
        "axes": [name for name, _ in AXES],
        "step_m": args.step,
        "p0": p0.tolist(),
        "q0": q0.tolist(),
        "points_px": {k: list(v) for k, v in points.items()},
        "tcp_moved_m": {k: v.tolist() for k, v in moved.items()},
        "cond": cond,
        "offsets_deg": list(np.round(ros_bridge.OFFSETS_DEG, 4)),
        "detector": {"kind": "physbrain_point_2d", "server": _CLIENT.url,
                     "question": GRIPPER_Q},
        "camera": {"index": CAM_INDEX, "width": CAM_W, "height": CAM_H},
    }, indent=2))
    print("\nDa ghi:", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

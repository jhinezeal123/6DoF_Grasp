"""P2 - Do ma tran Jacobian command -> image cho servo anh.

Ma tran J (2x2, pixel/met) tra loi: khi ra lenh dich TCP mot doan (dx, dy) met
trong he base_link, thi tam ngon kep trong anh dich bao nhieu pixel.

    e_px = J @ [dx, dy]

CALIBRATION PHAI CHIA CHO DELTA DA RA LENH, khong phai delta do duoc. J o day mo
ta anh huong cua LENH, nen sai so bam theo cua tay duoc bu tru thay vi thoi
phong J len ~65%. (Da tung ket luan sai cho nay o test 1.)

Cach do: tai tu the dau P, hoi model ngon kep o dau -> g0. Ra lenh +2cm theo x,
hoi lai -> g1, duoc cot 1. Ve P, ra lenh +2cm theo y, hoi lai -> g2, duoc cot 2.
Ve P.

Nhan dien ngon kep: HOI MODEL, khong do mau. Do that tren canh lam viec dem duoc
11 vat do (ghe do, ao do, do tren ban, nguoi di lai) va khong vat nao trong so do
la ngon kep - loc mau don thuan khong the phan biet. Model thi hieu "ngon kep cua
canh tay robot" la gi. Moi lan hoi ~17 s cho anh moi; chap nhan duoc.

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

    def wait_settled(timeout_s: float = 20.0) -> bool:
        """Cho tay dung yen that su. Tra False neu het thoi gian.

        Do that: executor bao "succeeded" (KHONG phai "idle") roi tay dung im
        hoan toan sau ~4s. Chi doi "idle" la doi mai mai - dung cai bay da lam
        lan chay truoc quay vong 15s roi bao loi oan cho con tay.
        """
        t_end = time.time() + timeout_s
        last, stable = None, 0
        while time.time() < t_end:
            time.sleep(0.25)
            if bridge.motion_state[0] in ("executing", "pending", ""):
                stable, last = 0, None
                continue
            cur = np.array(robot.tcp_pos, dtype=float)
            if last is not None and np.linalg.norm(cur - last) < 5e-4:
                stable += 1
                if stable >= 4:          # dung yen ~1s
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

    def goto(offset_xy, tag: str):
        """Ra lenh toi p0 + offset, cho dung han, hoi model ngon kep o dau.

        Tra ve (diem_anh, tcp_do_duoc, text_model).
        """
        target = p0 + np.array([offset_xy[0], offset_xy[1], 0.0])
        if not args.dry_run:
            ok = robot.set_tcp_pose(target, q0)
            if not ok:
                raise RuntimeError("set_tcp_pose tra False (lenh bi tu choi)")
            if not wait_settled():
                raise RuntimeError("tay khong dung yen sau 20s (tag=%s)" % tag)
        frame = capture_frame(cap, tag)
        tcp = np.array(robot.tcp_pos, dtype=float)
        print("  [%s] hoi model (co the mat ~17s)..." % tag)
        t_start = time.time()
        point, text = model_point(frame, GRIPPER_Q)
        print("  [%s] model tra loi sau %.1fs: %s" % (tag, time.time() - t_start, text.strip()))
        return point, tcp, text

    g0 = g1 = g2 = None
    t0 = t1 = t2 = tb = None
    try:
        print("\n=== DIEM GOC ===")
        g0, t0, _ = goto(np.zeros(2), "g0")
        if g0 is None:
            raise RuntimeError("model khong chi duoc ngon kep o tu the goc")
        print("  g0 = (%.1f, %.1f) px   tcp = %s" % (g0[0], g0[1], np.round(t0, 4)))

        print("\n=== BUOC 1: +%.0f mm theo x ===" % (args.step * 1000))
        g1, t1, _ = goto(np.array([args.step, 0.0]), "g1")
        if g1 is None:
            raise RuntimeError("model khong chi duoc ngon kep sau buoc x")
        print("  g1 = (%.1f, %.1f) px   dich anh = (%.2f, %.2f) px"
              % (g1[0], g1[1], g1[0] - g0[0], g1[1] - g0[1]))
        print("  tcp = %s   dich TCP = %s m" % (np.round(t1, 4), np.round(t1 - t0, 4)))

        # Chot chan: TCP khong doi thi moi con so J deu la nhieu. Lan chay dau da
        # cho ra J "dep" (cond=2.1) tu mot con tay dung im - phai chan tu day.
        if not args.dry_run and np.linalg.norm(t1 - t0) < args.step * 0.2:
            raise RuntimeError(
                "tay KHONG di chuyen: lenh +%.0f mm nhung TCP chi doi %.2f mm. "
                "Kiem tra is_armed / motion_state truoc khi tin vao J."
                % (args.step * 1000, np.linalg.norm(t1 - t0) * 1000))

        print("\n=== VE TU THE DAU ===")
        _, tb, _ = goto(np.zeros(2), "gb")
        print("  tcp = %s   lech TCP = %s m" % (np.round(tb, 4), np.round(tb - t0, 4)))

        print("\n=== BUOC 2: +%.0f mm theo y ===" % (args.step * 1000))
        g2, t2, _ = goto(np.array([0.0, args.step]), "g2")
        if g2 is None:
            raise RuntimeError("model khong chi duoc ngon kep sau buoc y")
        print("  g2 = (%.1f, %.1f) px   dich anh = (%.2f, %.2f) px"
              % (g2[0], g2[1], g2[0] - g0[0], g2[1] - g0[1]))
        print("  tcp = %s   dich TCP = %s m" % (np.round(t2, 4), np.round(t2 - t0, 4)))

    finally:
        print("\n=== VE TU THE DAU (finally) ===")
        if not args.dry_run:
            robot.set_tcp_pose(p0, q0)
            wait_settled()
            back = np.array(robot.tcp_pos, dtype=float)
            print("  tcp sau khi ve = %s   lech %.1f mm so voi goc"
                  % (np.round(back, 4), np.linalg.norm(back - p0) * 1000))
        cap.release()

    if g0 is None or g1 is None or g2 is None:
        print("\nLOI: thieu diem (g0/g1/g2). Khong tinh duoc J.")
        return 1

    d1 = np.array(g1) - np.array(g0)
    d2 = np.array(g2) - np.array(g0)
    J = np.column_stack([d1 / args.step, d2 / args.step])

    print("\n" + "=" * 62)
    print("MA TRAN J (pixel/met)   [cot 1 = x, cot 2 = y]")
    print("  [[%9.1f  %9.1f]" % (J[0, 0], J[0, 1]))
    print("  " + " " * 9 + "%9.1f  %9.1f]]" % (J[1, 0], J[1, 1]))

    det = float(np.linalg.det(J))
    cond = float(np.linalg.cond(J))
    print("\n  det(J)      = %.3g" % det)
    print("  cond(J)     = %.1f   %s" % (cond, "OK" if cond < 20 else "XAU - J gan suy bien"))

    # Buoc tay 1mm dich bao nhieu pixel: con so quyet dinh do chinh xac servo.
    px_per_mm = float(np.linalg.norm(J[:, 0])) / 1000.0
    print("  1 mm theo x -> %.2f px" % px_per_mm)
    print("  => sai so dinh vi 1 px tuong duong %.2f mm" % (1.0 / max(px_per_mm, 1e-9)))

    if args.dry_run:
        print("\n(DRY RUN - khong ghi file)")
        return 0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "J": J.tolist(),
        "step_m": args.step,
        "p0": p0.tolist(),
        "q0": q0.tolist(),
        "g0": list(g0),
        "g1": list(g1),
        "g2": list(g2),
        "tcp_moved_x_m": (t1 - t0).tolist(),
        "tcp_moved_y_m": (t2 - t0).tolist(),
        "det": det,
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

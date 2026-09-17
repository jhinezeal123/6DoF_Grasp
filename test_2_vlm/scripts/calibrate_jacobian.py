"""P2 - Do ma tran Jacobian command -> image cho servo anh.

Ma tran J (2x2, pixel/met) tra loi: khi ra lenh dich TCP mot doan (dx, dy) met
trong he base_link, thi tam ngon kep trong anh dich bao nhieu pixel.

    e_px = J @ [dx, dy]

CALIBRATION PHAI CHIA CHO DELTA DA RA LENH, khong phai delta do duoc. J o day mo
ta anh huong cua LENH, nen sai so bam theo cua tay duoc bu tru thay vi thoi
phong J len ~65%. (Da tung ket luan sai cho nay o test 1.)

Cach do: tai tu the dau P, do g0. Ra lenh +2cm theo x, do g1 -> cot 1. Ve P,
ra lenh +2cm theo y, do g2 -> cot 2. Ve P.

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

from program.robot.robot import Robot  # noqa: E402
from program.ros_bridge import get_bridge  # noqa: E402
import program.ros_bridge as ros_bridge  # noqa: E402

# /dev/video2 = SPCA2650 = camera primary. Phai ep CAP_V4L2, khong thi OpenCV
# chon GStreamer va bao "Failed to open camera 'primary' on index 2".
CAM_INDEX = 2
CAM_W, CAM_H = 1280, 720

OUT = Path(__file__).resolve().parents[1] / "configs" / "servo_jacobian.json"
ART = Path(__file__).resolve().parents[1] / "artifacts"
ART.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Thi giac: tim tam ngon kep do
# --------------------------------------------------------------------------- #
def _red_mask(frame: np.ndarray) -> np.ndarray:
    """Mat na do. Nguong do bang so do that, khong doan.

    V>=90 (nguong dau tien) lam ROT mot ngon kep khi auto-exposure cua camera
    tut xuong luc co nguoi di vao khung: do do hai ngon lech nhau ve do sang.
    Do tai cho: ngon sang V~110, ngon toi V~80. Lay V>=70 cho ca hai lot.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 90, 70]), np.array([12, 255, 255]))
    mask |= cv2.inRange(hsv, np.array([168, 90, 70]), np.array([180, 255, 255]))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))


def _blobs(mask: np.ndarray, min_area: int = 120):
    n, _, stats, cent = cv2.connectedComponentsWithStats(mask)
    return sorted(
        ((int(stats[i, 4]), cent[i]) for i in range(1, n) if stats[i, 4] >= min_area),
        key=lambda b: -b[0],
    )


def red_moved_point(frame_a: np.ndarray, frame_b: np.ndarray,
                    min_px: int = 60):
    """Tam khoi DO da DI CHUYEN giua hai khung, o ca hai khung.

    Tra (tam_o_A, tam_o_B, so_px) hoac None.

    Vi sao khong di tim "hai blob do to nhat": do that tren canh that dem duoc
    11 vat do (ghe do, ao do, vat tren ban, nguoi di lai) va khong vat nao trong
    so do la ngon kep. Mau do don thuan khong du de phan biet.

    Nhung vat tinh thi khong di chuyen. Giao mat na do voi vung anh da thay doi
    giua hai tu the se giu lai dung phan ngon kep, va loai sach hau canh.
    """
    diff = cv2.absdiff(frame_a, frame_b).max(axis=2)
    moved = ((diff > 25).astype(np.uint8)) * 255
    # No ra mot chut: ngon kep dich vai chuc px thi vien truoc/sau moi cham nhau.
    moved = cv2.dilate(moved, np.ones((9, 9), np.uint8))

    out = []
    for frame in (frame_a, frame_b):
        sel = cv2.bitwise_and(_red_mask(frame), moved)
        n = int(cv2.countNonZero(sel))
        if n < min_px:
            out.append(None)
            continue
        m = cv2.moments(sel, binaryImage=True)
        out.append((m["m10"] / m["m00"], m["m01"] / m["m00"]))
    if out[0] is None or out[1] is None:
        return None
    return out[0], out[1], int(cv2.countNonZero(cv2.bitwise_and(_red_mask(frame_b), moved)))


def gripper_point(frame: np.ndarray) -> tuple[float, float] | None:
    """Tam ngon kep do, hoac None neu khong tim thay cap nao hop ly.

    Chi dung khi khong co anh tham chieu (khung goc). Chon theo CAP chu khong
    lay 2 blob to nhat: nut dung khan cap cung mau do va co the to hon mot ngon kep.
    """
    blobs = _blobs(_red_mask(frame))
    if len(blobs) < 2:
        return None

    best, best_score = None, None
    for i in range(min(len(blobs), 6)):
        for j in range(i + 1, min(len(blobs), 6)):
            a0, c0 = blobs[i]
            a1, c1 = blobs[j]
            d = float(np.linalg.norm(c0 - c1))
            ratio = max(a0, a1) / max(1, min(a0, a1))
            if d > 300 or ratio > 2.5:
                continue
            # Uu tien: gan nhau, cung co, va to.
            score = d / 300.0 + ratio - (a0 + a1) / 2000.0
            if best_score is None or score < best_score:
                best, best_score = (c0 + c1) / 2.0, score

    if best is None:
        return None
    return float(best[0]), float(best[1])


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
# Do
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=float, default=0.02, help="do lon buoc do (met)")
    ap.add_argument("--dry-run", action="store_true", help="khong ra lenh di chuyen")
    args = ap.parse_args()

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

    print("  is_real_connected:", robot.is_real_connected)
    p0 = np.array(robot.tcp_pos, dtype=float)
    q0 = np.array(robot.tcp_quat, dtype=float)
    print("  tcp_pos  :", np.round(p0, 4))
    print("  tcp_quat :", np.round(q0, 4))
    if not np.all(np.isfinite(p0)) or not np.all(np.isfinite(q0)):
        print("LOI: tcp_pos/tcp_quat khong hop le (NaN). Khong the hieu chuan.")
        return 1

    bridge = get_bridge()

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
        """Ra lenh toi p0 + offset, cho dung han, chup anh. Tra (khung, tcp_do_duoc).

        Tra ve ca tcp do duoc: neu tay khong nhuc nhich thi moi con so Jacobian
        tinh ra deu la nhieu, khong phai do nhay.
        """
        target = p0 + np.array([offset_xy[0], offset_xy[1], 0.0])
        if not args.dry_run:
            ok = robot.set_tcp_pose(target, q0)
            if not ok:
                raise RuntimeError("set_tcp_pose tra False (lenh bi tu choi)")
            if not wait_settled():
                raise RuntimeError("tay khong dung yen sau 20s (tag=%s)" % tag)
        frame = capture_frame(tag)
        return frame, np.array(robot.tcp_pos, dtype=float)

    f0 = f1 = f2 = None
    t0 = t1 = t2 = tb = None
    try:
        print("\n=== DO DIEM GOC ===")
        f0, t0 = goto(np.zeros(2), "g0")
        p_g0 = gripper_point(f0)
        print("  tcp do duoc = %s" % np.round(t0, 4))
        print("  (chi de tham khao) ngon kep theo mau do: %s"
              % ("(%.1f, %.1f)" % p_g0 if p_g0 else "khong xac dinh - canh co nhieu vat do"))

        print("\n=== BUOC 1: +%.0f mm theo x ===" % (args.step * 1000))
        f1, t1 = goto(np.array([args.step, 0.0]), "g1")
        print("  tcp do duoc = %s   dich TCP = %s m"
              % (np.round(t1, 4), np.round(t1 - t0, 4)))

        # Chot chan: TCP khong doi thi moi con so J deu la nhieu. Lan chay dau da
        # cho ra J "dep" (cond=2.1) tu mot con tay dung im - phai chan tu day.
        if not args.dry_run and np.linalg.norm(t1 - t0) < args.step * 0.2:
            raise RuntimeError(
                "tay KHONG di chuyen: lenh +%.0f mm nhung TCP chi doi %.2f mm. "
                "Kiem tra is_armed / motion_state truoc khi tin vao J."
                % (args.step * 1000, np.linalg.norm(t1 - t0) * 1000))

        print("\n=== VE TU THE DAU ===")
        _, tb = goto(np.zeros(2), "gb")
        print("  tcp do duoc = %s   lech TCP = %s m"
              % (np.round(tb, 4), np.round(tb - t0, 4)))

        print("\n=== BUOC 2: +%.0f mm theo y ===" % (args.step * 1000))
        f2, t2 = goto(np.array([0.0, args.step]), "g2")
        print("  tcp do duoc = %s   dich TCP = %s m"
              % (np.round(t2, 4), np.round(t2 - t0, 4)))

    finally:
        print("\n=== VE TU THE DAU (finally) ===")
        if not args.dry_run:
            robot.set_tcp_pose(p0, q0)
            wait_settled()
            back = np.array(robot.tcp_pos, dtype=float)
            print("  tcp sau khi ve = %s   lech %.1f mm so voi goc"
                  % (np.round(back, 4), np.linalg.norm(back - p0) * 1000))
        cap.release()

    if f0 is None or f1 is None or f2 is None:
        print("\nLOI: khong chup du 3 khung (g0/g1/g2). Khong tinh duoc J.")
        return 1

    # Do dich chuyen cua khoi DO DA DI CHUYEN, khong phai cua "blob do to nhat".
    r1 = red_moved_point(f0, f1)
    r2 = red_moved_point(f0, f2)
    if r1 is None or r2 is None:
        print("\nLOI: khong thay khoi do nao di chuyen giua cac tu the.")
        print("  nghia la mat na do giao vung anh-thay-doi khong con gi.")
        print("  kiem tra: ngon kep co bi khuat khong, anh co bi loa khong.")
        return 1

    d1 = np.array(r1[1]) - np.array(r1[0])
    d2 = np.array(r2[1]) - np.array(r2[0])
    print("\n=== DICH CHUYEN ANH CUA NGON KEP ===")
    print("  +%.0f mm theo x -> (%7.2f, %7.2f) px   [%d px do chung minh]"
          % (args.step * 1000, d1[0], d1[1], r1[2]))
    print("  +%.0f mm theo y -> (%7.2f, %7.2f) px   [%d px do chung minh]"
          % (args.step * 1000, d2[0], d2[1], r2[2]))

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
        "d_px_x": d1.tolist(),
        "d_px_y": d2.tolist(),
        "tcp_moved_x_m": (t1 - t0).tolist(),
        "tcp_moved_y_m": (t2 - t0).tolist(),
        "det": det,
        "cond": cond,
        "offsets_deg": list(np.round(ros_bridge.OFFSETS_DEG, 4)),
        "camera": {"index": CAM_INDEX, "width": CAM_W, "height": CAM_H},
    }, indent=2))
    print("\nDa ghi:", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

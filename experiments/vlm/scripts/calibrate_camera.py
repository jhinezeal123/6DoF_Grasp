"""Hieu chuan camera chinh (SPCA2650) de biet mot diem anh nam o dau trong 3D.

Hai che do:

  1. In ban co de chup:
         python scripts/calibrate_camera.py --make-board
     Ghi ra artifacts/chessboard.png. IN RA GIAY A4, dung ti le 100% (khong
     "fit to page"). Sau khi in PHAI lay thuoc do lai mot o vuong that chinh xac
     roi truyen --square-mm dung so do - may in thuong co gian ti le, va day la
     nguon sai so lon nhat cua ca phep hieu chuan.

  2. Chup va tinh thong so:
         python scripts/calibrate_camera.py --square-mm 24.5
     Cua so hien anh live. Giua yen roi bam SPACE de chup mot tu the. Chup 15-20
     tu the: nghieng trai/phai/len/xuong, xa/gan, va dat ban co o bon goc anh
     chu khong chi giua - meo ong kinh chi lo ra o ria anh. Bam ESC de dung va
     tinh. Ket qua ghi vao configs/camera_intrinsics.json.

Ket qua dung de lam gi: tu mot diem anh (u, v) suy ra tia nhin trong he toa do
camera, roi giao voi mat ban (biet z_table) de ra toa do 3D trong base_link. Do
la manh ghep con thieu: tay da biet chinh xac ngon kep cua no o dau qua tcp_pos,
chi con thieu buoc anh -> 3D.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_COLS = 9          # so goc trong theo chieu ngang (so o = cols + 1)
DEFAULT_ROWS = 6          # so goc trong theo chieu doc
DEFAULT_SQUARE_MM = 25.0  # PHẢI do lai sau khi in


def make_board(cols: int, rows: int, out: Path, dpi: int = 300,
               square_mm: float = DEFAULT_SQUARE_MM) -> None:
    """Ve ban co den-trang de in. cols/rows la so GOC TRONG."""
    square_px = max(20, int(round(square_mm / 25.4 * dpi)))
    width = square_px * (cols + 1)
    height = square_px * (rows + 1)
    board = np.full((height, width), 255, dtype=np.uint8)
    for r in range(rows + 2):
        for c in range(cols + 2):
            if (r + c) % 2 == 0:
                board[r * square_px:(r + 1) * square_px,
                      c * square_px:(c + 1) * square_px] = 0

    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), board)
    print("da ghi %s" % out)
    print("  %d x %d goc trong, o vuong %.1f mm, %d dpi -> %d x %d px"
          % (cols, rows, square_mm, dpi, width, height))
    print("  IN O TI LE 100%% (tat 'fit to page'), roi DO LAI mot o bang thuoc.")
    print("  Kich thuoc that tren giay A4: %.0f x %.0f mm"
          % (width / dpi * 25.4, height / dpi * 25.4))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--make-board", action="store_true",
                        help="chi ve ban co de in roi thoat")
    parser.add_argument("--cols", type=int, default=DEFAULT_COLS,
                        help="so goc trong theo chieu ngang (mac dinh 9)")
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS,
                        help="so goc trong theo chieu doc (mac dinh 6)")
    parser.add_argument("--square-mm", type=float, default=DEFAULT_SQUARE_MM,
                        help="canh o vuong THAT tren giay, mm (do lai sau khi in)")
    parser.add_argument("--camera", type=int, default=2, help="chi so /dev/videoN")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--out", default=str(ROOT / "configs" / "camera_intrinsics.json"))
    args = parser.parse_args()

    board_out = ROOT / "artifacts" / "chessboard.png"
    if args.make_board:
        make_board(args.cols, args.rows, board_out, square_mm=args.square_mm)
        return 0

    pattern = (args.cols, args.rows)

    def object_points():
        """Toa do 3D cua cac goc trong he ban co, don vi MET."""
        square_m = args.square_mm / 1000.0
        grid = np.zeros((args.rows * args.cols, 3), np.float32)
        grid[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2)
        return grid * square_m

    objp = object_points()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4)
    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
             | cv2.CALIB_CB_FAST_CHECK)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("LOI: khong mo duoc /dev/video%d" % args.camera)
        if not board_out.exists():
            make_board(args.cols, args.rows, board_out, square_mm=args.square_mm)
            print("  (da ve ban co de in: %s)" % board_out)
        return 1
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    for _ in range(6):
        cap.read()
        time.sleep(0.05)

    obj_points, img_points = [], []
    print("\nSPACE = chup mot tu the co ban co   |   ESC = dung va tinh")
    print("Can 15-20 tu the, dat ban co ca o bon goc anh chu dung chi o giua.\n")

    window = "calibrate camera"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 960, 540)
    found_now = False
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("khong doc duoc khung anh"); break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found_now, corners = cv2.findChessboardCorners(gray, pattern, flags)
        view = frame.copy()
        if found_now:
            cv2.drawChessboardCorners(view, pattern, corners, found_now)
        label = ("THAY ban co (%d da chup) - bam SPACE" % len(obj_points) if found_now
                 else "khong thay ban co (%d da chup)" % len(obj_points))
        cv2.putText(view, label, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 255, 0) if found_now else (0, 0, 255), 2)
        cv2.imshow(window, view)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == 32 and found_now:
            # refine lam goc chinh xac duoi pixel - thieu buoc nay thi ket qua
            # thap hon nhieu so voi kha nang that cua camera.
            refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            obj_points.append(objp.copy())
            img_points.append(refined)
            print("  da chup %d/%d" % (len(obj_points), 20))

    cap.release()
    cv2.destroyAllWindows()

    if len(obj_points) < 6:
        print("\nLOI: chi chup duoc %d tu the, can it nhat 6 (nen 15-20)."
              % len(obj_points))
        return 1

    print("\n dang tinh...")
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, (args.width, args.height), None, None
    )
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    fov_x = 2.0 * np.degrees(np.arctan(args.width / (2.0 * fx)))
    fov_y = 2.0 * np.degrees(np.arctan(args.height / (2.0 * fy)))

    print("\n================ KET QUA ================")
    print("  so tu the     : %d" % len(obj_points))
    print("  sai so RMS    : %.3f px   (duoi 0.5 la tot; tren 1.5 la co van de)" % rms)
    print("  fx, fy        : %.2f, %.2f px" % (fx, fy))
    print("  cx, cy        : %.2f, %.2f px   (tam anh la %.0f, %.0f)"
          % (cx, cy, args.width / 2, args.height / 2))
    print("  FOV ngang/doc : %.1f x %.1f do" % (fov_x, fov_y))
    print("  meo           : %s" % np.array2string(dist.ravel(), precision=5))

    # Sai so tai hien: chieu lai diem vat qua K/dist roi so voi diem do duoc.
    # Day moi la con so noi len camera co dung duoc cho dieu khien hay khong.
    worst = 0.0
    for i in range(len(obj_points)):
        projected, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], K, dist)
        worst = max(worst, float(np.abs(projected - img_points[i]).max()))
    print("  sai so chieu lai lon nhat: %.3f px" % worst)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "image_size": [args.width, args.height],
        "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        "distortion": [float(v) for v in dist.ravel()],
        "fov_x_deg": float(fov_x), "fov_y_deg": float(fov_y),
        "rms_px": float(rms),
        "square_mm": args.square_mm,
        "views": len(obj_points),
    }, indent=2) + "\n")
    print("\n  da ghi %s" % out)
    print("\nCON THIEU: vi tri + huong camera so voi base_link (extrinsics).")
    print("Cach chac nhat khong phai do thuoc: cho TCP cham vao 4-6 diem danh dau")
    print("tren mat ban, ghi lai tcp_pos va diem anh tuong ung.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

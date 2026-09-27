"""Kiem tra 2 camera theo DUNG config Octo: primary=index 2, wrist=index 0.

Vi sao phai viet script nay thay vi tin vao config: config cu cua dan anh ghi
primary=index 0 / wrist=index 2, tuc la GAN NGUOC vai tro so voi thuc te.
Anh chup tu 2 camera cho thay:
  - /dev/video0 (Logitech C925e) : can can mat ban  -> WRIST (gan tren tay)
  - /dev/video2 (SPCA2650)       : toan canh phong  -> PRIMARY (camera canh)
Nen index dung la primary=2, wrist=0.

CANH BAO: chi MOT tien trinh duoc stream V4L2. Phai tat SDK truoc:
    pkill -9 -f run_web.sh ; pkill -9 -f web_control
Sau khi kiem tra xong, bat lai SDK bang htc/SDK/run_web.sh.

Chay:
  cd /workspace/6DoF_Grasp/htc/test_1_octo
  LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1 \
    /home/ktmt-agx-xv/Data/khoanhd/Octo_Lab/Octo_env/bin/python check_cameras.py
"""
from __future__ import annotations

import sys

import cv2
import numpy as np

# (ten, index, width, height, fps, fourcc) — copy tu
# scripts/robot_configs/myarm_m750_primary_wrist_720p_mjpg.json
CAMERAS = (
    ("primary (SPCA2650, canh)", 2, 1280, 720, 30, "MJPG"),
    ("wrist   (Logitech C925e)", 0, 1280, 720, 30, "MJPG"),
)

READ_FRAMES = 10


def check(name, index, width, height, fps, fourcc):
    print(f"\n=== {name}  index={index} ===")
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("  KHONG MO DUOC (thiet bi dang bi tien trinh khac giu?)")
        return False

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    got = None
    for _ in range(READ_FRAMES):
        ok, frame = cap.read()
        if ok and frame is not None:
            got = frame

    if got is None:
        print("  Mo duoc nhung KHONG doc duoc khung nao")
        cap.release()
        return False

    # Doc lai cac thong so THAT SU duoc thiet lap, khong tin gia tri yeu cau.
    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    real_fps = cap.get(cv2.CAP_PROP_FPS)
    cc = int(cap.get(cv2.CAP_PROP_FOURCC))
    real_fourcc = "".join(chr((cc >> (8 * i)) & 0xFF) for i in range(4))
    cap.release()

    out = f"/tmp/cam_{index}.jpg"
    cv2.imwrite(out, got)
    print(f"  yeu cau : {width}x{height} {fourcc} @{fps}")
    print(f"  thuc te : {real_w}x{real_h} {real_fourcc} @{real_fps:.1f}")
    print(f"  khung   : shape={got.shape} dtype={got.dtype} mean={np.mean(got):.1f}")
    print(f"  da ghi  : {out}")

    if np.mean(got) < 1.0:
        print("  CANH BAO: anh gan nhu den hoan toan")
    if (real_w, real_h) != (width, height):
        print("  CANH BAO: do phan giai thuc te KHAC yeu cau")
    return True


def main():
    print("Kiem tra camera theo config Octo (primary=2, wrist=0)")
    results = [check(*c) for c in CAMERAS]
    print("\n=== TONG KET ===")
    for (name, *_), ok in zip(CAMERAS, results):
        print(f"  {'OK  ' if ok else 'LOI '} {name}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())

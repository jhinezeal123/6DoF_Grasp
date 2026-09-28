"""Chup mot khung, hoi model diem dich + diem ngon kep, ve len anh de NHIN.

Dung truoc khi cho tay chay that: neu diem dich khong nam tren vat thi moi thu
phia sau deu sai, ma servo thi van chay cham chi - no se dua ngon kep toi mot cho
khong co gi ca.

Chay:  python scripts/check_target.py [--instruction "..."]
Ket qua: artifacts/check_target.jpg
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vla.physbrain_model import PhysBrainClient  # noqa: E402
from vla.primary_camera import PrimaryCamera  # noqa: E402
from vla.prompts import GRIPPER_QUESTION, point_question  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "grasp_myarm_m750.json"))
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--out", default=str(ROOT / "artifacts" / "check_target.jpg"))
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    instruction = args.instruction or config["instruction"]

    cam_cfg = config.get("camera", {})
    camera = PrimaryCamera(
        index=cam_cfg.get("index", 2),
        width=cam_cfg.get("width", 1280),
        height=cam_cfg.get("height", 720),
    )
    camera.open()
    try:
        frame = camera.photo()
    finally:
        camera.close()
    if frame is None or not isinstance(frame, np.ndarray):
        print("LOI: camera khong tra ve khung anh")
        return 1
    print("khung anh: %s" % (frame.shape,))

    model = PhysBrainClient(url=config["model"]["url"])
    if not model.ready():
        print("LOI: model server khong tra loi /health")
        return 1

    vis = frame[:, :, ::-1].copy()  # RGB -> BGR de ve bang cv2

    for name, question, color in (
        ("DICH", point_question(instruction), (0, 0, 255)),
        ("KEP", GRIPPER_QUESTION, (0, 255, 0)),
    ):
        point = model.point(frame, question)
        if point is None:
            print("  %-5s: model khong tra loi duoc (text=%r)" % (name, model.last_text[:120]))
            continue
        print("  %-5s: (%.1f, %.1f) px   sau %.1fs"
              % (name, point[0], point[1], model.last_latency_s or 0.0))
        center = (int(round(point[0])), int(round(point[1])))
        cv2.drawMarker(vis, center, color, cv2.MARKER_CROSS, 44, 3)
        cv2.circle(vis, center, 26, color, 2)
        cv2.putText(vis, name, (center[0] + 30, center[1] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), vis)
    print("da ghi %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

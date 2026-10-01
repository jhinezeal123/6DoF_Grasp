#!/usr/bin/env python3
"""Run the pinned grasp pipeline with a supplied metric depth map.

This is a validation-only bridge. It runs inside the pipeline checkout's own
environment (JetPack Torch/TensorRT) and replaces the monocular depth adapter
with a depth map rendered by the simulator, so the rest of the pipeline (YOLOE
mask, TSDF, VGN, grasp decoding) is exercised unchanged.

The pipeline is never modified: only the ``LiteMonoDepth.predict`` method is
replaced in-process before the estimator is built.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def _matrix(flat, shape, name):
    values = np.asarray(json.loads(flat), dtype=np.float64)
    if values.size != int(np.prod(shape)):
        raise SystemExit("%s needs %d values" % (name, int(np.prod(shape))))
    return values.reshape(shape)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-repo", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--depth", required=True,
                        help=".npy float32 axial camera-Z depth in metres")
    parser.add_argument("--camera-k", required=True, help="JSON 9 values")
    parser.add_argument("--camera-from-volume", default=None, help="JSON 16 values")
    parser.add_argument("--prompt-id", default="cube")
    parser.add_argument("--max-width", type=float, default=0.069)
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()

    sys.path.insert(0, args.pipeline_repo)

    from grasppose.modules.depth.geometry import resolve_camera_intrinsics
    from grasppose.modules.depth.lite_mono import LiteMonoDepth
    from grasppose.modules.depth.types import DepthResult

    K = _matrix(args.camera_k, (3, 3), "camera_k")
    T_cam_volume = (
        None if args.camera_from_volume is None
        else _matrix(args.camera_from_volume, (4, 4), "camera_from_volume")
    )
    depth_map = np.load(args.depth).astype(np.float32)

    def predict(self, image, camera_K=None, fov_x=None):
        rgb = np.asarray(image)[:, :, :3]
        height, width = rgb.shape[:2]
        if depth_map.shape != (height, width):
            raise SystemExit(
                "depth map %r does not match image %r"
                % (depth_map.shape, (height, width))
            )
        intrinsics = resolve_camera_intrinsics(camera_K, fov_x, width, height)
        return DepthResult(
            depth=depth_map,
            intrinsics=intrinsics.astype(np.float32),
            fov_x_deg=float(2.0 * np.degrees(np.arctan(
                width / (2.0 * max(float(intrinsics[0, 0]), 1e-6))))),
            scale=1.0,
            reason="simulator ground-truth depth",
        )

    LiteMonoDepth.predict = predict

    from PIL import Image

    from grasppose.api import get_estimator

    estimator = get_estimator().load()
    try:
        rgb = np.asarray(Image.open(args.image).convert("RGB"))
        result = estimator.estimate(
            rgb,
            args.prompt_id,
            camera_K=K,
            max_width=args.max_width,
            top=args.top,
            T_cam_volume=T_cam_volume,
        )
        payload = {
            "grasps": [
                {
                    "score": float(grasp.score),
                    "width_m": float(grasp.width_m),
                    "translation_m": [float(v) for v in grasp.translation_m],
                    "rotation": [[float(v) for v in row] for row in grasp.rotation],
                }
                for grasp in result.grasps
            ],
            "depth_m": result.depth_m,
            "detection_count": int(result.detection_count),
            "mask_pixels": int(result.mask_pixels),
            "grasp_count": int(result.grasp_count),
        }
    finally:
        estimator.close()
    print(json.dumps(payload, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

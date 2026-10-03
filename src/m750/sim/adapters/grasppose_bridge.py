#!/usr/bin/env python3
"""Run the pinned grasp pipeline with a supplied metric depth map.

This is a validation-only bridge. It runs under the *pipeline* checkout's own
interpreter (JetPack Torch/TensorRT) and substitutes a depth map rendered by the
simulator, so the rest of the pipeline (YOLOE mask, TSDF, VGN, grasp decoding) is
exercised unchanged.

It is a program, not a module: the harness launches it by absolute path and it
must never import m750, because that interpreter does not have this package.

The substitution goes through the pipeline's declared seam: ``DepthPort`` is the
contract, and ``build_default_pipeline(depth=...)`` is the composition root that
accepts one. Nothing here names a concrete depth adapter, because an earlier
version of this file replaced ``LiteMonoDepth.predict`` directly and went
silently dead the moment the pipeline switched to a different depth model --
it kept reporting ``simulator_ground_truth`` while running the real model.

Two guards keep that from recurring:

* the injected port counts its calls, and the bridge fails if the pipeline never
  used it;
* the harness is told the pipeline's actual ``TSDF_SIZE_M`` so the two repos can
  no longer drift apart on a constant neither side can import from the other.
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


def make_simulator_depth(DepthPort, DepthResult, resolve_camera_intrinsics, depth_map):
    """Build a DepthPort that returns the supplied simulator depth map."""

    class SimulatorDepth(DepthPort):
        def __init__(self):
            self.calls = 0

        def load(self):
            return self

        def warmup(self):
            return self

        def close(self):
            return None

        def predict(self, image, camera_K=None, fov_x=None):
            self.calls += 1
            rgb = np.asarray(image)[:, :, :3]
            height, width = rgb.shape[:2]
            if depth_map.shape != (height, width):
                raise ValueError(
                    "depth map %r does not match image %r" % (depth_map.shape, (height, width))
                )
            intrinsics = resolve_camera_intrinsics(camera_K, fov_x, width, height)
            return DepthResult(
                depth=depth_map,
                intrinsics=intrinsics.astype(np.float32),
                fov_x_deg=float(
                    2.0 * np.degrees(np.arctan(width / (2.0 * max(float(intrinsics[0, 0]), 1e-6))))
                ),
                scale=1.0,
                reason="simulator ground-truth depth",
            )

    return SimulatorDepth()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-repo", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--depth", required=True, help=".npy float32 axial camera-Z depth in metres"
    )
    parser.add_argument("--camera-k", required=True, help="JSON 9 values")
    parser.add_argument("--camera-from-volume", default=None, help="JSON 16 values")
    parser.add_argument("--prompt-id", default="cube")
    parser.add_argument("--max-width", type=float, default=0.069)
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()

    sys.path.insert(0, args.pipeline_repo)

    from grasppose.application.service import LocalGraspEstimator
    from grasppose.infrastructure.composition import build_default_pipeline
    from grasppose.infrastructure.settings import TSDF_SIZE_M
    from grasppose.modules.depth.geometry import resolve_camera_intrinsics
    from grasppose.modules.depth.port import DepthPort
    from grasppose.modules.depth.types import DepthResult

    K = _matrix(args.camera_k, (3, 3), "camera_k")
    T_cam_volume = (
        None
        if args.camera_from_volume is None
        else _matrix(args.camera_from_volume, (4, 4), "camera_from_volume")
    )
    depth_map = np.load(args.depth).astype(np.float32)

    depth_port = make_simulator_depth(DepthPort, DepthResult, resolve_camera_intrinsics, depth_map)
    estimator = LocalGraspEstimator(build_default_pipeline(depth=depth_port)).load()

    from PIL import Image

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
        # The whole point of this bridge is that the supplied map is what the
        # pipeline sees. If the port was never asked for a depth map, the
        # pipeline used its own model and every number below is a lie.
        if depth_port.calls == 0:
            raise SystemExit(
                "the injected DepthPort was never used: the pipeline ignored the "
                "supplied depth map, so this run would report simulator "
                "ground-truth depth while using its own model"
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
            "depth_port_calls": int(depth_port.calls),
            "tsdf_size_m": float(TSDF_SIZE_M),
        }
    finally:
        estimator.close()
    print(json.dumps(payload, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

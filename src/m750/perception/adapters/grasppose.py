"""Adapter for the pinned pipeline_grasppose GraspEstimator API.

Target commit: 5703506a9d012eaf807387e305cfba4c68d0d6e3.
The adapter intentionally uses the public GraspEstimator and EstimateResult
boundary exposed by that commit instead of internal PipelineResult data.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

from ..contracts import PerceptionProvider
from ..types import GraspCandidate, PerceptionRequest, PerceptionResult


GRASPPOSE_COMMIT = "5703506a9d012eaf807387e305cfba4c68d0d6e3"


def _matrix_to_quaternion_xyzw(matrix) -> tuple:
    m = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    quaternion = np.asarray([x, y, z, w], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    return tuple(float(value) for value in quaternion)


class GraspPosePerceptionAdapter(PerceptionProvider):
    """Adapt the pinned repository public GraspEstimator to local types."""

    def __init__(self, estimator: Optional[Any] = None) -> None:
        self._estimator = estimator

    def _get_estimator(self):
        if self._estimator is None:
            try:
                from grasppose.api import get_estimator
            except ImportError as exc:
                raise RuntimeError(
                    "pipeline_grasppose is unavailable. Checkout commit "
                    + GRASPPOSE_COMMIT
                    + " and add that checkout to PYTHONPATH."
                ) from exc
            self._estimator = get_estimator()
        return self._estimator

    def open(self) -> None:
        """Optional lifecycle helper for LocalGraspEstimator."""

        estimator = self._get_estimator()
        loader = getattr(estimator, "load", None)
        if callable(loader):
            loader()
        warmup = getattr(estimator, "warmup", None)
        if callable(warmup):
            warmup()

    def infer(self, request: PerceptionRequest) -> PerceptionResult:
        estimate = self._get_estimator().estimate(
            request.image,
            request.prompt_id,
            camera_K=request.camera_matrix,
            fov_x=request.fov_x_deg,
            fov_y=request.fov_y_deg,
            camera_K_size=request.camera_matrix_size,
            max_width=request.max_width_m,
            top=request.top,
            T_cam_volume=request.camera_from_volume,
        )

        grasps = tuple(
            GraspCandidate(
                score=float(grasp.score),
                width_m=float(grasp.width_m),
                position_m=tuple(float(value) for value in grasp.translation_m),
                quaternion_xyzw=_matrix_to_quaternion_xyzw(grasp.rotation),
                metadata={
                    "source": "pipeline_grasppose",
                    "commit": GRASPPOSE_COMMIT,
                },
            )
            for grasp in estimate.grasps
        )
        return PerceptionResult(
            grasps=grasps,
            depth_m=estimate.depth_m,
            raw=estimate,
        )

    def close(self) -> None:
        """Optional lifecycle helper; not part of PerceptionProvider."""

        if self._estimator is not None:
            closer = getattr(self._estimator, "close", None)
            if callable(closer):
                closer()


__all__ = ["GRASPPOSE_COMMIT", "GraspPosePerceptionAdapter"]

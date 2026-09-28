"""Adapter for github.com/jhinezeal123/pipeline_grasppose."""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

from ..contracts import PerceptionProvider
from ..types import GraspCandidate, PerceptionRequest, PerceptionResult


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
    q = np.asarray([x, y, z, w], dtype=np.float64)
    q /= np.linalg.norm(q)
    return tuple(float(value) for value in q)


class GraspPosePerceptionAdapter(PerceptionProvider):
    """Translate the grasp-pose pipeline result into the local perception port.

    The external package is imported lazily. This repository therefore keeps
    working without YOLOE, TensorRT or the grasp-pose checkout installed.
    """

    def __init__(self, service: Optional[Any] = None) -> None:
        self._service = service

    def _get_service(self):
        if self._service is None:
            try:
                from grasppose.facade import DEFAULT_SERVICE
            except ImportError as exc:
                raise RuntimeError(
                    "pipeline_grasppose is not installed; add it to the runtime "
                    "environment or inject a compatible GraspService"
                ) from exc
            self._service = DEFAULT_SERVICE
        return self._service

    def open(self) -> None:
        service = self._get_service()
        service.load()
        warmup = getattr(service.core, "warmup", None)
        if callable(warmup):
            warmup()

    def infer(self, request: PerceptionRequest) -> PerceptionResult:
        service = self._get_service()
        result = service.core.run(
            request.image,
            request.prompt_id,
            camera_K=request.camera_matrix,
            fov_x=request.fov_x_deg,
            T_cam_volume=request.camera_from_volume,
        )

        rows = np.asarray(result.grasp.graspgroup, dtype=np.float64)
        grasps = []
        for row in rows:
            if row.size < 17:
                continue
            rotation = row[4:13].reshape(3, 3)
            grasps.append(
                GraspCandidate(
                    score=float(row[0]),
                    width_m=float(row[1]),
                    position_m=tuple(float(value) for value in row[13:16]),
                    quaternion_xyzw=_matrix_to_quaternion_xyzw(rotation),
                    metadata={
                        "height_m": float(row[2]),
                        "depth_m": float(row[3]),
                        "object_id": float(row[16]),
                    },
                )
            )
        return PerceptionResult(
            grasps=tuple(grasps),
            depth_m=result.depth_m,
            raw=result,
        )

    def close(self) -> None:
        if self._service is not None:
            self._service.close()


__all__ = ["GraspPosePerceptionAdapter"]

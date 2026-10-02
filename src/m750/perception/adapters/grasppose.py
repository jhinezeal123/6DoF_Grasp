"""Adapter for the pinned pipeline_grasppose GraspEstimator API.

Target commit: 666c7eb608c5315ea252fd02b3f5446c39198ee6.
The adapter intentionally uses the public GraspEstimator and EstimateResult
boundary exposed by that commit instead of internal PipelineResult data.
"""

from __future__ import annotations

from typing import Any, Optional

from scipy.spatial.transform import Rotation

from ..contracts import PerceptionProvider
from ..types import GraspCandidate, PerceptionRequest, PerceptionResult


GRASPPOSE_COMMIT = "666c7eb608c5315ea252fd02b3f5446c39198ee6"


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
                quaternion_xyzw=tuple(
                    float(value)
                    for value in Rotation.from_matrix(grasp.rotation).as_quat()
                ),
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

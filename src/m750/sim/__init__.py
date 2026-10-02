"""Simulation grasp validation feature.

MuJoCo only: nothing here can reach a physical robot driver or ROS. The harness
is a validation tool rather than a contract other features compose, so the root
facade deliberately does not re-export it.

This facade exposes camera/grasp geometry, the fixed scenario constants and the
domain values. The MuJoCo world, the bridge to the pinned pipeline and the CLI
are concrete integrations -- import those from their own modules.
"""

from .geometry import (
    camera_k,
    camera_optical_transform,
    candidate_transform,
    gravity_aligned_volume,
    oracle_candidate,
    project_camera,
    rotation_error_deg,
    tf,
)
from .scenario import (
    CAMERA_Q_DEG,
    CUBE,
    FOVY,
    HEIGHT,
    LIFT_HEIGHT_M,
    LIGHTS,
    MAX_OPEN,
    OFFSETS,
    SCENE,
    SEED,
    VOLUME_SIZE_M,
    WIDTH,
)
from .types import MotionPlan, ValidationFailure

__all__ = [
    "CAMERA_Q_DEG",
    "CUBE",
    "FOVY",
    "HEIGHT",
    "LIFT_HEIGHT_M",
    "LIGHTS",
    "MAX_OPEN",
    "MotionPlan",
    "OFFSETS",
    "SCENE",
    "SEED",
    "VOLUME_SIZE_M",
    "ValidationFailure",
    "WIDTH",
    "camera_k",
    "camera_optical_transform",
    "candidate_transform",
    "gravity_aligned_volume",
    "oracle_candidate",
    "project_camera",
    "rotation_error_deg",
    "tf",
]

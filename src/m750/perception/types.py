"""Backend-neutral perception values."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple


@dataclass(frozen=True)
class PerceptionRequest:
    image: Any
    prompt_id: str
    camera_matrix: Any = None
    fov_x_deg: Optional[float] = None
    fov_y_deg: Optional[float] = None
    camera_matrix_size: Any = None
    max_width_m: float = 0.080
    top: int = 1
    camera_from_volume: Any = None


@dataclass(frozen=True)
class GraspCandidate:
    score: float
    width_m: float
    position_m: Tuple[float, float, float]
    quaternion_xyzw: Tuple[float, float, float, float]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PerceptionResult:
    grasps: Tuple[GraspCandidate, ...]
    depth_m: Optional[float]
    raw: Any = None


__all__ = [
    "GraspCandidate",
    "PerceptionRequest",
    "PerceptionResult",
]

"""Coordinate mappings used by sim/real synchronization."""

from __future__ import annotations

from abc import ABC, abstractmethod
import math
from typing import Sequence, Tuple


def _joints(values: Sequence[float]) -> Tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != 6 or not all(math.isfinite(value) for value in result):
        raise ValueError("joint vector must contain six finite values")
    return result


class JointMapper(ABC):
    @abstractmethod
    def map(self, joints_rad: Sequence[float]) -> Tuple[float, ...]:
        """Convert source joint coordinates into target joint coordinates."""


class IdentityJointMapper(JointMapper):
    def map(self, joints_rad: Sequence[float]) -> Tuple[float, ...]:
        return _joints(joints_rad)


class AffineJointMapper(JointMapper):
    """Per-joint mapping: target = source * scale + offset."""

    def __init__(
        self,
        scale: Sequence[float] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        offset_rad: Sequence[float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    ) -> None:
        self._scale = _joints(scale)
        self._offset = _joints(offset_rad)

    def map(self, joints_rad: Sequence[float]) -> Tuple[float, ...]:
        source = _joints(joints_rad)
        return tuple(
            value * scale + offset
            for value, scale, offset in zip(source, self._scale, self._offset)
        )


__all__ = ["AffineJointMapper", "IdentityJointMapper", "JointMapper"]

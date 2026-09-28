"""Perception extension point used by robot applications."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .types import PerceptionRequest, PerceptionResult


class PerceptionProvider(ABC):
    """Minimal client-specific interface for grasp estimation."""

    @abstractmethod
    def infer(self, request: PerceptionRequest) -> PerceptionResult:
        """Return backend-neutral grasp candidates."""


__all__ = ["PerceptionProvider"]

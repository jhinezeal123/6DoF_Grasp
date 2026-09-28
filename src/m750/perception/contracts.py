"""Perception extension point used by robot applications."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .types import PerceptionRequest, PerceptionResult


class PerceptionProvider(ABC):
    @abstractmethod
    def open(self) -> None:
        """Load persistent resources."""

    @abstractmethod
    def infer(self, request: PerceptionRequest) -> PerceptionResult:
        """Return backend-neutral grasp candidates."""

    @abstractmethod
    def close(self) -> None:
        """Release persistent resources."""


__all__ = ["PerceptionProvider"]

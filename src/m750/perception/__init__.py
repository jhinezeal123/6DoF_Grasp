"""Perception feature.

Concrete model stacks are optional adapters. The root API exports only this
feature's contract and domain values.
"""

from .contracts import PerceptionProvider
from .types import GraspCandidate, PerceptionRequest, PerceptionResult

__all__ = [
    "GraspCandidate",
    "PerceptionProvider",
    "PerceptionRequest",
    "PerceptionResult",
]

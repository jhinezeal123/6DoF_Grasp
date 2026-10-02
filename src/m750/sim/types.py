"""Backend-neutral simulation validation values."""

from __future__ import annotations

from dataclasses import dataclass


class ValidationFailure(RuntimeError):
    """A simulated episode could not be trusted."""


@dataclass(frozen=True)
class MotionPlan:
    q_pre_path: tuple
    q_approach_path: tuple
    q_lift_path: tuple
    ik_errors: tuple
    grasp_base: tuple
    tool_to_grasp: tuple
    opening_m: float


__all__ = ["MotionPlan", "ValidationFailure"]

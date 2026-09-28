"""Explicit real/simulation transfer use cases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from m750.robot.contracts import (
    GripperController,
    JointPositionController,
    RobotStateReader,
)

from .mapping import IdentityJointMapper, JointMapper


@dataclass(frozen=True)
class TransferResult:
    ok: bool
    joints_applied: bool
    gripper_applied: bool
    reason: str = ""


class _StateTransfer:
    def __init__(
        self,
        source: RobotStateReader,
        target_joints: JointPositionController,
        *,
        target_gripper: Optional[GripperController] = None,
        mapper: Optional[JointMapper] = None,
    ) -> None:
        self._source = source
        self._target_joints = target_joints
        self._target_gripper = target_gripper
        self._mapper = mapper or IdentityJointMapper()

    def execute(self, *, copy_gripper: bool = True) -> TransferResult:
        state = self._source.read_state()
        if state.joints_rad is None:
            return TransferResult(False, False, False, "source has no joint state")

        target = self._mapper.map(state.joints_rad)
        if not self._target_joints.move_joints(target):
            return TransferResult(False, False, False, "target rejected joint state")

        gripper_applied = False
        if copy_gripper and self._target_gripper is not None:
            opening = state.gripper_opening_m
            if opening is not None:
                if not self._target_gripper.set_gripper(opening):
                    return TransferResult(
                        False,
                        True,
                        False,
                        "target rejected gripper state",
                    )
                gripper_applied = True

        return TransferResult(True, True, gripper_applied)


class RealToSim(_StateTransfer):
    """Copy a measured real state into a simulator driver."""


class SimToReal(_StateTransfer):
    """Apply a simulated state to a real driver.

    Safety policy remains a responsibility of the injected real driver and the
    composition root; this use case never bypasses driver validation.
    """


__all__ = ["RealToSim", "SimToReal", "TransferResult"]

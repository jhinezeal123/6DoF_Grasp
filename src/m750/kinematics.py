"""kinematics.py - FK toi tool0 bang Pinocchio + URDF.

Class ArmKinematics thay ham _model() cu (cache bang cach gan thuoc tinh vao
function object). Logic giu NGUYEN: nap URDF dung 1 lan, URDF la nguon gioi
han chinh. Khong mo port serial -> test offline duoc.
"""
from __future__ import annotations

import numpy as np
import pinocchio as pin

from .spec import JOINT_NAMES, RobotSpec


class ArmKinematics:
    """Nap URDF 1 lan (Pinocchio nang). FK + gioi han khop, khong can hardware."""

    def __init__(self, spec: RobotSpec | None = None) -> None:
        self.spec = spec or RobotSpec()
        # URDF la nguon gioi han chinh - tai lieu lab muc 7.5
        # (get_joint_max() khong dang tin).
        self.m = pin.buildModelFromUrdf(str(self.spec.urdf_path))
        self.d = self.m.createData()
        self.idx = [int(self.m.joints[self.m.getJointId(n)].idx_q) for n in JOINT_NAMES]
        self.lo_deg = [float(np.degrees(self.m.lowerPositionLimit[i])) for i in self.idx]
        self.hi_deg = [float(np.degrees(self.m.upperPositionLimit[i])) for i in self.idx]

    def fk_tool0(self, q_deg) -> pin.SE3:
        """FK toi tool0: 6 goc khop (do) -> pin.SE3."""
        q = pin.neutral(self.m)
        for i, ix in enumerate(self.idx):
            q[ix] = float(np.radians(q_deg[i]))
        pin.forwardKinematics(self.m, self.d, q)
        pin.updateFramePlacements(self.m, self.d)
        return self.d.oMf[self.m.getFrameId("tool0")]

    def grip_mm(self, M: pin.SE3) -> np.ndarray:
        """pin.SE3 cua tool0 -> vi tri gripper (mm, he URDF)."""
        return M.translation * 1000.0 + M.rotation @ np.array(
            [0.0, 0.0, self.spec.grip_l_mm]
        )

    def tool0_of_gripper(self, pos_mm, Rg) -> np.ndarray:
        """Dich gripper [x,y,z mm] -> vi tri tool0 (m): tru offset GRIP_L theo truc z cua tool0."""
        return (np.asarray(pos_mm, dtype=float)
                - Rg @ np.array([0.0, 0.0, self.spec.grip_l_mm])) / 1000.0

    @property
    def limits_deg(self) -> tuple:
        """(lo, hi) gioi han URDF, don vi do, theo thu tu JOINT_NAMES."""
        return tuple(self.lo_deg), tuple(self.hi_deg)


__all__ = ["ArmKinematics"]

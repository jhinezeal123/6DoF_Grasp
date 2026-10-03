"""ik.py - IK 6 DOF thuan toan hoc (khong mo port -> test duoc).

IKResult: ket qua giai kem sai so THAT sau khi giai - phai kiem tra ca hai, vi
bo giai co the khop vi tri nhung KHONG xoay duoc co tay theo huong yeu cau
(pose ngoai vung lam viec).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

from .kinematics import ArmKinematics


@dataclass(frozen=True)
class IKResult:
    """Mot nghiem IK: goc khop (do) + sai so vi tri (mm) + sai so huong (do)."""

    q_deg: tuple | None
    pos_err_mm: float
    rot_err_deg: float

    @property
    def reached(self) -> bool:
        """Nghiem dat trong sai so cho phep mac dinh (0.05mm/0.05do)."""
        return self.q_deg is not None and self.pos_err_mm < 0.05 and self.rot_err_deg < 0.05

    def within(self, tol_pos: float, tol_rot: float) -> bool:
        """Nghiem dat trong nguong tuy y (mm/do)."""
        return self.q_deg is not None and self.pos_err_mm <= tol_pos and self.rot_err_deg <= tol_rot


class IKSolver:
    """IK 6 DOF (least_squares + multi-restart, seed cung dinh rng(0))."""

    def __init__(self, kin: ArmKinematics, n_restart: int = 15) -> None:
        self.kin = kin
        self.n_restart = n_restart

    def solve(self, T_target: pin.SE3, q_seed) -> tuple:
        """Tra (q_goal_do, ep_mm, eo_do) - giu API tuple nhu ban cu.

        ep/eo la sai so THAT sau khi giai - phai kiem tra ca hai, vi bo giai co
        the khop vi tri nhung KHONG xoay duoc co tay theo huong yeu cau (pose
        ngoai vung lam viec).
        """
        LO, HI = self.kin.limits_deg
        Rg = T_target.rotation

        def res(qd):
            # e[:3] la MET, e[3:] la RADIAN -> nhan 1000 de cung thang do voi nhau,
            # neu khong sai so vi tri (m) bi be so voi sai so huong (rad).
            e = pin.log6(self.kin.fk_tool0(qd).actInv(T_target)).vector
            return np.concatenate([e[:3] * 1000.0, e[3:]])

        rng = np.random.default_rng(0)
        best = (None, 1e18, 1e18, 1e18)
        for k in range(self.n_restart):
            seed = np.array(q_seed, dtype=float) if k == 0 else rng.uniform(LO, HI)
            if k == 0 and not all(LO[i] <= seed[i] <= HI[i] for i in range(6)):
                seed = rng.uniform(LO, HI)
            try:
                r = least_squares(res, seed, bounds=(LO, HI), max_nfev=3000)
            except Exception:
                continue
            qs = [float(v) for v in r.x]
            Tf = self.kin.fk_tool0(qs)
            ep = float(np.linalg.norm(Tf.translation * 1000.0 - T_target.translation * 1000.0))
            eo = float(np.degrees(np.linalg.norm(R.from_matrix(Tf.rotation.T @ Rg).as_rotvec())))
            if ep + eo < best[1]:
                best = (qs, ep + eo, ep, eo)
            if ep < 0.05 and eo < 0.05:
                break
        return best[0], best[2], best[3]


__all__ = ["IKResult", "IKSolver"]

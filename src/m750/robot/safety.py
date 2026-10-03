"""safety.py - SafetyGate: kiem tra truoc khi ra lenh chuyen dong.

Tach khoi set_gripper_pose cu hai kiem tra an toan:
  1. gioi han khop (URDF la nguon chinh)
  2. drop-check duong di: noi suy goc khop, neu z tut qua drop_max so voi
     ca hai dau -> chan (bai hoc tu lan tay dap ban: +10 do cho CA 6 khop
     lam tool tut 106mm).
"""

from __future__ import annotations

import numpy as np

from .kinematics import ArmKinematics


class SafetyGate:
    """Chan lenh neu vuot gioi han khop, hoac tool TUT xuong qua drop_max tren duong di."""

    def __init__(self, kin: ArmKinematics, drop_max_mm: float = 30.0) -> None:
        self.kin = kin
        self.drop_max_mm = drop_max_mm

    def check_joints(self, q_deg) -> bool:
        """True neu moi khop trong gioi han URDF."""
        lo, hi = self.kin.limits_deg
        for i, (v, vlo, vhi) in enumerate(zip(q_deg, lo, hi)):
            if not (vlo <= v <= vhi):
                print("  !!! CHAN: khop %d = %.2f ngoai [%.1f..%.1f]" % (i + 1, v, vlo, vhi))
                return False
        return True

    def check_path(self, q_from, q_to) -> bool:
        """Chan duong di neu z cua gripper TUT qua drop_max so voi ca hai dau.

        Noi suy 11 diem theo GOC KHOP (khong phai theo duong thang TCP).
        """
        lo, hi = self.kin.limits_deg
        # gioi han khop phai dat ca hai dau
        for q, name in ((q_from, "diem di"), (q_to, "dich")):
            for i, (v, vlo, vhi) in enumerate(zip(q, lo, hi)):
                if not (vlo <= v <= vhi):
                    print(
                        "  !!! CHAN %s: khop %d = %.2f ngoai [%.1f..%.1f]"
                        % (name, i + 1, v, vlo, vhi)
                    )
                    return False
        zs = [
            float(
                self.kin.grip_mm(
                    self.kin.fk_tool0([p + t * (g - p) for p, g in zip(q_from, q_to)])
                )[2]
            )
            for t in np.linspace(0.0, 1.0, 11)
        ]
        z_min_ok = min(zs[0], zs[-1]) - self.drop_max_mm
        if min(zs) < z_min_ok:
            print(
                "DUONG DI TUT QUA: xuong %.1f mm, gioi han %.1f -> khong chay gi"
                % (min(zs), z_min_ok)
            )
            return False
        return True


__all__ = ["SafetyGate"]

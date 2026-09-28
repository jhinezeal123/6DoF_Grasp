"""tools/verify_refactor.py - So khop FK/IK/viewpoints: code CU (git) vs MOI (m750).

Chay tren server (can pinocchio, numpy, scipy):

    python tools/verify_refactor.py

Code CU lay tu git blob b877263 (commit Phase 0 - ban ductocbatdat.py 769 dong
CUOI CUNG truoc khi class hoa; backup/pre-refactor la ban 348 dong cu hon, thieu
_wait_stop/_move/views_6 nen khong dung so duoc). MOI goi m750. Neu moi dong in
"KHOP" thi refactor KHONG lam doi ket qua tinh toan.
"""
from __future__ import annotations

import subprocess
import sys
import types

import numpy as np
import pinocchio as pin

Q_SETS = [
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [10.0, -20.0, 30.0, -40.0, 50.0, 60.0],
    [-45.0, 25.0, -15.0, 20.0, -35.0, 80.0],
    [84.5, 64.5, -13.1, -87.1, -59.1, 143.4],
]


def load_old():
    """Load ductocbatdat.py cu tu git blob, exec thanh module (ten m750 clash tranh)."""
    src = subprocess.check_output(
        ["git", "show", "b877263:ductocbatdat.py"], text=True,
        encoding="utf-8")
    old = types.ModuleType("ductocbatdat_old")
    old.__dict__["__file__"] = "ductocbatdat_old.py"
    exec(src, old.__dict__)
    return old


def main() -> int:
    from m750.kinematics import ArmKinematics
    from m750.ik import IKSolver
    from m750.viewpoints import ring_views

    old = load_old()
    kin = ArmKinematics()
    ik = IKSolver(kin)

    ok = True

    # FK + grip
    for q in Q_SETS:
        M_old, M_new = old._fk(q), kin.fk_tool0(q)
        d_pos = np.linalg.norm(M_old.translation - M_new.translation)
        d_rot = np.abs(M_old.rotation - M_new.rotation).max()
        match = d_pos < 1e-12 and d_rot < 1e-12
        ok &= match
        print("FK %s: dpos=%.2e drot=%.2e %s"
              % (q, d_pos, d_rot, "KHOP" if match else "LECH !!!"))
        g_old, g_new = old._grip(M_old), kin.grip_mm(M_new)
        match = np.allclose(g_old, g_new, atol=1e-9)
        ok &= match
        print("  grip: %s" % ("KHOP" if match else "LECH !!! %s vs %s" % (g_old, g_new)))

    # IK vong tron
    for q in Q_SETS:
        T = kin.fk_tool0(q)
        r_old = old._solve_ik(T, [0.0] * 6)
        r_new = ik.solve(T, [0.0] * 6)
        match = (r_old[0] is None and r_new[0] is None) or (
            r_old[0] is not None and r_new[0] is not None
            and np.allclose(r_old[0], r_new[0], atol=1e-9)
            and abs(r_old[1] - r_new[1]) < 1e-9
            and abs(r_old[2] - r_new[2]) < 1e-9)
        ok &= match
        print("IK %s: old(ep=%.4f,eo=%.4f) new(ep=%.4f,eo=%.4f) %s"
              % (q, r_old[1], r_old[2], r_new[1], r_new[2],
                 "KHOP" if match else "LECH !!!"))

    # views_6 (cu can robot khi center=None -> dung center cu the)
    v_old = old.views_6(r=150, theta=30.0, center=[400.0, 0.0, 250.0])
    v_new = ring_views([400.0, 0.0, 250.0], r=150.0, theta=30.0)
    match = v_old == v_new
    ok &= match
    print("views_6: %s" % ("KHOP" if match else "LECH !!!"))

    # limits
    _, _, _, lo_old, hi_old = old._model()
    lo_new, hi_new = kin.limits_deg
    match = np.allclose(lo_old, lo_new) and np.allclose(hi_old, hi_new)
    ok &= match
    print("limits: %s" % ("KHOP" if match else "LECH !!!"))

    print()
    print("TAT CA KHOP - refactor khong lam doi ket qua tinh toan." if ok
          else "CO LECH - xem cac dong LECH o tren.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

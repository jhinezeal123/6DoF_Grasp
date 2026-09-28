"""tools/joint_check.py - Kiem tra write_angles (joint space) - theo P0-A cua tai lieu lab.

Tai lieu: myarm_sdk/docs/myarm_m750_pymycobot_api.md
  - P0-A chi gom: get_angles(), write_angles(angles, speed), stop()
  - KHONG dung is_in_position() lam tieu chi: firmware tolerance khong duoc cong bo
  - Tieu chi dung: e_max = max_i |q_do - q_dat|  (do tren GOC KHOP)

AN TOAN - bai hoc tu lan tay dap ban:
  +10 do cho CA 6 khop KHONG phai chuyen dong nho. Tool dich chuyen la TONG HOP
  cua 6 khop; trong tu the hien tai no lam tool tut 106mm -> dap ban.
  => MOI lenh phai qua check_fk() truoc khi gui. Khong bao gio gui mu.

Chay: python tools/joint_check.py   (can robot that + pinocchio)
"""
from __future__ import annotations

import sys
import time

import numpy as np

from m750.arm import MyArmM750
from m750.kinematics import ArmKinematics
from m750.spec import JOINT_NAMES, RobotSpec

STEP = 10.0        # do
SPEED = 20
SETTLE = 3.0       # giay
TOL = 1.0          # do
DROP_MAX = 25.0    # mm: tool duoc phep TUT xuong toi da tung lenh (tuong doi, khong phai z tuyet doi)
                   # vi tri hien tai z=-9mm (duoi mat de), nguong tuyet doi se chan het moi lenh

kin = ArmKinematics(RobotSpec())
LO, HI = kin.limits_deg


def fk(qd):
    return kin.grip_mm(kin.fk_tool0(qd))


def check_fk(q_dat, q_truoc, nhan=""):
    """Chan lenh neu vuot gioi han khop, hoac tool TUT xuong qua DROP_MAX. Tra True neu an toan."""
    for i, (v, lo, hi) in enumerate(zip(q_dat, LO, HI)):
        if not (lo <= v <= hi):
            print("  !!! CHAN: khop %d = %.2f ngoai [%.1f..%.1f]" % (i + 1, v, lo, hi))
            return False
    t_moi, t_cu = fk(q_dat), fk(q_truoc)
    if t_moi[2] - t_cu[2] < -DROP_MAX:
        print("  !!! CHAN: tool tut %.1f mm (qua %.1f), z: %.1f -> %.1f"
              % (t_cu[2] - t_moi[2], DROP_MAX, t_cu[2], t_moi[2]))
        return False
    d = t_moi - t_cu
    print("  %s tool dich: dx=%6.1f dy=%6.1f dz=%6.1f mm (z moi = %.1f)"
          % (nhan, d[0], d[1], d[2], t_moi[2]))
    return True


def emax(q_do, q_dat):
    """e_max = max_i |q_i,do - q_i,dat| - dung dinh nghia tai lieu lab."""
    return max(abs(a - b) for a, b in zip(q_do, q_dat))


def di_chuyen(arm, q_dat, nhan=""):
    """Gui write_angles SAU KHI da qua check_fk. Tra e_max, hoac None neu bi chan."""
    q_truoc = arm.open().get_angles()
    if not check_fk(q_dat, q_truoc, nhan):
        return None
    arm.open().write_angles(q_dat, SPEED)
    time.sleep(SETTLE)
    q_sau = arm.open().get_angles()
    e = emax(q_sau, q_dat)
    print("  %s do: %s" % (nhan, [round(x, 2) for x in q_sau]))
    print("  %s e_max = %.2f do -> %s" % (nhan, e, "OK" if e <= TOL else "LECH"))
    return e


def main() -> int:
    arm = MyArmM750()
    a = arm.open()
    print("=== TRANG THAI BAN DAU ===")
    q0 = a.get_angles()
    print("angles:", q0)
    print("tool  :", np.round(fk(q0), 1), "mm")
    print("gioi han khop (do):")
    for n, lo, hi in zip(JOINT_NAMES, LO, HI):
        print("  %-22s [%8.2f .. %8.2f]" % (n, lo, hi))
    print()

    print("=== 1. TUNG KHOP RIENG LE (an toan: 5 khop kia dung yen) ===")
    ket_qua = []
    for j in range(1, 7):
        goc_cu = a.get_angles()[j - 1]
        for dich in (goc_cu + STEP, goc_cu):
            q_dat = a.get_angles()
            q_dat[j - 1] = dich
            e = di_chuyen(arm, q_dat, "khop %d -> %.2f" % (j, dich))
            if e is not None:
                ket_qua.append(e)
            time.sleep(0.5)
    print()

    print("=== 2. CA 6 KHOP CUNG LUC - chi +10 do o 3 khop DAU (an toan hon) ===")
    q_truoc = a.get_angles()
    q_dat = list(q_truoc)
    for j in (1, 2, 3):
        q_dat[j - 1] += STEP * 0.5       # chi +5 do, va chi 3 khop than tren
    e = di_chuyen(arm, q_dat, "3 khop dau +5 do")
    if e is not None:
        ket_qua.append(e)
    time.sleep(0.5)

    print()
    print("=== 3. VE TU THE BAN DAU ===")
    e = di_chuyen(arm, list(q0), "ve goc")
    print()

    print("=== KET QUA ===")
    if ket_qua:
        print("e_max lon nhat trong %d lenh: %.2f do" % (len(ket_qua), max(ket_qua)))
        print("Khop bam chinh xac." if max(ket_qua) <= TOL
              else "Khop KHONG bam chinh xac - xem tung dong o tren.")
    else:
        print("Khong co lenh nao chay duoc.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

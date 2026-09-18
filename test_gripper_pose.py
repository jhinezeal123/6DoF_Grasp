"""Kiem tra set_gripper_pose KHONG lam tay di chuyen: chi mo port de doc goc khop,
roi giai IK. Khong co write_angles nao duoc goi.

Chay: python3 test_gripper_pose.py
Ky vong: 4 dong OK, ket thuc bang "TAT CA OK".
"""
import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation as R

import ductocbatdat as d

a = d._open()
q_now = a.get_angles()
assert isinstance(q_now, list) and len(q_now) == 6 and -1 not in q_now, "doc goc khop loi: %s" % (q_now,)

M = d._fk(q_now)
p = M.translation * 1000.0 + M.rotation @ np.array([0.0, 0.0, d.GRIP_L])
rpy = R.from_matrix(M.rotation).as_euler("xyz", degrees=True)
print("tu the  :", q_now)
print("gripper :", np.round(p, 1), " rpy:", np.round(rpy, 2))
print()


def thu(nhan, pos, rpy_t):
    Rg = R.from_euler("xyz", rpy_t, degrees=True).as_matrix()
    qg, ep, eo = d._solve_ik(
        pin.SE3(Rg, (np.array(pos, float) - Rg @ np.array([0.0, 0.0, d.GRIP_L])) / 1000.0), q_now)
    return qg is not None and ep < 1.0 and eo < 1.0


# _fk phai tu kiem chung: giai lai chinh tu the hien tai phai ra dung goc cu
assert thu("gio nguyen", p, rpy), "_fk/_solve_ik khong nhat quan - bo giai hong"
print("  OK  _solve_ik tim lai dung tu the hien tai")

# huong giu nguyen, nhich len cao - phai toi duoc
assert thu("+30mm Z", p + [0, 0, 30], rpy), "khong toi duoc +30mm Z"
print("  OK  +30mm Z toi duoc, huong giu nguyen")

# cac ca PHAI bi tu choi (chuc thang xuong vuot wrist_flex; dich ngoai tam voi)
assert not thu("chuc thang xuong", p, [180, 0, 0]), "chuc thang xuong khong duoc phep toi duoc"
print("  OK  chuc thang xuong bi tu choi")
assert not thu("xa 2000mm", [2000, 0, 300], rpy), "dich 2000mm khong duoc phep toi duoc"
print("  OK  dich ngoai tam voi bi tu choi")

print()
print("TAT CA OK - khong co write_angles nao duoc goi")

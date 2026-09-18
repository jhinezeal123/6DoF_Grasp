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

# get_gripper_pose phai tra dung thu set_gripper_pose nhan vao
gp = d.get_gripper_pose()
assert gp is not None and len(gp) == 6, "get_gripper_pose tra ve: %s" % (gp,)
assert np.allclose(gp[:3], p, atol=0.01), "vi tri get %s != FK %s" % (gp[:3], np.round(p, 2))
assert np.allclose(gp[3:], rpy, atol=0.01), "huong get %s != FK %s" % (gp[3:], np.round(rpy, 2))
print("  OK  get_gripper_pose khop voi _fk:", gp)


def thu(nhan, pos, rpy_t):
    Rg = R.from_euler("xyz", rpy_t, degrees=True).as_matrix()
    qg, ep, eo = d._solve_ik(
        pin.SE3(Rg, (np.array(pos, float) - Rg @ np.array([0.0, 0.0, d.GRIP_L])) / 1000.0), q_now)
    return qg is not None and ep < 1.0 and eo < 1.0


# _fk phai tu kiem chung: giai lai chinh tu the hien tai phai ra dung goc cu
assert thu("gio nguyen", p, rpy), "_fk/_solve_ik khong nhat quan - bo giai hong"
print("  OK  _solve_ik tim lai dung tu the hien tai")

# vong tron get -> set: IK tren chinh pose doc duoc phai ra lai dung goc khop cu
qg, ep, eo = d._solve_ik(pin.SE3(R.from_euler("xyz", gp[3:], degrees=True).as_matrix(),
                                 (np.array(gp[:3]) - R.from_euler("xyz", gp[3:], degrees=True).as_matrix()
                                  @ np.array([0.0, 0.0, d.GRIP_L])) / 1000.0), q_now)
assert ep < 0.01 and eo < 0.01, "vong tron get->set lech: %.3f mm / %.3f do" % (ep, eo)
assert max(abs(x - y) for x, y in zip(qg, q_now)) < 0.5, "goc khop khac: %s vs %s" % (np.round(qg, 2), q_now)
print("  OK  vong tron get -> set -> IK ra lai dung goc khop cu")

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

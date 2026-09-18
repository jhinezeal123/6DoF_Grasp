"""Kiem tra set_gripper_pose KHONG lam tay di chuyen: chi mo port de doc goc khop,
roi giai IK. Khong co write_angles nao duoc goi.

Chay: python3 test_gripper_pose.py
Ky vong: moi dong OK, ket thuc bang "TAT CA OK".
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


def huong(rpy_t):
    """Huong gripper chuc ra khi tool0 xoay theo rpy_t (huong ra ngoai = -z cua tool0)."""
    return R.from_euler("xyz", rpy_t, degrees=True).as_matrix() @ np.array([0.0, 0.0, -1.0])


def thu(pos, rpy_t):
    Rg = R.from_euler("xyz", rpy_t, degrees=True).as_matrix()
    qg, ep, eo = d._solve_ik(
        pin.SE3(Rg, (np.array(pos, float) - Rg @ np.array([0.0, 0.0, d.GRIP_L])) / 1000.0), q_now)
    return qg is not None and ep < 1.0 and eo < 1.0


# --- GRIP_L phai lay tu URDF, khong phai so mo ho ---
m, dt, idx, _, _ = d._model()
q = pin.neutral(m)
for i, ix in enumerate(idx):
    q[ix] = float(np.radians(q_now[i]))
pin.forwardKinematics(m, dt, q); pin.updateFramePlacements(m, dt)
Mt = dt.oMf[m.getFrameId("tool0")]
off = Mt.rotation.T @ (dt.oMf[m.getFrameId("gripper_base_link")].translation - Mt.translation) * 1000.0
assert np.allclose(off, [0, 0, d.GRIP_L], atol=0.01), \
    "GRIP_L=%.2f khong khop URDF gripper_base_link %s" % (d.GRIP_L, np.round(off, 2))
print("  OK  GRIP_L=%.1fmm khop URDF gripper_base_link" % d.GRIP_L)

# --- quy uoc huong: rpy=[0,0,0] la CHUC XUONG ---
assert np.allclose(huong([0, 0, 0]), [0, 0, -1]), "rpy=[0,0,0] khong phai chuc xuong"
assert np.allclose(huong([180, 0, 0]), [0, 0, 1]), "rpy=[180,0,0] khong phai chuc len"
print("  OK  quy uoc: rpy=[0,0,0] chuc XUONG, rpy=[180,0,0] chuc LEN")

# --- get_gripper_pose tra dung thu set_gripper_pose nhan vao ---
gp = d.get_gripper_pose()
assert gp is not None and len(gp) == 6, "get_gripper_pose tra ve: %s" % (gp,)
assert np.allclose(gp[:3], p, atol=0.01) and np.allclose(gp[3:], rpy, atol=0.01), \
    "get_gripper_pose %s khong khop _fk" % (gp,)
print("  OK  get_gripper_pose khop voi _fk:", gp)

# --- vong tron get -> IK phai ra lai dung goc khop cu ---
Rg = R.from_euler("xyz", gp[3:], degrees=True).as_matrix()
qg, ep, eo = d._solve_ik(
    pin.SE3(Rg, (np.array(gp[:3]) - Rg @ np.array([0.0, 0.0, d.GRIP_L])) / 1000.0), q_now)
assert ep < 0.01 and eo < 0.01, "vong tron get->set lech: %.3f mm / %.3f do" % (ep, eo)
assert max(abs(x - y) for x, y in zip(qg, q_now)) < 0.5, "goc khop khac: %s vs %s" % (np.round(qg, 2), q_now)
print("  OK  vong tron get -> set -> IK ra lai dung goc khop cu")

# --- huong giu nguyen, nhich len cao: phai toi duoc ---
assert thu(p + [0, 0, 30], rpy), "khong toi duoc +30mm Z"
print("  OK  +30mm Z toi duoc, huong giu nguyen")

# --- cac ca PHAI bi tu choi ---
assert not thu(p, [180, 0, 0]), "chuc thang LEN khong duoc phep toi duoc"
print("  OK  chuc thang len bi tu choi")
assert not thu([2000, 0, 300], rpy), "dich 2000mm khong duoc phep toi duoc"
print("  OK  dich ngoai tam voi bi tu choi")

print()
print("TAT CA OK - khong co write_angles nao duoc goi")

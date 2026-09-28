"""Test hardware (doc robot that, KHONG ra lenh chuyen dong): chay thu cong.

KHONG chay trong pytest mac dinh - doc /dev/ttyACM1 that. Chay tren server:

    python tests/test_gripper_pose_hardware.py

Ky vong: moi dong OK, ket thuc bang "TAT CA OK".
"""
import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation as R

from m750.control import ArmController


def huong(rpy_t):
    """Huong gripper chuc ra khi tool0 xoay theo rpy_t (huong ra ngoai = -z cua tool0)."""
    return R.from_euler("xyz", rpy_t, degrees=True).as_matrix() @ np.array([0.0, 0.0, -1.0])


def main() -> int:
    ctrl = ArmController()
    a = ctrl.arm.open()
    q_now = a.get_angles()
    assert isinstance(q_now, list) and len(q_now) == 6 and -1 not in q_now, \
        "doc goc khop loi: %s" % (q_now,)

    M = ctrl.kin.fk_tool0(q_now)
    p = M.translation * 1000.0 + M.rotation @ np.array([0.0, 0.0, ctrl.spec.grip_l_mm])
    rpy = R.from_matrix(M.rotation).as_euler("xyz", degrees=True)
    print("tu the  :", q_now)
    print("gripper :", np.round(p, 1), " rpy:", np.round(rpy, 2))
    print()

    # --- GRIP_L phai lay tu URDF, khong phai so mo ho ---
    m, d = ctrl.kin.m, ctrl.kin.d
    q = pin.neutral(m)
    for i, ix in enumerate(ctrl.kin.idx):
        q[ix] = float(np.radians(q_now[i]))
    pin.forwardKinematics(m, d, q)
    pin.updateFramePlacements(m, d)
    Mt = d.oMf[m.getFrameId("tool0")]
    off = Mt.rotation.T @ (d.oMf[m.getFrameId("gripper_base_link")].translation - Mt.translation) * 1000.0
    assert np.allclose(off, [0, 0, ctrl.spec.grip_l_mm], atol=0.01), \
        "GRIP_L=%.2f khong khop URDF gripper_base_link %s" % (ctrl.spec.grip_l_mm, np.round(off, 2))
    print("  OK  GRIP_L=%.1fmm khop URDF gripper_base_link" % ctrl.spec.grip_l_mm)

    # --- quy uoc huong: rpy=[0,0,0] la CHUC XUONG ---
    assert np.allclose(huong([0, 0, 0]), [0, 0, -1]), "rpy=[0,0,0] khong phai chuc xuong"
    assert np.allclose(huong([180, 0, 0]), [0, 0, 1]), "rpy=[180,0,0] khong phai chuc len"
    print("  OK  quy uoc: rpy=[0,0,0] chuc XUONG, rpy=[180,0,0] chuc LEN")

    # --- gripper_pose() tra dung thu move_gripper_to() nhan vao ---
    gp = ctrl.gripper_pose()
    assert gp is not None and len(gp) == 6, "gripper_pose() tra ve: %s" % (gp,)
    assert np.allclose(gp[:3], p, atol=0.01) and np.allclose(gp[3:], rpy, atol=0.01), \
        "gripper_pose %s khong khop fk_tool0" % (gp,)
    print("  OK  gripper_pose khop voi fk_tool0:", gp)

    # --- vong tron get -> IK phai ra lai dung goc khop cu ---
    Rg = R.from_euler("xyz", gp[3:], degrees=True).as_matrix()
    qg, ep, eo = ctrl.ik.solve(
        pin.SE3(Rg, (np.array(gp[:3]) - Rg @ np.array([0.0, 0.0, ctrl.spec.grip_l_mm])) / 1000.0), q_now)
    assert ep < 0.01 and eo < 0.01, "vong tron get->set lech: %.3f mm / %.3f do" % (ep, eo)
    assert max(abs(x - y) for x, y in zip(qg, q_now)) < 0.5, \
        "goc khop khac: %s vs %s" % (np.round(qg, 2), q_now)
    print("  OK  vong tron get -> set -> IK ra lai dung goc khop cu")

    def thu(pos, rpy_t):
        Rg_ = R.from_euler("xyz", rpy_t, degrees=True).as_matrix()
        qg_, ep_, eo_ = ctrl.ik.solve(
            pin.SE3(Rg_, (np.array(pos, float) - Rg_ @ np.array([0.0, 0.0, ctrl.spec.grip_l_mm])) / 1000.0),
            q_now)
        return qg_ is not None and ep_ < 1.0 and eo_ < 1.0

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Test offline: FK/IK/viewpoints/spec - KHONG can robot that, KHONG can port.

Chay: python -m pytest tests/test_kinematics.py -v
(Can: pinocchio, scipy, numpy - nhu moi truong server.)
"""
import numpy as np
import pinocchio as pin
import pytest
from scipy.spatial.transform import Rotation as R

from m750.ik import IKSolver
from m750.kinematics import ArmKinematics
from m750.spec import GRIP_L_MM, URDF_PATH, JOINT_NAMES
from m750.viewpoints import ring_views


@pytest.fixture(scope="module")
def kin():
    return ArmKinematics()


@pytest.fixture(scope="module")
def ik(kin):
    return IKSolver(kin)


Q_SETS = [
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [10.0, -20.0, 30.0, -40.0, 50.0, 60.0],
    [-45.0, 25.0, -15.0, 20.0, -35.0, 80.0],
    [84.5, 64.5, -13.1, -87.1, -59.1, 143.4],  # do thuc tu ktmt recap
]


class TestSpec:
    def test_urdf_trong_package(self):
        assert URDF_PATH.is_file(), "URDF phai nam trong m750.model"

    def test_grip_l_khop_urdf(self, kin):
        """GRIP_L phai lay tu URDF (gripper_base_link), khong phai so mo ho."""
        m, d = kin.m, kin.d
        q = pin.neutral(m)
        pin.forwardKinematics(m, d, q)
        pin.updateFramePlacements(m, d)
        Mt = d.oMf[m.getFrameId("tool0")]
        off = Mt.rotation.T @ (d.oMf[m.getFrameId("gripper_base_link")].translation
                               - Mt.translation) * 1000.0
        assert np.allclose(off, [0, 0, GRIP_L_MM], atol=0.01), \
            "GRIP_L=%.2f khong khop URDF gripper_base_link %s" % (GRIP_L_MM, np.round(off, 2))

    def test_joint_names(self):
        assert len(JOINT_NAMES) == 6


class TestFK:
    def test_fk_dinh_tinh(self, kin):
        """FK la thuan toan: cung input -> dung output (goi 2 lan)."""
        M1 = kin.fk_tool0(Q_SETS[1])
        M2 = kin.fk_tool0(Q_SETS[1])
        assert np.allclose(M1.translation, M2.translation)
        assert np.allclose(M1.rotation, M2.rotation)

    def test_fk_khop_0(self, kin):
        """Q=0: tool0 phai nam tren mat phong xy (z>0) trong tam voi."""
        M = kin.fk_tool0([0] * 6)
        assert M.translation[2] > 0.05

    def test_grip_mm(self, kin):
        """grip_mm = tool0 + R @ [0,0,GRIP_L] (mm)."""
        M = kin.fk_tool0(Q_SETS[1])
        expect = M.translation * 1000.0 + M.rotation @ np.array([0.0, 0.0, GRIP_L_MM])
        assert np.allclose(kin.grip_mm(M), expect)


class TestIK:
    @pytest.mark.parametrize("q", Q_SETS)
    def test_ik_vong_tron(self, ik, kin, q):
        """FK(q) lam target -> IK giai lai tu seed 0 -> phai ra ep<0.05mm, eo<0.05do."""
        T = kin.fk_tool0(q)
        q_goal, ep, eo = ik.solve(T, [0.0] * 6)
        assert q_goal is not None
        assert ep < 0.05, "sai so vi tri %.4f mm" % ep
        assert eo < 0.05, "sai so huong %.4f do" % eo

    def test_ik_dinh_tinh(self, ik, kin):
        """IK dinh tinh (seed rng(0) co dinh): goi 2 lan -> cung nghiem."""
        T = kin.fk_tool0(Q_SETS[2])
        r1 = ik.solve(T, [0.0] * 6)
        r2 = ik.solve(T, [0.0] * 6)
        assert r1[0] == pytest.approx(r2[0])
        assert r1[1] == pytest.approx(r2[1])

    def test_ik_pose_quay(self, ik, kin):
        """Pose quay 180/20/-30 tai [0.35,-0.05,0.12] -> van phai giai duoc (trong vung lam viec)."""
        T2 = pin.SE3(
            R.from_euler("xyz", [180, 20, -30], degrees=True).as_matrix(),
            np.array([0.35, -0.05, 0.12]),
        )
        q_goal, ep, eo = ik.solve(T2, [10.0, -20.0, 30.0, -40.0, 50.0, 60.0])
        assert q_goal is not None and ep < 0.05 and eo < 0.05

    def test_ik_quy_uoc_grip(self, ik, kin):
        """rpy=[0,0,0] = CHUC XUONG: huong ra ngoai (=-z tool0) phai = [0,0,-1]."""
        Rg = R.from_euler("xyz", [0, 0, 0], degrees=True).as_matrix()
        huong = Rg @ np.array([0.0, 0.0, -1.0])
        assert np.allclose(huong, [0, 0, -1])
        Rg180 = R.from_euler("xyz", [180, 0, 0], degrees=True).as_matrix()
        assert np.allclose(Rg180 @ np.array([0.0, 0.0, -1.0]), [0, 0, 1])


class TestLimits:
    def test_limits_urdf(self, kin):
        lo, hi = kin.limits_deg
        assert len(lo) == 6 and len(hi) == 6
        assert all(low < high for low, high in zip(lo, hi))


class TestViewpoints:
    def test_hinh_dang(self):
        v = ring_views([400.0, 0.0, 250.0], r=150.0, theta=30.0)
        assert len(v) == 6
        assert all(len(p) == 6 for p in v)

    def test_khoang_cach_tam(self):
        """6 pose cach tam dung r (mm)."""
        import numpy as np
        center = np.array([400.0, 0.0, 250.0])
        r = 150.0
        for p in ring_views(list(center), r=r, theta=30.0):
            d = np.linalg.norm(np.array(p[:3]) - center)
            assert abs(d - r) < 0.5, "khoang cach %.2f != r %.2f" % (d, r)

"""control.py - ArmController: API muc ung dung tren MyArmM750 + IK + safety.

Quy uoc pose (GIU NGUYEN, da kiem chung tren tay that):
  - [x, y, z mm | rx, ry, rz do Euler XYZ] trong he URDF
  - rpy=[0,0,0] la gripper CHUC THANG XUONG (huong ra ngoai cua tool = -z cua
    tool0); rpy=[180,0,0] la chuc thang LEN
  - doc roi ghi lai duoc: move_gripper_to(*gripper_pose()) khong lam tay di chuyen
"""
from __future__ import annotations

import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation as R

from .arm import MyArmM750, _ok6
from .ik import IKSolver
from .kinematics import ArmKinematics
from .safety import SafetyGate
from .spec import RobotSpec
from .viewpoints import ring_views


class ArmController:
    """Compose: MyArmM750 + ArmKinematics + IKSolver + SafetyGate."""

    def __init__(self, arm: MyArmM750 | None = None,
                 kin: ArmKinematics | None = None,
                 ik: IKSolver | None = None,
                 safety: SafetyGate | None = None,
                 spec: RobotSpec | None = None) -> None:
        self.spec = spec or RobotSpec()
        self.arm = arm or MyArmM750(self.spec)
        self.kin = kin or ArmKinematics(self.spec)
        self.ik = ik or IKSolver(self.kin)
        self.safety = safety or SafetyGate(self.kin)

    # ------------------------------------------------------------- doc
    def gripper_pose(self) -> list | None:
        """Pose gripper hien tai [x,y,z mm | rx,ry,rz do Euler XYZ], he URDF.

        CUNG quy uoc voi move_gripper_to -> doc roi ghi lai duoc:
            move_gripper_to(*gripper_pose())   # khong di chuyen
        Tra None neu loi doc goc khop.
        """
        q = self.arm.q_deg
        if q is None:
            return None
        M = self.kin.fk_tool0(q)
        return ([round(float(v), 2) for v in self.kin.grip_mm(M)]
                + [round(float(v), 2) for v in R.from_matrix(M.rotation).as_euler("xyz", degrees=True)])

    def state(self) -> dict:
        """Trang thai: 6 khop (do), do mo gripper (0..100), pose gripper URDF va toa do firmware.

        'gripper' la con so dung duoc: cung quy uoc voi gripper_pose() -> truyen
        thang duoc vao move_gripper_to. 'fw' la toa do firmware vendor, KHONG
        dung cho IK (khac he quy chieu). -1 = chua bat dien/loi doc.
        """
        a = self.arm.open()
        return {
            "joints_deg": a.get_angles(),
            "gripper_pct": a.get_gripper_value(),
            "gripper_pose": self.gripper_pose(),
            "fw_coords": a.get_coords(),
        }

    def print_state(self) -> None:
        """In trang thai theo dung format ban cu (ductocbatdat.state())."""
        s = self.state()
        print("khop   :", [round(x, 2) for x in s["joints_deg"]])
        print("grip   :", s["gripper_pct"])
        print("gripper:", s["gripper_pose"], " <- he URDF, dung duoc")
        print("fw     :", [round(x, 2) for x in s["fw_coords"]], " <- firmware vendor")

    def ring_views(self, r: float = 150.0, theta: float = 30.0,
                   center=None) -> list | None:
        """6 pose quanh tam (GraspNeRF IV-B). Center mac dinh = gripper hien tai.

        Tra None neu doc goc khop loi.
        """
        if center is None:
            p = self.gripper_pose()
            if p is None:
                return None
            center = p[:3]
        return ring_views(center, r, theta)

    # ------------------------------------------------------------- ghi
    def move_tcp_to(self, coords, speed: int, tol: float = 2.0,
                    timeout_s: float = 30) -> bool:
        """TCP [x,y,z mm, rx,ry,rz do Euler] -> IK (Pinocchio+URDF) -> write_angles. Tra True neu toi noi.

        KHONG dung write_coords: firmware IK cua vendor lech URDF 50mm, sai so
        do that 17-106mm, tai lieu lab (myarm_m750_pymycobot_api.md muc 8.1)
        loai API Cartesian khoi driver. write_angles do that bam chinh xac
        e_max < 0.6 do.
        IK khong giai duoc (ngoai tam voi / vuot gioi han khop) -> tra False,
        KHONG chay gi.
        """
        q_now = self.arm.q_deg
        if q_now is None:
            return False
        q_goal, ep, eo = self.ik.solve(
            pin.SE3(R.from_euler("xyz", coords[3:], degrees=True).as_matrix(),
                    np.array(coords[:3], dtype=float) / 1000.0), q_now)
        if q_goal is None or ep > tol or eo > tol:
            print("POSE KHONG TOI DUOC: vi tri lech %.2f mm, huong lech %.2f do -> khong chay gi"
                  % (ep, eo))
            return False
        self.arm.write_joints(q_goal, speed, timeout_s, tol=tol)
        q_end = self.arm.open().get_angles()
        if not _ok6(q_end):
            print("doc goc khop loi sau khi chay")
            return False
        M = self.kin.fk_tool0(q_end)
        ep = float(np.linalg.norm(M.translation * 1000.0 - np.array(coords[:3], dtype=float)))
        eo = float(np.degrees(np.linalg.norm(R.from_matrix(
            M.rotation.T @ R.from_euler("xyz", coords[3:], degrees=True).as_matrix()).as_rotvec())))
        print("tool0 lech %.2f mm / %.2f do (cho phep %.1f): %s"
              % (ep, eo, tol, "TOI" if ep <= tol and eo <= tol else "LECH"))
        self.print_state()
        return ep <= tol and eo <= tol

    def move_gripper_to(self, x, y, z, rx=0.0, ry=0.0, rz=0.0, speed: int = 20,
                        tol_pos: float = 1.0, tol_rot: float = 1.0,
                        timeout_s: float = 30, drop_max: float = 30.0) -> bool:
        """Dua GRIPPER toi pose [x,y,z mm | rx,ry,rz do Euler XYZ] trong he URDF. Tra True/False.

        Gripper = tam gap (gripper_base_link), cach tool0 87mm (spec.grip_l_mm).
        Huong ra ngoai cua tool la -z cua tool0, nen rpy=[0,0,0] la CHUC THANG
        XUONG (da kiem chung 2 cach: truc -z cua tool0 va vector
        flange->gripper_base deu = [0,0,-1] khi rpy=0). rpy=[180,0,0] la chuc
        thang LEN. Nghieng: rx=180 khong phai chuc xuong. Muon nghieng thi doi
        ry (vd ry=30 -> nghieng 30 do).
        Giai khong ra (vi tri HOAC huong lech qua tol) -> tra False, KHONG chay gi.
        Kiem tra them duong di: noi suy goc khop, neu tut qua drop_max so voi
        ca hai dau -> chan.
        """
        q_now = self.arm.q_deg
        if q_now is None:
            return False

        Rg = R.from_euler("xyz", [rx, ry, rz], degrees=True).as_matrix()
        # dich cho gripper -> quy ve tool0 (bo offset GRIP_L doc truc z cua tool0)
        p_tool = np.array([x, y, z], dtype=float) - Rg @ np.array(
            [0.0, 0.0, self.spec.grip_l_mm])

        q_goal, ep, eo = self.ik.solve(pin.SE3(Rg, p_tool / 1000.0), q_now)
        if q_goal is None or ep > tol_pos or eo > tol_rot:
            print("POSE KHONG TOI DUOC: vi tri lech %.2f mm (cho phep %.1f),"
                  " huong lech %.2f do (cho phep %.1f) -> khong chay gi"
                  % (ep, tol_pos, eo, tol_rot))
            return False

        # duong di: tay noi suy theo GOC KHOP, kiem tra co tut qua sau khong
        if not self.safety.check_path(q_now, q_goal):
            return False

        self.arm.write_joints(q_goal, speed, timeout_s, tol=0.15, tries=6)
        # 0.15 do x 470mm ~ 1.2mm: phai chat moi duoi tol_pos
        q_end = self.arm.open().get_angles()
        if not _ok6(q_end):
            print("doc goc khop loi sau khi chay")
            return False
        M = self.kin.fk_tool0(q_end)
        p_end = self.kin.grip_mm(M)
        err = float(np.linalg.norm(p_end - np.array([x, y, z], dtype=float)))
        eo = float(np.degrees(np.linalg.norm(
            R.from_matrix(M.rotation.T @ Rg).as_rotvec())))
        print("gripper (URDF): %s  | dich: %s" % (np.round(p_end, 1), [round(v, 1) for v in (x, y, z)]))
        print("sai so gripper %.2f mm (cho phep %.1f) | huong lech %.2f do (cho phep %.1f): %s"
              % (err, tol_pos, eo, tol_rot, "TOI" if err <= tol_pos and eo <= tol_rot else "LECH"))
        return err <= tol_pos and eo <= tol_rot


__all__ = ["ArmController"]

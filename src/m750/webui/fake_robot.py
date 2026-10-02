"""
webui/fake_robot.py - Lop FakeRobot: mat na ROS 2 + trang thai MuJoCo de hien thi.

THAY DOI LON so voi ban cu:
  - Khong con tu mo phong vat ly lam nguon su that. Nguon su that DUY NHAT la
    /myarm/state/joint_state (do myarm_robot_driver phat, du backend la
    FakeRobotArm hay MyArmM750RobotArm).
  - MuJoCo chi con la lop HIEN THI: sync() keo goc khop do duoc tu ROS vao m/d
    roi mj_forward. Khong mj_step, khong _virt_q, khong _settle.
  - Khong con mode independent/synchronized: chay backend nao la viec cua
    launch profile (services.yaml), khong phai cua lop nay. Do do cung khong
    con duong nao de vo tinh ra lenh cho robot that khi dang o che do mo phong.
  - Moi lenh (change/set_joint/FollowTrajectory) thua huong tu Robot va di qua
    /myarm/command/joint_goal, khong tac dong thang vao m/d.

MuJoCo KHONG thread-safe: chi mot thread duoc goi sync()/mj_qpos. Vong render
cua web_control phai la thread do.
"""
from __future__ import annotations

import os
from typing import Optional

# Phai dat truoc khi import mujoco: Windows dung wgl, Linux headless dung egl.
os.environ.setdefault("MUJOCO_GL", "wgl" if os.name == "nt" else "egl")

import mujoco  # noqa: E402 - phai sau khi dat MUJOCO_GL
import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from m750.ros.robot import Robot  # noqa: E402
from m750.ros.bridge import model_rad  # noqa: E402

# Scene trong package: m750/model/scene_vla.xml
from m750.spec import model_dir  # noqa: E402

SCENE_XML = os.path.join(str(model_dir()), "scene_vla.xml")


class FakeRobot(Robot):
    """
    myArm M750 trong MuJoCo, hien thi theo trang thai that tren ROS.

    Dung duoc ca khi backend la fake_robot_arm (khong co phan cung) lan khi la
    myarm_m750_robot_arm (phan cung that) - ca hai deu phat cung mot topic.

    Duoi day chi giu phan HIEN THI. Moi lenh deu thua huong tu Robot.
    """

    def __init__(self, scene_path: Optional[str] = None) -> None:
        super().__init__()
        # Keep the absolute scene path so the web renderer can open an
        # independent MuJoCo model in a worker process.  MuJoCo/EGL rendering
        # can block the Python interpreter on headless Jetson systems; sharing
        # the renderer with the ROS bridge process then prevents ROS callbacks
        # from updating the hardware state cache.
        self.scene_path = os.path.abspath(scene_path or SCENE_XML)
        self.m = mujoco.MjModel.from_xml_path(self.scene_path)
        self.d = mujoco.MjData(self.m)
        self.qpos_addrs = [self.m.joint(n).qposadr[0] for n in self.JOINT_NAMES]
        self.dof_addrs = [self.m.joint(n).dofadr[0] for n in self.JOINT_NAMES]
        self.joint_ids = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)
                          for n in self.JOINT_NAMES]
        self.flange_body_id = mujoco.mj_name2id(
            self.m, mujoco.mjtObj.mjOBJ_BODY, "flange_link")
        self.tool0_site_id = mujoco.mj_name2id(
            self.m, mujoco.mjtObj.mjOBJ_SITE, "tool0_preview")
        self.synced_pose: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ TCP preview
    # The ROS kinematics stack uses the URDF ``tool0`` frame as its TCP.  The
    # MuJoCo display model intentionally contains the physical gripper but no
    # separate tool0 body, so reconstruct the fixed flange -> tool0 transform
    # here.  This is used only to preview a pose; the real command still goes
    # through Robot.set_tcp_pose() and myarm_kinematics.
    _TOOL0_OFFSET = np.array([0.118, 0.0, 0.0], dtype=np.float64)
    _TOOL0_ROT = np.array(
        [[0.0, 0.0, -1.0],
         [0.0, 1.0, 0.0],
         [1.0, 0.0, 0.0]], dtype=np.float64
    )  # Ry(-pi/2), matching flange_to_tool0_joint in the URDF.

    @staticmethod
    def _quat_xyzw_to_mat(quaternion) -> np.ndarray:
        q = np.asarray(quaternion, dtype=np.float64).reshape(4)
        norm = float(np.linalg.norm(q))
        if not np.isfinite(norm) or norm < 1e-9:
            raise ValueError("quaternion phai la vector 4 so huu han va khac 0")
        x, y, z, w = q / norm
        return np.array([
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),
             2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z),
             2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w),
             1.0 - 2.0 * (x * x + y * y)],
        ], dtype=np.float64)

    def _tool0_pose(self):
        if self.flange_body_id < 0:
            raise RuntimeError("MuJoCo model khong co body flange_link")
        r_flange = self.d.xmat[self.flange_body_id].reshape(3, 3)
        p_flange = self.d.xpos[self.flange_body_id]
        p_tool = p_flange + r_flange @ self._TOOL0_OFFSET
        r_tool = r_flange @ self._TOOL0_ROT
        return p_tool.copy(), r_tool.copy()

    def tcp_pose(self):
        """Return the ghost tool0 pose as (position, ROS quaternion)."""
        position, rotation = self._tool0_pose()
        return position, Rotation.from_matrix(rotation).as_quat()

    @staticmethod
    def _orientation_error(r_current, r_target) -> np.ndarray:
        # First-order SO(3) error used by the damped least-squares IK update.
        return 0.5 * (
            np.cross(r_current[:, 0], r_target[:, 0])
            + np.cross(r_current[:, 1], r_target[:, 1])
            + np.cross(r_current[:, 2], r_target[:, 2])
        )

    def preview_tcp_pose(self, position, quaternion, *, max_iterations: int = 120):
        """Solve a display-only IK target and put the ghost robot at that pose.

        The method never publishes a ROS command.  It modifies MuJoCo only, so
        the web UI can show the requested tool0 pose in the third-person view
        before the user submits the real ``set_tcp_pose`` command.
        """
        target_p = np.asarray(position, dtype=np.float64).reshape(3)
        target_r = self._quat_xyzw_to_mat(quaternion)
        if not np.all(np.isfinite(target_p)):
            raise ValueError("position phai gom 3 so huu han")

        original_qpos = self.d.qpos.copy()
        original_synced_pose = (None if self.synced_pose is None
                                else self.synced_pose.copy())
        q = np.asarray([self.d.qpos[a] for a in self.qpos_addrs], dtype=np.float64)
        ranges = np.asarray(self.m.jnt_range[self.joint_ids], dtype=np.float64)
        lo = np.maximum(self.rad_min, ranges[:, 0])
        hi = np.minimum(self.rad_max, ranges[:, 1])
        if np.any(lo > hi):
            raise RuntimeError("mien gioi han IK rong")
        q = np.clip(q, lo, hi)

        position_error = np.inf
        orientation_error = np.inf
        converged = False
        iterations = 0
        try:
            for iterations in range(1, max_iterations + 1):
                self._write(q)
                current_p, current_r = self._tool0_pose()
                e_p = target_p - current_p
                e_r = self._orientation_error(current_r, target_r)
                position_error = float(np.linalg.norm(e_p))
                orientation_error = float(np.linalg.norm(e_r))
                if position_error <= 0.002 and orientation_error <= 0.035:
                    converged = True
                    break

                jac_p = np.zeros((3, self.m.nv), dtype=np.float64)
                jac_r = np.zeros((3, self.m.nv), dtype=np.float64)
                mujoco.mj_jac(self.m, self.d, jac_p, jac_r,
                              current_p, self.flange_body_id)
                jac = np.vstack((jac_p[:, self.dof_addrs], jac_r[:, self.dof_addrs]))
                error = np.concatenate((e_p, e_r))
                damping = 0.03
                lhs = jac @ jac.T + (damping * damping) * np.eye(6)
                dq = jac.T @ np.linalg.solve(lhs, error)
                dq = np.clip(dq, -0.15, 0.15)
                q = np.clip(q + dq, lo, hi)

            self._write(q)
            actual_p, actual_r = self._tool0_pose()
            position_error = float(np.linalg.norm(target_p - actual_p))
            orientation_error = float(np.linalg.norm(
                self._orientation_error(actual_r, target_r)))
            converged = position_error <= 0.002 and orientation_error <= 0.035
            if not converged:
                self.d.qpos[:] = original_qpos
                mujoco.mj_forward(self.m, self.d)
                self.synced_pose = original_synced_pose
            elif self.tool0_site_id >= 0:
                self.m.site_rgba[self.tool0_site_id, 3] = 0.95

            return {
                "ok": bool(converged),
                "qpos": q.tolist() if converged else None,
                "actual_position": actual_p.tolist() if converged else None,
                "position_error_m": position_error,
                "orientation_error_rad": orientation_error,
                "iterations": iterations,
                "message": "IK preview thanh cong" if converged
                           else "Khong tim thay IK phu hop trong gioi han model",
            }
        except Exception:
            self.d.qpos[:] = original_qpos
            mujoco.mj_forward(self.m, self.d)
            self.synced_pose = original_synced_pose
            raise

    # ------------------------------------------------------------------ hien thi
    def sync(self) -> bool:
        """
        Keo goc khop DO DUOC tu ROS vao m/d roi mj_forward.

        Phai doi qua model_rad() - tuc la TRU OFFSETS_DEG. Gia tri tren topic la
        goc tho; con robot ao phai ve o goc model DA HIEU CHUAN, neu khong no
        lech khoi tay that dung bang OFFSETS_DEG. Che do OFFSET tren UI keo
        chinh con robot ao nay ve dung tu the thuc te, roi ghi lai offset.

        Goi tu vong render (mot thread duy nhat). Tra False neu chua co feedback.
        """
        measured = self._b.joint_rad
        if measured is None:
            return False
        self._write(model_rad(measured))
        self._hide_tool0_preview()
        return True

    def set_display(self, q_rad) -> None:
        """Chi doi hinh con robot ao trong MuJoCo, KHONG gui gi len ROS.

        Khac han set_joint/change: chung thua huong tu Robot nen di qua
        /myarm/command/joint_goal, tuc la lam TAY THAT chay. Che do chinh offset
        can keo slider de can con robot ao cho khop voi tay that ma tay that
        phai dung yen tuyet doi.

        Ben goi phai giu _mj_lock (MuJoCo khong thread-safe).
        """
        self._write(np.asarray(q_rad, dtype=np.float64).flatten()[:6])
        self._hide_tool0_preview()

    def _hide_tool0_preview(self) -> None:
        if self.tool0_site_id >= 0:
            self.m.site_rgba[self.tool0_site_id, 3] = 0.0

    def _write(self, q_rad) -> None:
        for i, addr in enumerate(self.qpos_addrs):
            self.d.qpos[addr] = float(q_rad[i])
        mujoco.mj_forward(self.m, self.d)
        self.synced_pose = np.asarray(q_rad, dtype=np.float64).copy()

    @property
    def mj_qpos(self) -> np.ndarray:
        """Toan bo qpos cua scene MuJoCo (6 khop + do tu do cua vat the)."""
        return self.d.qpos.copy()

    def reset_scene(self, initial_qpos=None, initial_gripper: float = 0.0) -> bool:
        """
        Dua cac vat the tren ban ve vi tri ban dau va xoa van toc.

        Goc khop luon lay lai tu ROS sau khi reset - khong tu dat pose cho sim,
        vi sim khong con la nguon su that. initial_gripper duoc giu lai chi de
        khong lam vo loi goi cu; do mo that do /myarm/gripper/state quyet dinh.
        """
        mujoco.mj_resetData(self.m, self.d)
        self._hide_tool0_preview()
        if initial_qpos is not None:
            self._write(np.asarray(initial_qpos, dtype=np.float64).flatten()[:6])
        else:
            self.sync()
        return True

    def mirror_real(self) -> None:
        """Giu lai ten cu cho vong render. Nay la sync() - sim luon theo ROS."""
        self.sync()

    def close(self) -> None:
        self._b.close()

    # ------------------------------------------------------------------ da bo
    # mode / set_mode / get_mode / copy_from / _settle / _virt_* : da xoa.
    # Chung ton tai chi de giu hai nguon trang thai song song (sim va that),
    # va chinh no la nguon loi. Chon backend o services.yaml thay vi o day.


__all__ = ["FakeRobot", "SCENE_XML"]

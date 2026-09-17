"""
fake_robot/fake_robot.py - Lop FakeRobot: mat na ROS 2 + trang thai MuJoCo de hien thi.

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

from program.robot.robot import Robot  # noqa: E402
from program.ros_bridge import model_rad  # noqa: E402

# robot_model/ nam canh user/ (user/robot_model/scene_vla.xml)
USER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE_XML = os.path.join(USER_DIR, "robot_model", "scene_vla.xml")


class FakeRobot(Robot):
    """
    myArm M750 trong MuJoCo, hien thi theo trang thai that tren ROS.

    Dung duoc ca khi backend la fake_robot_arm (khong co phan cung) lan khi la
    myarm_m750_robot_arm (phan cung that) - ca hai deu phat cung mot topic.

    Duoi day chi giu phan HIEN THI. Moi lenh deu thua huong tu Robot.
    """

    def __init__(self, scene_path: Optional[str] = None) -> None:
        super().__init__()
        self.m = mujoco.MjModel.from_xml_path(scene_path or SCENE_XML)
        self.d = mujoco.MjData(self.m)
        self.qpos_addrs = [self.m.joint(n).qposadr[0] for n in self.JOINT_NAMES]
        self.synced_pose: Optional[np.ndarray] = None

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

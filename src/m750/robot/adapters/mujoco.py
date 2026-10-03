"""MuJoCo implementation of RobotDriver with no ROS dependency."""

from __future__ import annotations

import math
import os
from typing import Optional, Sequence

import mujoco
import numpy as np

from m750.spec import JOINT_NAMES, model_dir

from ..contracts import RobotDriver
from ..types import JointLimits, RobotState, TcpPose


DEFAULT_SCENE = os.path.join(str(model_dir()), "scene_vla.xml")


def _finite(values: Sequence[float], size: int):
    try:
        result = np.asarray(tuple(values), dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if result.size != size or not np.all(np.isfinite(result)):
        return None
    return result


class MujocoRobotDriver(RobotDriver):
    """Kinematic simulator backend.

    Commands mutate only the MuJoCo model/data owned by this object. No ROS
    publisher, serial port or real-robot facade is reachable from this class.
    """

    _TOOL0_OFFSET = np.array([0.118, 0.0, 0.0], dtype=np.float64)
    _TOOL0_ROT = np.array(
        [
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    def __init__(self, scene_path: Optional[str] = None) -> None:
        self.scene_path = os.path.abspath(scene_path or DEFAULT_SCENE)
        self.model = mujoco.MjModel.from_xml_path(self.scene_path)
        self.data = mujoco.MjData(self.model)

        self._joint_names = tuple(JOINT_NAMES)
        self._joint_ids = tuple(
            mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                name,
            )
            for name in self._joint_names
        )
        if any(joint_id < 0 for joint_id in self._joint_ids):
            raise RuntimeError("MuJoCo model is missing one or more robot joints")

        self._qpos_addrs = tuple(
            int(self.model.jnt_qposadr[joint_id])
            for joint_id in self._joint_ids
        )
        self._dof_addrs = tuple(
            int(self.model.jnt_dofadr[joint_id])
            for joint_id in self._joint_ids
        )

        self._flange_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "flange_link",
        )
        if self._flange_body_id < 0:
            raise RuntimeError("MuJoCo model is missing flange_link")

        self._left_gripper_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "left_gripper_joint",
        )
        self._right_gripper_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "right_gripper_joint",
        )
        self._gripper_opening_m = 0.0
        mujoco.mj_forward(self.model, self.data)

    @property
    def joint_limits_rad(self) -> JointLimits:
        return tuple(
            (
                float(self.model.jnt_range[joint_id, 0]),
                float(self.model.jnt_range[joint_id, 1]),
            )
            for joint_id in self._joint_ids
        )

    @property
    def max_gripper_opening_m(self) -> float:
        if self._left_gripper_id < 0:
            return 0.0
        upper = float(self.model.jnt_range[self._left_gripper_id, 1])
        return max(0.0, 2.0 * upper)

    def _read_joints(self):
        return tuple(float(self.data.qpos[address]) for address in self._qpos_addrs)

    def _write_joints(self, values) -> None:
        for address, value in zip(self._qpos_addrs, values):
            self.data.qpos[address] = float(value)
        mujoco.mj_forward(self.model, self.data)

    @staticmethod
    def _mat_to_quat_xyzw(matrix) -> tuple:
        m = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
        trace = float(np.trace(m))
        if trace > 0.0:
            s = math.sqrt(trace + 1.0) * 2.0
            w = 0.25 * s
            x = (m[2, 1] - m[1, 2]) / s
            y = (m[0, 2] - m[2, 0]) / s
            z = (m[1, 0] - m[0, 1]) / s
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
        quaternion = np.array([x, y, z, w], dtype=np.float64)
        quaternion /= np.linalg.norm(quaternion)
        return tuple(float(value) for value in quaternion)

    @staticmethod
    def _quat_xyzw_to_mat(quaternion) -> np.ndarray:
        q = np.asarray(quaternion, dtype=np.float64).reshape(4)
        q /= np.linalg.norm(q)
        x, y, z, w = q
        return np.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )

    def _tool0_pose(self):
        flange_rotation = self.data.xmat[self._flange_body_id].reshape(3, 3)
        flange_position = self.data.xpos[self._flange_body_id]
        tool_position = flange_position + flange_rotation @ self._TOOL0_OFFSET
        tool_rotation = flange_rotation @ self._TOOL0_ROT
        return tool_position.copy(), tool_rotation.copy()

    @staticmethod
    def _orientation_error(current, target) -> np.ndarray:
        return 0.5 * (
            np.cross(current[:, 0], target[:, 0])
            + np.cross(current[:, 1], target[:, 1])
            + np.cross(current[:, 2], target[:, 2])
        )

    def read_state(self) -> RobotState:
        position, rotation = self._tool0_pose()
        return RobotState(
            joints_rad=self._read_joints(),
            gripper_opening_m=self._gripper_opening_m,
            tcp_pose=TcpPose(
                tuple(float(value) for value in position),
                self._mat_to_quat_xyzw(rotation),
            ),
            connected=True,
            ready=True,
            metadata={"backend": "mujoco", "scene_path": self.scene_path},
        )

    def move_joints(self, joints_rad: Sequence[float]) -> bool:
        values = _finite(joints_rad, 6)
        if values is None:
            return False
        for value, (lower, upper) in zip(values, self.joint_limits_rad):
            if value < lower or value > upper:
                return False
        self._write_joints(values)
        return True

    def move_tcp(self, pose: TcpPose) -> bool:
        target_position = np.asarray(pose.position_m, dtype=np.float64)
        target_rotation = self._quat_xyzw_to_mat(pose.quaternion_xyzw)
        values = np.asarray(self._read_joints(), dtype=np.float64)
        ranges = np.asarray(self.joint_limits_rad, dtype=np.float64)
        lower, upper = ranges[:, 0], ranges[:, 1]

        for _ in range(120):
            self._write_joints(values)
            current_position, current_rotation = self._tool0_pose()
            position_error = target_position - current_position
            orientation_error = self._orientation_error(
                current_rotation,
                target_rotation,
            )
            if (
                np.linalg.norm(position_error) <= 0.002
                and np.linalg.norm(orientation_error) <= 0.035
            ):
                return True

            jac_position = np.zeros((3, self.model.nv), dtype=np.float64)
            jac_rotation = np.zeros((3, self.model.nv), dtype=np.float64)
            mujoco.mj_jac(
                self.model,
                self.data,
                jac_position,
                jac_rotation,
                current_position,
                self._flange_body_id,
            )
            jacobian = np.vstack(
                (
                    jac_position[:, self._dof_addrs],
                    jac_rotation[:, self._dof_addrs],
                )
            )
            error = np.concatenate((position_error, orientation_error))
            damping = 0.03
            lhs = jacobian @ jacobian.T + (damping * damping) * np.eye(6)
            delta = jacobian.T @ np.linalg.solve(lhs, error)
            values = np.clip(values + np.clip(delta, -0.15, 0.15), lower, upper)

        self._write_joints(values)
        position, rotation = self._tool0_pose()
        return bool(
            np.linalg.norm(target_position - position) <= 0.002
            and np.linalg.norm(self._orientation_error(rotation, target_rotation)) <= 0.035
        )

    def set_gripper(self, opening_m: float) -> bool:
        try:
            opening = float(opening_m)
        except (TypeError, ValueError):
            return False
        if (
            not math.isfinite(opening)
            or opening < 0.0
            or opening > self.max_gripper_opening_m
        ):
            return False

        half = opening / 2.0
        for joint_id in (self._left_gripper_id, self._right_gripper_id):
            if joint_id >= 0:
                address = int(self.model.jnt_qposadr[joint_id])
                self.data.qpos[address] = half
        self._gripper_opening_m = opening
        mujoco.mj_forward(self.model, self.data)
        return True

    def close(self) -> None:
        pass


__all__ = ["DEFAULT_SCENE", "MujocoRobotDriver"]

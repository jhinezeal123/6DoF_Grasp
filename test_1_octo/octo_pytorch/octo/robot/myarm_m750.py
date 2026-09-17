from __future__ import annotations

import time
from typing import Dict, Optional

import numpy as np

from octo.robot.config import RobotConfig
from octo.robot.interfaces import RobotInterface
from octo.robot.transform_integrator import BridgeStyleActionIntegrator, TransformStepResult
from octo.robot.utils import (
    absolute_pose_to_relative_neutral_pose,
    clamp_pose_to_workspace,
    get_neutral_orientation_rpy,
    pose_error_metrics,
    pose_m_rad_to_mm_deg,
    pose_mm_deg_to_m_rad,
    select_rpy_near_reference,
    wrap_to_pi,
)


class MyArmM750Robot(RobotInterface):
    """MyArm M750 wrapper with Bridge-like Cartesian target propagation.

    Public interface remains SI units (meters + radians), while the hardware firmware
    is still driven through the vendor Cartesian API `write_coords(...)`.
    """

    def __init__(self, config: RobotConfig):
        self.config = config
        self._arm = None
        self._gripper_open = 1.0 if config.open_gripper_on_reset else 0.0
        self._neutral_orientation_rpy = get_neutral_orientation_rpy(
            config.neutral_orientation_rpy_rad,
            config.home_pose_m_rad,
        )
        self._tcp_pose_m_rad = clamp_pose_to_workspace(
            np.asarray(self.config.home_pose_m_rad, dtype=np.float64),
            self.config.workspace,
        )
        self._integrator = BridgeStyleActionIntegrator(config)
        self._step_counter = 0
        self._last_hardware_pose_m_rad: Optional[np.ndarray] = None
        self._last_resync_applied = False
        self._last_debug: Dict[str, object] = {}

    def connect(self) -> None:
        try:
            from pymycobot import MyArmMControl
        except Exception as e:
            raise ImportError(
                "pymycobot is required for MyArm M750 control. Install the vendor package on the robot machine."
            ) from e

        if self.config.baudrate is None:
            self._arm = MyArmMControl(self.config.port)
        else:
            self._arm = MyArmMControl(self.config.port, self.config.baudrate)
        print(f"Connected successfully to MyArm M750 on port {self.config.port}")

    def configure_for_policy(self) -> None:
        self._require_arm()
        if self.config.power_on:
            try:
                self._arm.power_on()
                time.sleep(0.2)
            except Exception:
                pass
        try:
            self._arm.set_fresh_mode(int(self.config.fresh_mode))
        except Exception:
            pass
        try:
            self._arm.set_world_reference([0, 0, 0, 0, 0, 0])
        except Exception:
            pass
        try:
            self._arm.set_tool_reference([0, 0, 0, 0, 0, 0])
        except Exception:
            pass
        try:
            self._arm.set_reference_frame(int(self.config.reference_frame))
        except Exception:
            pass
        try:
            self._arm.set_movement_type(int(self.config.movement_type))
        except Exception:
            pass
        try:
            self._arm.set_end_type(int(self.config.end_type))
        except Exception:
            pass
        try:
            self._arm.set_gripper_enabled()
        except Exception:
            pass

    def disconnect(self) -> None:
        self._arm = None

    def reset_to_home(self) -> None:
        self._require_arm()
        home_pose = clamp_pose_to_workspace(
            np.asarray(self.config.home_pose_m_rad, dtype=np.float64),
            self.config.workspace,
        )
        self._write_absolute_pose(home_pose, sync_integrator=True)
        self._step_counter = 0
        self._last_resync_applied = False
        if self.config.open_gripper_on_reset:
            self._command_gripper(1.0)
        time.sleep(float(self.config.reset_sleep_s))
        if bool(getattr(self.config, "reset_readback_sync", True)):
            measured = self._read_hardware_pose()
            if measured is not None:
                self._tcp_pose_m_rad = measured
                self._integrator.reset(measured)
                self._last_hardware_pose_m_rad = measured.copy()

    def get_tcp_pose_m_rad(self) -> np.ndarray:
        source = getattr(self.config, "tcp_pose_source", "internal")
        if source == "hardware":
            pose = self._read_hardware_pose()
            if pose is None:
                return clamp_pose_to_workspace(
                    np.asarray(self._tcp_pose_m_rad, dtype=np.float64),
                    self.config.workspace,
                )
            self._tcp_pose_m_rad = pose
            return pose
        if source == "internal":
            return clamp_pose_to_workspace(
                np.asarray(self._tcp_pose_m_rad, dtype=np.float64),
                self.config.workspace,
            )
        raise ValueError(f"Unsupported tcp_pose_source: {source!r}")

    def _get_proprio_pose(self, pose_m_rad: np.ndarray) -> np.ndarray:
        pose = np.asarray(pose_m_rad, dtype=np.float64).reshape(6)
        if self.config.proprio_orientation_mode == "absolute":
            return pose
        return absolute_pose_to_relative_neutral_pose(pose, self._neutral_orientation_rpy)

    def get_proprio(self) -> np.ndarray:
        pose = self.get_tcp_pose_m_rad()
        proprio_pose = self._get_proprio_pose(pose)
        return np.concatenate([proprio_pose, [self.get_gripper_open_fraction()]], axis=0)

    def get_gripper_open_fraction(self) -> float:
        self._require_arm()
        try:
            value = self._arm.get_gripper_value()
            if value is not None:
                self._gripper_open = float(np.clip(float(value) / 100.0, 0.0, 1.0))
        except Exception:
            pass
        return float(self._gripper_open)

    def _get_measured_pose_for_control(self) -> Optional[np.ndarray]:
        if bool(getattr(self.config, "hardware_measured_enabled", True)) or self.config.tcp_pose_source == "hardware":
            return self._read_hardware_pose()
        return None

    def apply_action(self, delta_pose_m_rad: np.ndarray, gripper_open_fraction: float) -> np.ndarray:
        self._require_arm()
        measured_before = self._get_measured_pose_for_control()
        current_pose = self.get_tcp_pose_m_rad()
        self._last_resync_applied = False

        step_result: TransformStepResult = self._integrator.step(
            np.asarray(delta_pose_m_rad, dtype=np.float64),
            measured_eef_pose_m_rad=measured_before,
        )
        target_pose = step_result.target_pose_m_rad
        clipped_delta = step_result.clipped_delta_pose_m_rad

        self._write_absolute_pose(target_pose, sync_integrator=False)
        self._command_gripper(gripper_open_fraction)
        self._step_counter += 1
        resync_reason = self._maybe_resync_from_hardware()

        measured_after = self._last_hardware_pose_m_rad.copy() if self._last_hardware_pose_m_rad is not None else current_pose.copy()
        if self.config.tcp_pose_source == "hardware" or bool(getattr(self.config, "hardware_measured_enabled", True)):
            maybe_after = self._read_hardware_pose()
            if maybe_after is not None:
                measured_after = maybe_after
                self._tcp_pose_m_rad = measured_after.copy()

        err = pose_error_metrics(self._integrator.get_target_pose(), measured_after)
        self._last_debug = {
            "step_counter": int(self._step_counter),
            "integration_mode": step_result.integration_mode,
            "pivot_source": step_result.pivot_source,
            "compose_base": step_result.compose_base_source,
            "delta_pose_m_rad": np.asarray(delta_pose_m_rad, dtype=np.float64).copy(),
            "clipped_delta_pose_m_rad": clipped_delta.copy(),
            "target_pose_before_step_m_rad": step_result.target_pose_before_step_m_rad.copy(),
            "target_pose_before_step_rel_m_rad": self._get_proprio_pose(step_result.target_pose_before_step_m_rad),
            "target_transform_before_step": step_result.target_transform_before_step.copy(),
            "target_pose_m_rad": self._integrator.get_target_pose().copy(),
            "target_pose_after_step_m_rad": self._integrator.get_target_pose().copy(),
            "target_pose_rel_m_rad": self._get_proprio_pose(self._integrator.get_target_pose()),
            "target_pose_after_step_rel_m_rad": self._get_proprio_pose(self._integrator.get_target_pose()),
            "target_transform": self._integrator.get_target_transform().copy(),
            "target_transform_after_step": self._integrator.get_target_transform().copy(),
            "compose_base_transform": step_result.compose_base_transform.copy(),
            "measured_pose_before_step_m_rad": None if measured_before is None else measured_before.copy(),
            "measured_pose_before_step_rel_m_rad": None if measured_before is None else self._get_proprio_pose(measured_before),
            "measured_pose_m_rad": measured_after.copy(),
            "measured_pose_after_step_m_rad": measured_after.copy(),
            "measured_pose_rel_m_rad": self._get_proprio_pose(measured_after),
            "measured_pose_after_step_rel_m_rad": self._get_proprio_pose(measured_after),
            "hardware_pose_m_rad": None if self._last_hardware_pose_m_rad is None else self._last_hardware_pose_m_rad.copy(),
            "neutral_orientation_rpy_rad": self._neutral_orientation_rpy.copy(),
            "pivot_position_m": step_result.pivot_position_m.copy(),
            "gripper_open": float(self._gripper_open),
            "translation_error_m": float(err["translation_m"]),
            "rotation_error_rad": float(err["rotation_rad"]),
            "translation_error_norm_m": float(err["translation_m"]),
            "rotation_error_norm_rad": float(err["rotation_rad"]),
            "resync_applied": bool(self._last_resync_applied),
            "resync_reason": resync_reason,
        }

        if self.config.debug_print_every_n_steps > 0 and self._step_counter % int(self.config.debug_print_every_n_steps) == 0:
            print(
                "[robot] "
                f"step={self._step_counter:04d} mode={step_result.integration_mode} "
                f"pivot={step_result.pivot_source} compose={step_result.compose_base_source} "
                f"target_xyz={np.round(self._integrator.get_target_pose()[:3], 4)} "
                f"measured_xyz={np.round(measured_after[:3], 4)} "
                f"trans_err={err['translation_m']:.4f}m rot_err={err['rotation_rad']:.4f}rad "
                f"resync={bool(self._last_resync_applied)}"
            )
        return target_pose.copy()

    def move_tcp_absolute(self, pose_m_rad: np.ndarray, gripper_open_fraction: float) -> np.ndarray:
        """Command an absolute TCP pose (meters + radians) and gripper fraction.

        This keeps the internal integrator synchronized to the commanded pose so
        subsequent delta-based steps remain well-defined.
        """
        self._require_arm()
        target_pose = np.asarray(pose_m_rad, dtype=np.float64).reshape(6)
        self._write_absolute_pose(target_pose, sync_integrator=True)
        self._command_gripper(gripper_open_fraction)
        self._step_counter += 1
        # Optional periodic resync is handled by apply_action(); for absolute commands
        # we keep behavior simple and rely on tcp_pose_source/hardware settings.
        return self.get_tcp_pose_m_rad().copy()

    def _write_absolute_pose(self, pose_m_rad: np.ndarray, *, sync_integrator: bool) -> None:
        pose_m_rad = clamp_pose_to_workspace(
            np.asarray(pose_m_rad, dtype=np.float64),
            self.config.workspace,
        )
        self._tcp_pose_m_rad = pose_m_rad
        if sync_integrator:
            self._integrator.reset(pose_m_rad)
        pose_mm_deg = pose_m_rad_to_mm_deg(pose_m_rad)
        coords = [float(x) for x in pose_mm_deg.tolist()]
        self._arm.write_coords(coords, int(self.config.move_speed), int(self.config.coord_mode))
        should_wait = bool(self.config.wait_for_motion) or bool(getattr(self.config, "blocking_move", False))
        if should_wait:
            self._wait_until_idle_or_timeout()
        elif float(getattr(self.config, "post_command_sleep_s", 0.0)) > 0:
            time.sleep(float(self.config.post_command_sleep_s))

    def _command_gripper(self, gripper_open_fraction: float) -> None:
        target = float(np.clip(gripper_open_fraction, 0.0, 1.0))
        if self.config.gripper_use_value_api:
            value = int(
                round(
                    self.config.gripper_close_value
                    + target * (self.config.gripper_open_value - self.config.gripper_close_value)
                )
            )
            self._arm.set_gripper_value(value, int(self.config.gripper_speed))
        else:
            flag = 0 if target >= self.config.gripper_threshold else 1
            self._arm.set_gripper_state(int(flag), int(self.config.gripper_speed))
        self._gripper_open = target

    def _wait_until_idle_or_timeout(self) -> None:
        deadline = time.time() + float(self.config.wait_timeout_s)
        while time.time() < deadline:
            try:
                moving = self._arm.is_moving()
            except Exception:
                break
            if moving in (0, False):
                break
            time.sleep(0.02)

    def _read_hardware_pose(self):
        self._require_arm()
        settle_s = float(getattr(self.config, "hardware_read_settle_s", 0.0))
        if settle_s > 0:
            time.sleep(settle_s)
        try:
            coords = self._arm.get_coords()
        except Exception:
            return None
        if coords is None or len(coords) < 6:
            return None
        pose = pose_mm_deg_to_m_rad(coords[:6])
        pose = np.asarray(pose, dtype=np.float64).reshape(6)
        pose[:3] = np.clip(
            pose[:3],
            np.asarray(self.config.workspace.translation_min_m, dtype=np.float64),
            np.asarray(self.config.workspace.translation_max_m, dtype=np.float64),
        )
        pose[3:6] = wrap_to_pi(pose[3:6])
        try:
            pose[3:6] = select_rpy_near_reference(pose[3:6], self._integrator.get_target_pose()[3:6])
        except Exception:
            pass
        self._last_hardware_pose_m_rad = pose.copy()
        return pose

    def _maybe_resync_from_hardware(self) -> str:
        if not bool(getattr(self.config, "periodic_resync_enabled", False)):
            return "disabled"
        every_n = int(getattr(self.config, "periodic_resync_every_n_steps", 0))
        if every_n <= 0:
            return "disabled"
        if self._step_counter % every_n != 0:
            return "not_due"
        measured = self._read_hardware_pose()
        if measured is None:
            return "no_measurement"
        target = self._integrator.get_target_pose()
        err = pose_error_metrics(target, measured)
        if (
            err["translation_m"] >= float(getattr(self.config, "periodic_resync_position_threshold_m", 0.03))
            or err["rotation_rad"] >= float(getattr(self.config, "periodic_resync_rotation_threshold_rad", 0.35))
        ):
            print(
                "Resyncing internal target from hardware pose: "
                f"translation_err={err['translation_m']:.4f}m, rotation_err={err['rotation_rad']:.4f}rad"
            )
            self._tcp_pose_m_rad = measured
            self._integrator.reset(measured)
            self._last_resync_applied = True
            return "threshold_exceeded"
        return "within_threshold"

    def get_debug_snapshot(self) -> Dict[str, object]:
        snap = dict(self._last_debug)
        if not snap:
            measured = self.get_tcp_pose_m_rad()
            target = self._integrator.get_target_pose()
            err = pose_error_metrics(target, measured)
            snap = {
                "step_counter": int(self._step_counter),
                "integration_mode": getattr(self.config, "se3_integration_mode", "bridge_exact"),
                "pivot_source": getattr(self.config, "pivot_source", "hardware_measured"),
                "compose_base": getattr(self.config, "compose_base", "previous_target"),
                "target_pose_before_step_m_rad": target.copy(),
                "target_pose_before_step_rel_m_rad": self._get_proprio_pose(target),
                "target_transform_before_step": self._integrator.get_target_transform().copy(),
                "target_pose_m_rad": target.copy(),
                "target_pose_after_step_m_rad": target.copy(),
                "target_pose_rel_m_rad": self._get_proprio_pose(target),
                "target_pose_after_step_rel_m_rad": self._get_proprio_pose(target),
                "target_transform": self._integrator.get_target_transform().copy(),
                "target_transform_after_step": self._integrator.get_target_transform().copy(),
                "compose_base_transform": self._integrator.get_target_transform().copy(),
                "measured_pose_before_step_m_rad": measured.copy(),
                "measured_pose_before_step_rel_m_rad": self._get_proprio_pose(measured),
                "measured_pose_m_rad": measured.copy(),
                "measured_pose_after_step_m_rad": measured.copy(),
                "measured_pose_rel_m_rad": self._get_proprio_pose(measured),
                "measured_pose_after_step_rel_m_rad": self._get_proprio_pose(measured),
                "hardware_pose_m_rad": None if self._last_hardware_pose_m_rad is None else self._last_hardware_pose_m_rad.copy(),
                "neutral_orientation_rpy_rad": self._neutral_orientation_rpy.copy(),
                "pivot_position_m": target[:3].copy(),
                "translation_error_m": float(err["translation_m"]),
                "rotation_error_rad": float(err["rotation_rad"]),
                "translation_error_norm_m": float(err["translation_m"]),
                "rotation_error_norm_rad": float(err["rotation_rad"]),
                "resync_applied": False,
                "resync_reason": "snapshot_only",
                "gripper_open": float(self._gripper_open),
            }
        return snap

    def _require_arm(self):
        if self._arm is None:
            raise RuntimeError("MyArm M750 is not connected")

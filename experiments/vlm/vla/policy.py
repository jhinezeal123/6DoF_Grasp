"""Policy: hoi model hai diem roi chay servo anh, tra ve tu the TCP tuyet doi.

Module nay cam vao khe ``policy`` cua Pipeline. No khong tu doc camera, khong tu
gui lenh - chi nhan Observation va tra Action, dung nhu interface Policy doi.

Moi buoc can HAI lan goi model:
  1. diem dich (vung vat can gap) - chi goi MOT lan cho ca episode, vi vat dung yen
  2. diem ngon kep              - goi lai MOI buoc, vi tay vua dich

Buoc 1 ton ~15 s o lan goi dau. Vi vay Source phai dat max_age_s lon hon ca hai
lan goi cong lai, neu khong runner se nem TimeoutError o cho kiem tra do tuoi
sau predict() - xem runner.py dong 180.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from m750.pipeline.interfaces import Policy
from m750.pipeline.types import Action, ActionSpec, Episode, Observation

from .physbrain_model import PhysBrainClient
from .prompts import GRIPPER_QUESTION, point_question
from .servo import ImageServo


class GraspPointPolicy(Policy):
    """Dieu khien Cartesian: chi x, y, z; huong co dinh vuong goc mat ban."""

    def __init__(
        self,
        robot: Any,
        model: PhysBrainClient,
        servo: ImageServo,
        spec: ActionSpec,
        *,
        camera_name: str = "primary",
        z_floor_m: float = 0.02,
        max_descend_m: float = 0.28,
        max_target_drift_m: float = 0.30,
        log: Any = print,
    ) -> None:
        self.robot = robot
        self.model = model
        self.servo = servo
        self.spec = spec
        self.camera_name = camera_name
        self.z_floor_m = float(z_floor_m)
        self.max_descend_m = float(max_descend_m)
        self.max_target_drift_m = float(max_target_drift_m)
        self.log = log

        self._target_px: tuple[float, float] | None = None
        self._orientation = np.zeros(4)
        self._start_pos = np.zeros(3)
        self._descended_m = 0.0
        # Trang thai cua lan quan sat truoc, de hoc J tu chinh buoc vua roi.
        self._last_tcp: np.ndarray | None = None
        self._last_gripper_px: tuple[float, float] | None = None

    # ------------------------------------------------------------------ vong doi
    def reset(self, episode: Episode) -> None:
        """Bat dau episode moi: quen diem dich cu, chup lai huong va goc."""
        self._target_px = None
        self._descended_m = 0.0
        self._last_tcp = None
        self._last_gripper_px = None
        self._orientation = np.array(self.robot.tcp_quat, dtype=np.float64)
        self._start_pos = np.array(self.robot.tcp_pos, dtype=np.float64)
        if not np.all(np.isfinite(self._orientation)):
            raise RuntimeError("tcp_quat khong hop le khi bat dau episode")

    # ------------------------------------------------------------------ suy luan
    def predict(self, observation: Observation, instruction: str) -> Action:
        frame = observation.images[self.camera_name]

        if self._target_px is None:
            self.log("  [policy] hoi model diem dich cho: %r" % instruction)
            self._target_px = self.model.point(frame, point_question(instruction))
            if self._target_px is None:
                raise RuntimeError(
                    "model khong chi duoc vat nao cho lenh %r (tra loi: %r)"
                    % (instruction, self.model.last_text[:160])
                )
            self.log("  [policy] diem dich = (%.1f, %.1f) px sau %.1fs"
                     % (self._target_px[0], self._target_px[1],
                        self.model.last_latency_s or 0.0))

        gripper_px = self.model.point(frame, GRIPPER_QUESTION)
        if gripper_px is None:
            raise RuntimeError(
                "model khong chi duoc ngon kep (tra loi: %r)"
                % self.model.last_text[:160]
            )

        current = np.array(self.robot.tcp_pos, dtype=np.float64)
        if not np.all(np.isfinite(current)):
            raise RuntimeError("tcp_pos khong hop le khi tinh lenh")

        # Hoc J tu chinh buoc vua roi. Day la nguyen lieu mien phi: tay da di roi,
        # anh da doi roi, chi viec ghi lai. Dung TCP DO DUOC chu khong phai toa do
        # da ra lenh - tay bam theo lech 13-20 mm nen dung luong da ra lenh se
        # nhet sai so bam do vao J.
        if self._last_tcp is not None and self._last_gripper_px is not None:
            predicted = self.servo.observe(
                current - self._last_tcp,
                np.array(gripper_px, dtype=np.float64) - np.array(self._last_gripper_px,
                                                                  dtype=np.float64),
            )
            if predicted is not None:
                self.log("     [hoc J] %d cap, sai so du doan truoc do %.1f px, cond %.1f"
                         % (self.servo.estimator.updates, predicted, self.servo.cond))
        self._last_tcp = current.copy()
        self._last_gripper_px = (float(gripper_px[0]), float(gripper_px[1]))

        error = np.array(self._target_px, dtype=np.float64) - np.array(gripper_px,
                                                                       dtype=np.float64)
        self.servo.check()
        step = self.servo.step(error)

        target = current + step.delta_m
        target[2] = max(target[2], self.z_floor_m)

        # Chan ha qua sau: gripper xuong mai ma khong kep duoc la hong vat hoac
        # hong ban. Tinh theo hieu so voi tu the dau chu khong theo tong so buoc.
        if step.descend:
            self._descended_m = self._start_pos[2] - target[2]
            if self._descended_m > self.max_descend_m:
                raise RuntimeError(
                    "da ha %.0f mm (gioi han %.0f mm) ma chua can hang duoc"
                    % (self._descended_m * 1000, self.max_descend_m * 1000)
                )

        # Chan troi dat: diem dich nhay qua xa nghia la model doi y hoac doc sai.
        drift = float(np.linalg.norm(target[:2] - self._start_pos[:2]))
        if drift > self.max_target_drift_m:
            raise RuntimeError(
                "dich xa goc %.0f mm (gioi han %.0f mm); nghi model doc sai"
                % (drift * 1000, self.max_target_drift_m * 1000)
            )

        self.log("  [policy] ngon kep = (%.1f, %.1f)  sai so = %.1f px%s%s -> buoc (%.1f, %.1f, %.1f) mm"
                 % (gripper_px[0], gripper_px[1], step.error_px,
                    " [CAN HANG]" if step.aligned else "",
                    " [CAT]" if step.saturated else "",
                    step.delta_m[0] * 1000, step.delta_m[1] * 1000, step.delta_m[2] * 1000))

        values = (float(target[0]), float(target[1]), float(target[2]),
                  float(self._orientation[0]), float(self._orientation[1]),
                  float(self._orientation[2]), float(self._orientation[3]))
        return Action(values=values, spec=self.spec)

    # ------------------------------------------------------------------ tien ich
    @property
    def target_px(self) -> tuple[float, float] | None:
        return self._target_px

    @property
    def descended_m(self) -> float:
        return self._descended_m


__all__ = ["GraspPointPolicy"]

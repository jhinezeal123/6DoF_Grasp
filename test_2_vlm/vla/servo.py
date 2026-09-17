"""Servo anh: doi sai so pixel thanh buoc dich TCP, qua ma tran Jacobian J.

Toan hoc thuan, khong biet gi ve ROS hay robot. J do tu `scripts/calibrate_jacobian.py`.

    e_px   = diem_dich - diem_ngon_kep        (pixel)
    [dx,dy] = J^-1 @ e_px                     (met, he base_link)
    dz     = -z_step khi da can hang, nguoc lai 0

Huong luon giu nguyen (vung goc voi mat ban) - test 2 chi dieu khien x, y, z.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class ServoError(RuntimeError):
    """J suy bien hoac cau hinh servo vo ly."""


@dataclass(frozen=True)
class ServoStep:
    """Mot buoc servo da tinh xong, truoc khi cong vao tu the hien tai."""

    delta_m: np.ndarray      # [dx, dy, dz]
    error_px: float
    aligned: bool            # sai so nam trong nguong can hang
    saturated: bool          # buoc bi cat vi vuot max_step
    descend: bool            # buoc nay co ha xuong khong


class ImageServo:
    def __init__(self, jacobian, *, max_step_xy_m: float = 0.02,
                 z_step_m: float = 0.005, align_px: float = 25.0,
                 min_abs_det: float = 1.0) -> None:
        self.J = np.asarray(jacobian, dtype=np.float64)
        if self.J.shape != (2, 2):
            raise ServoError("J phai la ma tran 2x2, nhan duoc %s" % (self.J.shape,))
        det = float(np.linalg.det(self.J))
        # J gan suy bien thi J^-1 khuech dai nhieu anh thanh buoc chay loan.
        # Nguong 1.0 pixel/met rat thap so voi J do duoc (~-6.9e5), chi de bat
        # truong hop J bi dien sai hoac bang 0.
        if not np.isfinite(det) or abs(det) < float(min_abs_det):
            raise ServoError("det(J) = %g qua nho; J suy bien, khong servo duoc" % det)
        if max_step_xy_m <= 0 or z_step_m <= 0 or align_px <= 0:
            raise ServoError("max_step_xy_m, z_step_m, align_px phai duong")

        self.J_inv = np.linalg.inv(self.J)
        self.cond = float(np.linalg.cond(self.J))
        self.max_step_xy_m = float(max_step_xy_m)
        self.z_step_m = float(z_step_m)
        self.align_px = float(align_px)
        self.det = det

    def step(self, error_px) -> ServoStep:
        """Tinh buoc tiep theo tu vector sai so anh (pixel)."""
        error = np.asarray(error_px, dtype=np.float64).reshape(2)
        if not np.all(np.isfinite(error)):
            raise ServoError("sai so anh chua NaN/inf")

        magnitude = float(np.linalg.norm(error))
        aligned = magnitude <= self.align_px

        delta_xy = self.J_inv @ error
        length = float(np.linalg.norm(delta_xy))
        saturated = length > self.max_step_xy_m
        if saturated:
            delta_xy = delta_xy * (self.max_step_xy_m / length)

        # Chi ha xuong khi da can hang ngang: ha som la truot khoi vat.
        descend = aligned
        dz = -self.z_step_m if descend else 0.0

        return ServoStep(
            delta_m=np.array([delta_xy[0], delta_xy[1], dz], dtype=np.float64),
            error_px=magnitude,
            aligned=aligned,
            saturated=saturated,
            descend=descend,
        )


__all__ = ["ImageServo", "ServoStep", "ServoError"]

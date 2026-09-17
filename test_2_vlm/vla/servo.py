"""Servo anh: doi sai so pixel thanh buoc dich TCP, qua ma tran Jacobian J.

Toan hoc thuan, khong biet gi ve ROS hay robot. J do tu `scripts/calibrate_jacobian.py`.

    e_px      = diem_dich - diem_ngon_kep          (pixel, 2 chieu)
    [dx,dy,dz] = pinv(J) @ e_px                    (met, he base_link)

J la 2x3 (x, y, z) chu KHONG phai 2x2. Hai lan do cho hai ket qua khac han nhau,
va lan dau sai vi ly do vat ly chu khong phai toan hoc:

  - Lan 1, kẹp tì xuống mặt bàn: cot z do duoc DUNG BANG 0 - tay khong the ha
    xuong nen anh khong nhuc nhich. Moi ket luan rut ra tu J do deu vo nghia.
  - Lan 2, nang tay len: cond = 2.5, ba truc deu dung duoc (x 0.64, y 2.05,
    z 0.90 px/mm).

Bai hoc: truoc khi tin vao J, phai kiem tra cot nao bang 0 hoac gan 0 - do la
dau hieu co cai gi do dang CHAN chuyen dong, khong phai J "dep".
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
                 max_step_z_m: float = 0.02, align_px: float = 25.0,
                 min_singular: float = 1.0) -> None:
        self.J = np.atleast_2d(np.asarray(jacobian, dtype=np.float64))
        if self.J.shape[0] != 2:
            raise ServoError("J phai co 2 hang (toa do anh), nhan duoc %s" % (self.J.shape,))
        if self.J.shape[1] not in (2, 3):
            raise ServoError("J phai co 2 hoac 3 cot (x, y[, z]), nhan duoc %s"
                             % (self.J.shape,))
        if max_step_xy_m <= 0 or max_step_z_m <= 0 or align_px <= 0:
            raise ServoError("max_step_xy_m, max_step_z_m, align_px phai duong")

        # Gia nghich dao: dung duoc cho ca J 2x2 lan 2x3, va khong no ra khi J
        # gan suy bien theo mot huong nao do - no chi bo huong do di.
        self.J_pinv = np.linalg.pinv(self.J)
        self.cond = float(np.linalg.cond(self.J))
        self.singular = np.linalg.svd(self.J, compute_uv=False)

        # Nguong nay bat J bi dien sai hoac bang 0, khong phai de danh gia chat
        # luong. J do duoc co gia tri suy bien ~900-2000 nen nguong 1.0 rat thap.
        if not np.all(np.isfinite(self.singular)) or self.singular[0] < float(min_singular):
            raise ServoError("J suy bien (singular value lon nhat = %g); khong servo duoc"
                             % self.singular[0])

        self.max_step_xy_m = float(max_step_xy_m)
        self.max_step_z_m = float(max_step_z_m)
        self.align_px = float(align_px)
        self.has_z = self.J.shape[1] == 3

        # Do nhay tung truc, de bao cao chu khong dung trong tinh toan.
        self.px_per_m = np.linalg.norm(self.J, axis=0)

    def step(self, error_px) -> ServoStep:
        """Tinh buoc tiep theo tu vector sai so anh (pixel)."""
        error = np.asarray(error_px, dtype=np.float64).reshape(2)
        if not np.all(np.isfinite(error)):
            raise ServoError("sai so anh chua NaN/inf")

        magnitude = float(np.linalg.norm(error))
        aligned = magnitude <= self.align_px

        delta = self.J_pinv @ error
        if delta.shape[0] == 2:
            delta = np.array([delta[0], delta[1], 0.0])
        delta = delta.astype(np.float64, copy=True)

        # KHONG chan ha xuong theo "da can hang chua". Da thu va no tu khoa chinh
        # minh: voi gripper cao hon vat ~220 mm, sai so doc trong anh (111 px)
        # chinh LA do chieu cao, nen buoc ha xuong chiem 142/150 mm cua hieu chinh.
        # Chi cho ha khi da can hang ngang thi khong bao gio ha duoc, vi can hang
        # ngang khong bao gio xong. De Jacobian tu quyet dinh ty le ba truc.

        # Cat rieng phan ngang va phan dung. Cat chung theo chuan 3 chieu se khien
        # mot buoc ngang lon an het han muc va khong bao gio ha xuong duoc.
        horizontal = float(np.linalg.norm(delta[:2]))
        saturated = horizontal > self.max_step_xy_m
        if saturated:
            delta[:2] *= self.max_step_xy_m / horizontal
        delta[2] = float(np.clip(delta[2], -self.max_step_z_m, self.max_step_z_m))

        return ServoStep(
            delta_m=delta,
            error_px=magnitude,
            aligned=aligned,
            saturated=saturated,
            descend=bool(delta[2] < 0.0),
        )


__all__ = ["ImageServo", "ServoStep", "ServoError"]

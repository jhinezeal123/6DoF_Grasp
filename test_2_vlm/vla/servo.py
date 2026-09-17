"""Servo anh: doi sai so pixel thanh buoc dich TCP, qua ma tran Jacobian J.

Toan hoc thuan, khong biet gi ve ROS hay robot. J do tu `scripts/calibrate_jacobian.py`.

    e_px      = diem_dich - diem_ngon_kep          (pixel, 2 chieu)
    [dx,dy,dz] = pinv(J) @ e_px                    (met, he base_link)
    dz        = 0 khi chua can hang ngang

J la 2x3 (x, y, z) chu KHONG phai 2x2. Ly do do that: camera nhin cheo tu
truoc-trai nen do nhay anh cua ba truc chenh nhau 5.3 lan (x 1.92 px/mm,
z 1.47 px/mm, y 0.36 px/mm). Voi J 2x2 chi gom (x, y), mot sai so 104 px doi
346 mm dich chuyen - khong bao gio hoi tu. Gia nghich dao trai deu hieu chinh
sang ca ba truc va tu dong dung it truc y nhat.

Khi J vuong (2x2, tu file cu), pinv suy bien ve nghich dao thuong nhu cu.
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
                 min_singular: float = 1.0) -> None:
        self.J = np.atleast_2d(np.asarray(jacobian, dtype=np.float64))
        if self.J.shape[0] != 2:
            raise ServoError("J phai co 2 hang (toa do anh), nhan duoc %s" % (self.J.shape,))
        if self.J.shape[1] not in (2, 3):
            raise ServoError("J phai co 2 hoac 3 cot (x, y[, z]), nhan duoc %s"
                             % (self.J.shape,))
        if max_step_xy_m <= 0 or z_step_m <= 0 or align_px <= 0:
            raise ServoError("max_step_xy_m, z_step_m, align_px phai duong")

        # Gia nghich dao: dung duoc cho ca J 2x2 lan 2x3, va khong no ra khi J
        # gan suy bien theo mot huong nao do - no chi bo huong do di.
        self.J_pinv = np.linalg.pinv(self.J)
        self.cond = float(np.linalg.cond(self.J))
        self.singular = np.linalg.svd(self.J, compute_uv=False)

        # Nguong nay bat J bi dien sai hoac bang 0, khong phai de danh gia chat
        # luong. J do duoc co gia tri suy bien ~1500-2000 nen nguong 1.0 rat thap.
        if not np.all(np.isfinite(self.singular)) or self.singular[0] < float(min_singular):
            raise ServoError("J suy bien (singular value lon nhat = %g); khong servo duoc"
                             % self.singular[0])

        self.max_step_xy_m = float(max_step_xy_m)
        self.z_step_m = float(z_step_m)
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

        # Ha xuong CHI khi da can hang ngang: ha som la truot khoi vat.
        if not aligned:
            delta[2] = 0.0

        # Cat rieng phan ngang va phan dung. Cat chung theo chuan 3 chieu se khien
        # mot buoc ngang lon an het han muc va khong bao gio ha xuong duoc.
        horizontal = float(np.linalg.norm(delta[:2]))
        saturated = horizontal > self.max_step_xy_m
        if saturated:
            delta[:2] *= self.max_step_xy_m / horizontal
        delta[2] = float(np.clip(delta[2], -self.z_step_m, self.z_step_m))

        return ServoStep(
            delta_m=delta,
            error_px=magnitude,
            aligned=aligned,
            saturated=saturated,
            descend=bool(aligned and delta[2] < 0.0),
        )


__all__ = ["ImageServo", "ServoStep", "ServoError"]

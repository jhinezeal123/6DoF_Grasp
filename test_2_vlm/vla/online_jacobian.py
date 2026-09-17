"""Uoc luong J ngay trong luc servo - khong can buoc hieu chuan rieng.

Y tuong: moi buoc servo von di da tao san mot cap (tay dich bao nhieu, anh dich
bao nhieu). Do la nguyen lieu de hoc J. Khong ton them mot lan goi model nao.

    Broyden:  J <- J + (dp - J@dm) @ dm^T / (dm^T @ dm)

Trong do dm la dich chuyen TCP DO DUOC (khong phai luong da ra lenh) va dp la
dich chuyen diem ngon kep trong anh.

Vi sao cach nay hon han buoc hieu chuan rieng da thu:

  - Hieu chuan rieng phai do tung truc mot, moi truc mot lan goi model (~15 s),
    va ket qua chi dung TAI tu the do. Da do 4 lan, hai lan lien tiep cho ra cot
    nguoc dau nhau, vi model luong tu hoa 5 don vi (3.6 px) con buoc 2 cm chi
    dich anh 10-40 px.
  - Hoc truc tuyen dung dung cac buoc ma servo SE di that, o dung vung lam viec,
    va tu sua khi sai. Buoc cang lon thi tin hieu cang ro.

Dung dm do duoc chu khong phai dm da ra lenh: tay bam theo lech 13-20 mm tren
lenh 60 mm, nen dung luong da ra lenh se nhet sai so bam do vao J.
"""

from __future__ import annotations

import numpy as np


class OnlineJacobian:
    """J 2xN cap nhat dan bang Broyden, khoi tao tu mot gia tri tho."""

    def __init__(self, initial, *, damping: float = 1e-6,
                 max_scale: float = 20.0) -> None:
        self.J = np.atleast_2d(np.asarray(initial, dtype=np.float64)).copy()
        if self.J.shape[0] != 2:
            raise ValueError("J phai co 2 hang")
        self.damping = float(damping)
        # Chan J phinh to: mot cap (dm, dp) sai lam cung co the day J di rat xa.
        self.max_scale = float(max_scale)
        self.initial_norm = float(np.linalg.norm(self.J))
        self.updates = 0
        self.last_prediction_error = None

    def update(self, delta_m, delta_px) -> float | None:
        """Hoc tu mot cap (dich chuyen TCP do duoc, dich chuyen anh).

        Tra ve sai so du doan TRUOC khi cap nhat, hoac None neu bo qua.
        """
        dm = np.asarray(delta_m, dtype=np.float64).reshape(-1)
        dp = np.asarray(delta_px, dtype=np.float64).reshape(2)
        if not (np.all(np.isfinite(dm)) and np.all(np.isfinite(dp))):
            return None
        # Buoc qua nho thi ti le tin hieu/nhieu qua thap de hoc duoc gi.
        if float(dm @ dm) < (0.004 ** 2):
            return None
        if float(np.linalg.norm(dp)) < 1.0:
            return None

        residual = dp - self.J @ dm
        self.last_prediction_error = float(np.linalg.norm(residual))
        self.J = self.J + np.outer(residual, dm) / float(dm @ dm + self.damping)
        self.updates += 1

        # Kep J trong khoang suy bien: neu phinh qua max_scale lan gia tri ban dau
        # thi tra ve gia tri ban dau thay vi de servo chay loan.
        norm = float(np.linalg.norm(self.J))
        if norm > self.initial_norm * self.max_scale or norm < self.initial_norm / self.max_scale:
            self.J = self.J * (self.initial_norm / max(norm, 1e-12))
        return self.last_prediction_error

    @property
    def condition(self) -> float:
        try:
            return float(np.linalg.cond(self.J))
        except np.linalg.LinAlgError:
            return float("inf")

    def sensitivities_px_per_mm(self) -> np.ndarray:
        return np.linalg.norm(self.J, axis=0) / 1000.0


def seed_jacobian(horizontal_fov_deg: float = 70.0, width_px: int = 1280,
                  depth_m: float = 0.7) -> np.ndarray:
    """J tho de khoi dong, suy tu dinh luat camera loi (pinhole).

    f = (rong/2) / tan(FOV/2). Voi vat cach camera `depth_m` va huong nhin gan
    vuong goc voi mat ban:

        du/dx = f/depth   (dich ngang anh)      dv/dz = -f/depth  (dich doc anh)

    FOV cua module SPCA2650 khong duoc cong bo - tra co so du lieu USB chi ra
    chipset Sunplus 1bcf:0c15, khong co thong so quang hoc. Day la GIA DINH,
    khong phai so do, va Broyden se sua lai sau vai buoc.

    Cot y co dinh mot luong nho chu khong bang 0, du chua biet dau that. Ly do:
    Broyden chi sua J theo huong `dm` ma no TUNG di qua. Cot y bang 0 thi pinv
    khong bao gio ra lenh dich theo y, nen truc y vinh vien khong duoc hoc - mo
    phong cho thay dung loi nay (J hoc duoc lech 115% so voi J that). Cho no mot
    gia tri nho de truc y duoc thu, roi hoc lai cho dung.
    """
    f = (width_px / 2.0) / np.tan(np.radians(horizontal_fov_deg) / 2.0)
    scale = f / max(depth_m, 1e-6)
    # Thu tu cot: x, y, z. Truc x ngang anh, truc z doc anh (huong len).
    return np.array([[scale, 0.3 * scale, 0.0],
                     [0.0, 0.0, -scale]], dtype=np.float64)


__all__ = ["OnlineJacobian", "seed_jacobian"]

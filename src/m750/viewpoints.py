"""viewpoints.py - Sinh 6 pose quanh mot tam, theo GraspNeRF muc IV-B.

Thuan toan hoc: khong doc robot, khong mo port -> test duoc. Truyen ket qua
truc tiep vao ArmController.move_gripper_to (cung quy uoc pose).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R


def ring_views(center, r: float = 150.0, theta: float = 30.0) -> list:
    """6 pose quanh tam: ban cau ban kinh r, goc cuc theta, 6 phuong vi chia deu 60 do,
    moi pose chuc thang vao tam.

    Args:
        center: [x, y, z] mm - TAM GAP (gripper_base). Camera khong nam dung tam
            gap thi tru offset cua no vao r.
        r: mm tu TAM toi TAM GAP.
        theta: goc cuc (do). Tran DO THAT tren tay nay voi tam x=449: theta=20 ->
            r<=225, theta=30 -> r<=175, theta=40 -> r<150. Paper dung r=500m
            KHONG toi duoc (khuyu phai ra ~715mm > ~580mm).
    """
    center = np.array(center, dtype=float)
    up, th = np.array([0.0, 0.0, 1.0]), np.radians(theta)
    out = []
    for k in range(6):
        ph = np.radians(60.0 * k)
        u = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])  # tam -> camera
        yv = up - u * (up @ u)  # tool0 +y gan voi phuong dung nhat; roll quanh truc nhin la tu do
        yv = yv / np.linalg.norm(yv)
        M = np.column_stack([np.cross(yv, u), yv, u])  # cot 3 = u = -huong nhin (truc +z cua tool0)
        out.append([round(float(v), 2) for v in center + r * u]
                   + [round(float(v), 2) for v in R.from_matrix(M).as_euler("xyz", degrees=True)])
    return out


__all__ = ["ring_views"]

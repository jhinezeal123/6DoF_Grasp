"""
script/ package - Cac module tien ich va ham bo tro cho he thong Robot VLA.

Luu y: IK va noi suy quy dao KHONG con o day. Chung do stack ROS 2 lo
(myarm_kinematics giai IK, myarm_motion_execution noi suy va chay quy dao).
Xem ros_bridge.py.

scene.py da xoa (0 caller runtime; FakeRobot tu trien khai reset rieng).
Chi con compressor - KHONG import mujoco o day nua de from script.compressor
import ... khong keo physics engine theo (sua loi coupling ngam).
"""

from .compressor import ImageCompressor

__all__ = [
    "ImageCompressor",
]

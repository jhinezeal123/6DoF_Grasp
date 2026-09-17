"""
robot/ package - Lop Robot dieu khien canh tay myArm M750.

Robot la mat na (facade) tren stack ROS 2. Khong con pymycobot, khong con bang
offset rieng: serial, quy doi goc va gioi han deu do myarm_robot_driver lo.
"""
from .robot import Robot

__all__ = ["Robot"]

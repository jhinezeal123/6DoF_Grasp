"""
robot/ package - Lop Robot dieu khien canh tay myArm M750.

Robot la mat na (facade) tren stack ROS 2: serial, quy doi goc va gioi han deu do
myarm_robot_driver lo. Lop nay khong goi pymycobot, nhung pymycobot van duoc phep
dung o tang driver va o pre-flight cua run_web.sh (xem robot/robot.py).
"""
from .robot import Robot

__all__ = ["Robot"]

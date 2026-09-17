"""
fake_robot/ package - FakeRobot: MuJoCo hien thi trang thai that tren ROS 2.

Khong con mode doc lap/dong nhat: chay backend nao la viec cua launch profile.
"""
from .fake_robot import FakeRobot, SCENE_XML

__all__ = ["FakeRobot", "SCENE_XML"]

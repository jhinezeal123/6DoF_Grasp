"""
user/ - Lop INTERFACE cho nguoi dung thong thuong: truc quan, de nhin, de phat
hien bug. Khong phai thu vien de nhung vao chuong trinh khac.

    user/web_control.py    Web UI Control (WebControlApp, start_web_server)
    user/fake_robot/       FakeRobot - MuJoCo hien thi trang thai ROS
    user/robot_model/      scene XML + meshes

Chay: python user/web_control.py   (can stack ROS 2 dang chay)
"""

__all__ = ["fake_robot", "web_control"]

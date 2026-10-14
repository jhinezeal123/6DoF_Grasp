"""Vong doi port serial: close() phai that su nha cong.

Do that: pymycobot 4.0.5 khong co API close o MyArmMControl, chi bo tham chieu
la fd van mo -> mo lai trong cung tien trinh bi chot chan cua driver chan.
"""

from types import SimpleNamespace

from m750.robot.arm import MyArmM750
from m750.spec import RobotSpec


class FakeSerial:
    def __init__(self):
        self.is_open = True
        self.closed = False

    def close(self):
        self.closed = True
        self.is_open = False


def test_close_dong_that_port_serial():
    arm = MyArmM750(RobotSpec())
    serial_port = FakeSerial()
    arm._arm = SimpleNamespace(_serial_port=serial_port)
    arm.close()
    assert serial_port.closed is True
    assert arm.is_open is False


def test_close_khong_lo_khi_thiet_bi_khong_co_serial():
    arm = MyArmM750(RobotSpec())
    arm._arm = SimpleNamespace()  # khong co _serial_port
    arm.close()
    assert arm.is_open is False


def test_close_khi_chua_mo_thi_khong_lam_gi():
    arm = MyArmM750(RobotSpec())
    arm.close()
    assert arm.is_open is False

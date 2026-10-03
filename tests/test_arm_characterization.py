"""Hành vi serial hiện tại, chỉ dùng thiết bị giả và không mở cổng robot."""

import m750.arm as arm_module
from m750.arm import MyArmM750
from m750.spec import RobotSpec


class Device:
    def __init__(self):
        self.angles = [0.0] * 6
        self.commands = []
        self.powered = 0

    def get_angles(self):
        return self.angles

    def write_angles(self, angles, speed):
        self.commands.append((list(angles), speed))
        self.angles = list(angles)
        return 1

    def power_on(self):
        self.powered = 1
        return -1

    def is_powered_on(self):
        return self.powered


def make_arm(monkeypatch):
    device = Device()
    opened = []
    def factory(port, baudrate):
        opened.append((port, baudrate))
        return device
    monkeypatch.setattr(arm_module, "MyArmMControl", factory)
    monkeypatch.setattr(arm_module, "_held", lambda port: [])
    arm = MyArmM750(RobotSpec(port="/dev/fake-arm"))
    return arm, device, opened


def test_open_is_lazy_idempotent_and_close_releases_ownership(monkeypatch):
    arm, device, opened = make_arm(monkeypatch)
    assert opened == []
    assert arm.open() is device
    assert arm.open() is device
    assert opened == [("/dev/fake-arm", 1_000_000)]
    arm.close()
    assert not arm.is_open


def test_power_uses_feedback_instead_of_vendor_ack(monkeypatch):
    arm, device, _ = make_arm(monkeypatch)
    monkeypatch.setattr(arm_module.time, "sleep", lambda seconds: None)
    assert arm.power_on()
    assert device.powered == 1


def test_single_joint_preserves_the_other_five_targets(monkeypatch):
    arm, device, _ = make_arm(monkeypatch)
    device.angles = [1., 2., 3., 4., 5., 6.]
    assert arm.set_joint(3, 12., speed=20, wait=False)
    assert device.commands == [([1., 2., 12., 4., 5., 6.], 20)]


def test_verified_joint_move_keeps_retry_and_target_contract(monkeypatch):
    arm, device, _ = make_arm(monkeypatch)
    monkeypatch.setattr(arm, "wait_settled", lambda timeout: True)
    assert arm.write_joints([10.] * 6, speed=15)
    assert device.commands == [([10.] * 6, 15)]

"""Offline checks for the new robot and perception adapters."""

from types import SimpleNamespace

import numpy as np
import pytest

from m750 import PerceptionRequest, RobotControl, TcpPose
from m750.perception.adapters.grasppose import GraspPosePerceptionAdapter
from m750.robot.adapters.mujoco import MujocoRobotDriver
from m750.robot.adapters.pymycobot import PymycobotRobotDriver


def test_both_mujoco_backends_agree_on_the_tool0_frame():
    """The driver and the validation harness must share one tool0 calibration.

    They are independent code paths over the same robot model: robot.adapters
    builds the tool0 frame to command the arm, sim.adapters builds it to place a
    grasp. While each kept its own copy of the 118 mm offset and the flange
    rotation, changing one silently left the other with a stale idea of where the
    tool is, and only a physical run would show it.
    """
    import mujoco
    from scipy.spatial.transform import Rotation

    from m750.sim.adapters.mujoco import ValidationWorld

    joints = (0.1, -0.2, 0.3, -0.1, 0.2, -0.3)
    driver = MujocoRobotDriver()
    world = ValidationWorld()
    try:
        assert driver.move_joints(joints)
        world.data.qpos[world.qaddr] = joints
        mujoco.mj_forward(world.model, world.data)

        pose = driver.read_state().tcp_pose
        tool = world.tool0()

        assert np.allclose(tool[:3, 3], pose.position_m, atol=1e-9)
        quaternion = Rotation.from_matrix(tool[:3, :3]).as_quat()
        # q and -q are the same rotation, so compare the axis, not the sign.
        assert abs(float(np.dot(quaternion, pose.quaternion_xyzw))) == pytest.approx(
            1.0, abs=1e-9)
    finally:
        world.close()
        driver.close()


def test_mujoco_driver_executes_robot_control_without_hardware():
    driver = MujocoRobotDriver()
    try:
        app = RobotControl(driver)
        initial = app.state()
        assert initial.connected and initial.ready
        assert app.move_joints(initial.joints_rad)
        assert app.set_gripper(0.02)
        assert app.state().gripper_opening_m == pytest.approx(0.02)
    finally:
        driver.close()


class FakeDevice:
    def __init__(self):
        self.gripper_command = None

    def get_gripper_value(self):
        return 50

    def set_gripper_value(self, value, speed):
        self.gripper_command = (value, speed)
        return 1

    def is_powered_on(self):
        return 1


class FakeArm:
    q_deg = (0.0,) * 6

    def __init__(self):
        self.device = FakeDevice()
        self.joint_command = None

    def open(self):
        return self.device

    def write_joints(self, values, speed, timeout_s):
        self.joint_command = (values, speed, timeout_s)
        return True

    def temperatures(self):
        return (30.0,) * 6

    def close(self):
        pass


class FakeController:
    def __init__(self):
        self.kin = SimpleNamespace(
            fk_tool0=lambda q: SimpleNamespace(
                translation=np.zeros(3), rotation=np.eye(3)
            )
        )
        self.tcp_command = None

    def move_tcp_to(self, coords, speed, tol, timeout_s):
        self.tcp_command = (coords, speed, tol, timeout_s)
        return True


def test_pymycobot_driver_converts_public_units_without_serial_port():
    arm = FakeArm()
    controller = FakeController()
    driver = PymycobotRobotDriver(arm=arm, controller=controller)
    state = driver.read_state()
    assert state.connected and state.ready
    assert state.gripper_opening_m == pytest.approx(0.0345)
    assert driver.move_joints((0.1,) * 6)
    assert arm.joint_command[0] == pytest.approx((np.degrees(0.1),) * 6)
    assert driver.set_gripper(0.0345)
    assert arm.device.gripper_command == (50, 30)
    assert driver.move_tcp(TcpPose((0.1, 0.2, 0.3), (0.0, 0.0, 0.0, 1.0)))
    assert controller.tcp_command[0] == pytest.approx([100.0, 200.0, 300.0, 0.0, 0.0, 0.0])


def test_perception_adapter_maps_estimator_contract():
    class FakeEstimator:
        def estimate(self, image, prompt_id, **kwargs):
            assert image == "frame"
            assert prompt_id == "cube"
            assert kwargs["top"] == 2
            return SimpleNamespace(
                grasps=(SimpleNamespace(
                    score=0.9,
                    width_m=0.03,
                    translation_m=(0.1, 0.2, 0.3),
                    rotation=np.eye(3),
                ),),
                depth_m=0.4,
            )

    result = GraspPosePerceptionAdapter(FakeEstimator()).infer(
        PerceptionRequest(image="frame", prompt_id="cube", top=2)
    )
    assert result.depth_m == pytest.approx(0.4)
    assert len(result.grasps) == 1
    assert result.grasps[0].position_m == pytest.approx((0.1, 0.2, 0.3))
    assert result.grasps[0].quaternion_xyzw == pytest.approx((0.0, 0.0, 0.0, 1.0))

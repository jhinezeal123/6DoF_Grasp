from __future__ import annotations

import ast
import inspect
from pathlib import Path

import m750
import m750.robot.application as robot_application
from m750 import (
    AffineJointMapper,
    PowerController,
    RangeGripperMapper,
    RealToSim,
    RobotControl,
    RobotDriver,
    RobotState,
    SimToReal,
    TcpPose,
)


class MemoryDriver(RobotDriver):
    def __init__(self, backend: str) -> None:
        self.backend = backend
        self.joints = (0.0,) * 6
        self.gripper = 0.0
        self.pose = TcpPose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        self.closed = False

    @property
    def joint_limits_rad(self):
        return ((-3.14, 3.14),) * 6

    @property
    def max_gripper_opening_m(self):
        return 0.08

    def read_state(self):
        return RobotState(
            joints_rad=self.joints,
            gripper_opening_m=self.gripper,
            tcp_pose=self.pose,
            connected=True,
            ready=True,
            metadata={"backend": self.backend},
        )

    def move_joints(self, joints_rad):
        values = tuple(float(v) for v in joints_rad)
        if len(values) != 6:
            return False
        self.joints = values
        return True

    def move_tcp(self, pose):
        self.pose = pose
        return True

    def set_gripper(self, opening_m):
        value = float(opening_m)
        if not 0.0 <= value <= self.max_gripper_opening_m:
            return False
        self.gripper = value
        return True

    def close(self):
        self.closed = True


def test_root_exposes_stable_interfaces():
    assert m750.RobotDriver is RobotDriver
    assert m750.RobotControl is RobotControl
    assert hasattr(m750, "PerceptionProvider")
    assert hasattr(m750, "SimToReal")
    assert hasattr(m750, "RealToSim")


def test_robot_control_substitutes_backends_without_application_changes():
    for backend in ("real", "simulation"):
        driver = MemoryDriver(backend)
        app = RobotControl(driver)
        assert app.move_joints((0.1, 0.2, 0.3, 0.4, 0.5, 0.6))
        assert app.set_gripper(0.03)
        assert app.state().metadata["backend"] == backend


def test_real_to_sim_uses_injected_mapping():
    real = MemoryDriver("real")
    sim = MemoryDriver("simulation")
    real.joints = (1.0, 2.0, 3.0, 0.0, 0.0, 0.0)
    real.gripper = 0.02

    mapper = AffineJointMapper(offset_rad=(-0.1, -0.2, -0.3, 0.0, 0.0, 0.0))
    result = RealToSim(
        real,
        sim,
        target_gripper=sim,
        mapper=mapper,
    ).execute()

    assert result.ok
    expected = (0.9, 1.8, 2.7, 0.0, 0.0, 0.0)
    assert all(abs(a - b) < 1e-12 for a, b in zip(sim.joints, expected))
    assert sim.gripper == 0.02


def test_gripper_range_mapping_preserves_normalized_opening():
    mapper = RangeGripperMapper(source_max_m=0.08, target_max_m=0.069)
    assert abs(mapper.map(0.04) - 0.0345) < 1e-12


def test_sim_to_real_depends_on_same_contract():
    sim = MemoryDriver("simulation")
    real = MemoryDriver("real")
    sim.joints = (0.3,) * 6

    result = SimToReal(sim, real).execute(copy_gripper=False)

    assert result.ok
    assert real.joints == (0.3,) * 6


def test_power_capability_is_not_forced_on_every_robot_driver():
    assert not issubclass(RobotDriver, PowerController)


def test_application_layer_does_not_import_concrete_adapters():
    tree = ast.parse(inspect.getsource(robot_application))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
            imports.extend(alias.name for alias in node.names)
    forbidden = ("adapters", "m750.ros", "mujoco", "pymycobot")
    assert not [name for name in imports if any(word in name.lower() for word in forbidden)]


def test_new_real_backend_is_pymycobot_not_ros():
    root = Path(__file__).resolve().parents[1]
    adapter = (root / "src/m750/robot/adapters/pymycobot.py").read_text()
    assert "MyArmM750" in adapter
    assert "m750.ros" not in adapter
    assert not (root / "src/m750/robot/adapters/ros.py").exists()


def test_perception_adapter_is_pinned_to_shared_estimator_commit():
    root = Path(__file__).resolve().parents[1]
    adapter = (root / "src/m750/perception/adapters/grasppose.py").read_text()
    assert "5703506a9d012eaf807387e305cfba4c68d0d6e3" in adapter
    assert "grasppose.api import get_estimator" in adapter
    assert "graspgroup" not in adapter

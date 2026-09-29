"""Public API for m750.

The root exports stable abstractions and application use cases only. Heavy
MuJoCo, OpenCV and pymycobot implementations are imported explicitly from
feature adapters by the composition root.
"""

__version__ = "0.2.0"

from .perception import (
    GraspCandidate,
    PerceptionProvider,
    PerceptionRequest,
    PerceptionResult,
)
from .robot import (
    CartesianController,
    DriverLifecycle,
    EmergencyStopController,
    GripperController,
    JointLimits,
    JointPositionController,
    JointTuple,
    PowerController,
    RobotControl,
    RobotDriver,
    RobotState,
    RobotStateReader,
    TcpPose,
)
from .sync import (
    AffineJointMapper,
    GripperMapper,
    IdentityGripperMapper,
    IdentityJointMapper,
    JointMapper,
    RangeGripperMapper,
    RealToSim,
    SimToReal,
    TransferResult,
)


def __getattr__(name):
    """Lazy compatibility imports for the pre-refactor public API."""

    if name == "MyArmM750":
        from .arm import MyArmM750
        return MyArmM750
    if name == "ArmController":
        from .control import ArmController
        return ArmController
    if name == "ArmKinematics":
        from .kinematics import ArmKinematics
        return ArmKinematics
    if name == "IKSolver":
        from .ik import IKSolver
        return IKSolver
    if name == "SafetyGate":
        from .safety import SafetyGate
        return SafetyGate
    if name == "ring_views":
        from .viewpoints import ring_views
        return ring_views
    raise AttributeError("module 'm750' has no attribute %r" % name)


__all__ = [
    "AffineJointMapper",
    "CartesianController",
    "DriverLifecycle",
    "EmergencyStopController",
    "GraspCandidate",
    "GripperController",
    "GripperMapper",
    "IdentityGripperMapper",
    "IdentityJointMapper",
    "JointLimits",
    "JointMapper",
    "JointPositionController",
    "JointTuple",
    "PerceptionProvider",
    "PerceptionRequest",
    "PerceptionResult",
    "PowerController",
    "RangeGripperMapper",
    "RealToSim",
    "RobotControl",
    "RobotDriver",
    "RobotState",
    "RobotStateReader",
    "SimToReal",
    "TcpPose",
    "TransferResult",
]

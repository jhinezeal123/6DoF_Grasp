"""Pipeline VLA lắp ghép cho HTC SDK.

Ví dụ tối thiểu::

    from pipeline import ActionSpec, Pipeline
    from pipeline.adapters import CameraRobotSource, ModelPolicy, RobotSink

    spec = ActionSpec("joint_position", tuple(robot.JOINT_NAMES), ("rad",) * 6, "joint")
    pipeline = Pipeline(
        source=CameraRobotSource(camera, robot),
        policy=ModelPolicy(model, spec),
        sink=RobotSink(robot, spec),
    )
    pipeline.run("pick up the red block", max_steps=10)

Các adapter trong ``pipeline.adapters`` chỉ là lựa chọn mặc định; người dùng có
thể viết Source/Policy/Sink riêng mà không sửa runner.
"""

from .interfaces import Module, Policy, Sink, Source
from .runner import Pipeline, PipelineStopped
from .types import Action, ActionSpec, Episode, Observation, StepResult

__all__ = [
    "Action",
    "ActionSpec",
    "Episode",
    "Module",
    "Observation",
    "Pipeline",
    "PipelineStopped",
    "Policy",
    "Sink",
    "Source",
    "StepResult",
]

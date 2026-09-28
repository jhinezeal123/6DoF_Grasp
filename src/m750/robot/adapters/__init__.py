"""Concrete robot backends.

Import a backend explicitly in the composition root:

    from m750.robot.adapters.ros import RosRobotDriver
    from m750.robot.adapters.mujoco import MujocoRobotDriver

The package intentionally does not import either backend eagerly because ROS
and MuJoCo are optional runtime dependencies.
"""

__all__ = []

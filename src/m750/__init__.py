"""m750 - Package dieu khien myArm M750.

Import theo muc can, khong keo theo thu vien nang:
  from m750.spec import RobotSpec              # chi hang so (stdlib)
  from m750.kinematics import ArmKinematics    # pinocchio
  from m750 import MyArmM750, ArmController    # + pymycobot (chi tren server)

    MyArmM750        - so huu port serial pymycobot (retry, doc goc)
    ArmKinematics    - FK/IK bang Pinocchio + URDF (test offline duoc)
    IKSolver         - IK 6 DOF least_squares multi-restart
    ArmController    - API muc ung dung: gripper_pose/move_gripper_to/state
    SafetyGate       - gioi han khop + drop-check duong di
    ring_views       - 6 pose GraspNeRF quanh mot tam

Cac nhanh khac (import rieng de khong keo dependency):
    m750.camera      - MjpegStream + capture (opencv)
    m750.preview     - web xem truoc mo phong (mujoco)
    m750.pipeline    - Source -> Policy -> Sink VLA harness (doc lai README rieng)
    m750.ros         - stack ROS 2 (rclpy): bridge/robot/camera
    m750.webui       - web control day du (mujoco + rclpy)
"""

__version__ = "0.1.0"

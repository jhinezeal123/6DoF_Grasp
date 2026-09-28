# 6DoF_Grasp — myArm M750

Điều khiển myArm M750 thật, chạy MuJoCo simulation, đồng bộ real2sim /
sim2real và gắn perception/VLA qua các interface ổn định.

## Kiến trúc

Code mới đi theo feature-first. Application chỉ phụ thuộc abstraction; ROS,
MuJoCo và model perception nằm ở adapter.

    src/m750/
    ├── robot/
    │   ├── contracts.py          # RobotDriver + các interface nhỏ theo ISP
    │   ├── types.py              # RobotState, TcpPose
    │   ├── application.py        # RobotControl, không biết ROS/MuJoCo
    │   └── adapters/
    │       ├── ros.py            # RosRobotDriver -> robot thật
    │       └── mujoco.py         # MujocoRobotDriver -> simulation, không import ROS
    ├── sync/
    │   ├── application.py        # RealToSim, SimToReal
    │   └── mapping.py            # JointMapper; calibration mở rộng bằng policy
    ├── perception/
    │   ├── contracts.py          # PerceptionProvider
    │   ├── types.py              # GraspCandidate, PerceptionResult
    │   └── adapters/
    │       └── grasppose.py      # optional adapter cho pipeline_grasppose
    ├── pipeline/                 # Source -> Policy -> Sink VLA
    ├── ros/                      # legacy ROS implementation được adapter bọc lại
    ├── webui/                    # UI hiện tại; migrate dần sang application mới
    ├── arm.py / control.py       # API direct-pymycobot cũ, giữ compatibility
    └── model/                    # URDF/MJCF/scene/meshes

Public interface được expose ở package root:

    from m750 import (
        RobotDriver,
        RobotControl,
        RobotState,
        TcpPose,
        RealToSim,
        SimToReal,
        PerceptionProvider,
    )

Root không import ROS, MuJoCo, OpenCV, pymycobot hay TensorRT. Concrete backend
chỉ được chọn ở composition root.

## Real robot và simulation dùng chung application

Robot thật:

    from m750 import RobotControl
    from m750.robot.adapters.ros import RosRobotDriver

    app = RobotControl(RosRobotDriver())
    app.move_joints([0, 0, 0, 0, 0, 0])

Simulation chỉ thay driver, application không đổi:

    from m750 import RobotControl
    from m750.robot.adapters.mujoco import MujocoRobotDriver

    app = RobotControl(MujocoRobotDriver())
    app.move_joints([0, 0, 0, 0, 0, 0])

MujocoRobotDriver là implementation ngang hàng với RosRobotDriver, không kế
thừa robot thật và không có đường publish ROS. Điều này tránh vấn đề LSP của
mô hình cũ FakeRobot(Robot).

## real2sim / sim2real

    import numpy as np

    from m750 import AffineJointMapper, RangeGripperMapper, RealToSim, SimToReal
    from m750.robot.adapters.mujoco import MujocoRobotDriver
    from m750.robot.adapters.ros import RosRobotDriver

    real = RosRobotDriver()
    sim = MujocoRobotDriver()

    real_to_sim = AffineJointMapper(
        offset_rad=tuple(np.radians([-1, 2, 0, 0, 0, 0]))
    )

    real_to_sim_gripper = RangeGripperMapper(
        real.max_gripper_opening_m, sim.max_gripper_opening_m
    )
    sim_to_real_gripper = RangeGripperMapper(
        sim.max_gripper_opening_m, real.max_gripper_opening_m
    )

    RealToSim(
        real, sim,
        target_gripper=sim,
        mapper=real_to_sim,
        gripper_mapper=real_to_sim_gripper,
    ).execute()
    SimToReal(
        sim, real,
        target_gripper=real,
        gripper_mapper=sim_to_real_gripper,
    ).execute()

Nếu mapping thay đổi, thêm JointMapper/GripperMapper mới; không sửa RealToSim,
SimToReal hay RobotControl.

## Tích hợp pipeline_grasppose

Repo perception có thể được cài/đưa vào PYTHONPATH riêng. Core application
không phụ thuộc YOLOE/Lite-Mono/VGN/TensorRT.

    from m750 import PerceptionRequest
    from m750.perception.adapters.grasppose import GraspPosePerceptionAdapter

    perception = GraspPosePerceptionAdapter()
    perception.open()

    result = perception.infer(
        PerceptionRequest(
            image=rgb,
            prompt_id="blue_cube",
            camera_matrix=K,
        )
    )

    for grasp in result.grasps:
        print(grasp.score, grasp.position_m, grasp.quaternion_xyzw, grasp.width_m)

Adapter đọc trực tiếp PipelineResult.grasp.graspgroup của
jhinezeal123/pipeline_grasppose và chuyển sang GraspCandidate; application
không biết format 17 phần tử của VGN.

## SOLID áp dụng

- SRP: robot control, synchronization, perception và VLA pipeline là các
  feature độc lập.
- OCP: thêm backend mới bằng cách implement RobotDriver; thêm calibration bằng
  JointMapper; thêm perception bằng PerceptionProvider.
- LSP: real và sim thực hiện cùng contract. Simulator không subclass robot thật.
- ISP: power/emergency-stop tách khỏi RobotDriver; simulation không phải
  implement hành vi phần cứng vô nghĩa.
- DIP: RobotControl, RealToSim, SimToReal và pipeline adapter chỉ phụ thuộc
  interface.

## API cũ

Các module cũ vẫn tồn tại để Web UI/CLI hiện tại chạy trong giai đoạn migrate.
Các import sau vẫn được hỗ trợ lazy:

    from m750 import MyArmM750, ArmController, ArmKinematics, IKSolver

Code mới không nên phụ thuộc trực tiếp m750.ros.robot.Robot hoặc
m750.webui.fake_robot.FakeRobot.

## Cài đặt và chạy hiện tại

    python -m pip install -e . --no-deps

Dependencies được quản lý bởi môi trường robot: pymycobot, pinocchio, scipy,
numpy, opencv-python, mujoco; ROS 2 cần thêm rclpy và message packages.

| Việc | Lệnh |
|---|---|
| Trạng thái tay | m750-state |
| Camera MJPEG | m750-camera |
| Preview | m750-preview |
| ROS 2 + Web UI | bash run_web.sh |
| FK/IK offline | python -m pytest tests/test_kinematics.py |
| Test kiến trúc mới | python -m pytest tests/test_feature_first_architecture.py |

Quy ước pose legacy direct-pymycobot vẫn giữ nguyên: mm, Euler XYZ độ, hệ URDF.
Public TcpPose mới dùng mét và quaternion ROS [qx, qy, qz, qw].

# 6DoF_Grasp — myArm M750

Điều khiển myArm M750 thật bằng pymycobot official, chạy MuJoCo simulation,
đồng bộ real2sim/sim2real và plug perception qua interface ổn định.

## Kiến trúc

Code mới đi theo feature-first. Application chỉ phụ thuộc abstraction; chi tiết
pymycobot, MuJoCo và perception nằm ở adapter.

    src/m750/
    ├── robot/
    │   ├── contracts.py
    │   ├── types.py
    │   ├── application.py
    │   └── adapters/
    │       ├── pymycobot.py      # real robot, official Elephant Robotics API
    │       └── mujoco.py         # simulation
    ├── sync/
    │   ├── application.py        # RealToSim, SimToReal
    │   └── mapping.py            # JointMapper, GripperMapper
    ├── perception/
    │   ├── contracts.py
    │   ├── types.py
    │   └── adapters/
    │       └── grasppose.py      # pinned GraspEstimator adapter
    ├── pipeline/
    ├── arm.py / control.py       # pymycobot + Pinocchio implementation
    └── model/

Dependency direction:

    RobotControl ------> RobotDriver <------ PymycobotRobotDriver
                              ^
                              +-------------- MujocoRobotDriver

    RealToSim / SimToReal --> small robot interfaces

    robot application --> PerceptionProvider <-- GraspPosePerceptionAdapter
                                                <-- GraspEstimator

Public contracts/use-cases được expose ở root. Root không import pymycobot,
MuJoCo hay TensorRT. Concrete backend chỉ được chọn ở composition root.

## Real robot và simulation dùng chung application

Robot thật:

    from m750 import RobotControl
    from m750.robot.adapters.pymycobot import PymycobotRobotDriver

    driver = PymycobotRobotDriver()
    app = RobotControl(driver)

    driver.power_on()
    app.move_joints([0, 0, 0, 0, 0, 0])

Simulation chỉ thay driver:

    from m750 import RobotControl
    from m750.robot.adapters.mujoco import MujocoRobotDriver

    app = RobotControl(MujocoRobotDriver())
    app.move_joints([0, 0, 0, 0, 0, 0])

PymycobotRobotDriver cô lập quy ước vendor:
- public joint là radian, pymycobot là degree;
- public gripper là mét, pymycobot là 0..100;
- public Cartesian là mét + quaternion;
- real Cartesian dùng Pinocchio IK rồi gửi write_angles qua pymycobot.

Không dùng ROS2 cho backend thật.

## real2sim / sim2real

    from m750 import RangeGripperMapper, RealToSim, SimToReal
    from m750.robot.adapters.mujoco import MujocoRobotDriver
    from m750.robot.adapters.pymycobot import PymycobotRobotDriver

    real = PymycobotRobotDriver()
    sim = MujocoRobotDriver()

    RealToSim(
        real,
        sim,
        target_gripper=sim,
        gripper_mapper=RangeGripperMapper(
            real.max_gripper_opening_m,
            sim.max_gripper_opening_m,
        ),
    ).execute()

    SimToReal(
        sim,
        real,
        target_gripper=real,
        gripper_mapper=RangeGripperMapper(
            sim.max_gripper_opening_m,
            real.max_gripper_opening_m,
        ),
    ).execute()

Calibration joint khác nhau thì inject JointMapper khác, không sửa use-case.

## Perception: pin đúng commit

Integration target chính xác:

    repository: jhinezeal123/pipeline_grasppose
    commit: 5703506a9d012eaf807387e305cfba4c68d0d6e3

Không target nhánh main.

Commit này expose public contract GraspEstimator cùng EstimateResult. Adapter
trong repo này dùng đúng boundary đó, không truy cập PipelineResult/graspgroup
nội bộ.

Chuẩn bị checkout:

    git clone https://github.com/jhinezeal123/pipeline_grasppose.git
    cd pipeline_grasppose
    git checkout 5703506a9d012eaf807387e305cfba4c68d0d6e3
    export PYTHONPATH="$PWD:$PYTHONPATH"

Sử dụng:

    from m750 import PerceptionRequest
    from m750.perception.adapters.grasppose import GraspPosePerceptionAdapter

    perception = GraspPosePerceptionAdapter()
    perception.open()

    result = perception.infer(
        PerceptionRequest(
            image=rgb,
            prompt_id="blue_cube",
            camera_matrix=K,
            max_width_m=0.069,
            top=5,
        )
    )

Có thể inject bất kỳ implementation nào của GraspEstimator, ví dụ
WorkerGraspEstimator, mà không sửa application robot.

## SOLID

- SRP: robot control, sync, perception và VLA là feature riêng.
- OCP: thêm backend bằng RobotDriver; calibration bằng mapper; perception bằng
  PerceptionProvider/GraspEstimator adapter.
- LSP: PymycobotRobotDriver và MujocoRobotDriver cùng contract; sim không kế
  thừa real.
- ISP: power và emergency-stop là optional capability.
- DIP: application/sync/pipeline chỉ phụ thuộc abstraction.

## ROS2 legacy

Thư mục ros/ và Web UI ROS cũ vẫn còn tạm thời để tránh refactor phá toàn bộ
legacy UI trong cùng một commit. Kiến trúc mới không dùng ROS2, không có
RosRobotDriver và code mới không được import m750.ros.

Bước migration tiếp theo là chuyển UI cần giữ sang RobotControl +
PymycobotRobotDriver/MujocoRobotDriver, sau đó có thể xóa hẳn legacy ROS.

## Cài đặt

    python -m pip install -e . --no-deps

Runtime robot thật: pymycobot, pinocchio, scipy, numpy.
Simulation: mujoco.
Perception: checkout pipeline_grasppose đúng commit ở trên.

Test kiến trúc:

    python -m pytest tests/test_feature_first_architecture.py

Public TcpPose dùng mét + quaternion [qx, qy, qz, qw].

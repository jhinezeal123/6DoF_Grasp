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
                                                <-- WorkerGraspEstimator
                                                     | Unix socket
                                                     v
                                             pipeline_grasppose worker

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

## Môi trường cho hai repo trên KTMT

Mỗi repo tự quản dependencies của mình. `pipeline_grasppose` tạo `.venv` bằng
`scripts/prepare.sh` và dùng các gói Torch/CUDA/TensorRT của JetPack. Repo này
có `environment.yml` riêng cho Python 3.10, NumPy 1.26.4 và Pinocchio 2.7.0.
Trên Jetson/Linux, tạo Conda environment tại `.venv` của **repo này**:

```bash
cd /path/to/6DoF_Grasp
bash scripts/setup-env.sh
.venv/bin/python -m pytest -q
```

Script dùng `CONDA_EXE` nếu đã đặt, sau đó tìm `conda` trên `PATH` hoặc tại
`~/miniforge3/bin/conda`. Nó tạo/cập nhật `.venv` và cài package `m750` ở chế
độ editable. `.venv` đã nằm trong `.gitignore`. Pytest bỏ qua script kiểm tra
robot vật lý; các test backend dùng MuJoCo hoặc thiết bị giả.

Môi trường Conda của 6DoF không cần cài Torch/TensorRT hay import package
`grasppose`. Worker inference chạy trong `.venv` của pipeline và trao đổi với
6DoF qua Unix socket trên cùng máy. Chỉ dùng cách import pipeline trực tiếp
khi chủ động chạy cả hai repo trong một Python environment tương thích.

## Perception: worker giữa hai môi trường

Integration target chính xác:

    repository: jhinezeal123/pipeline_grasppose
    commit: 5703506a9d012eaf807387e305cfba4c68d0d6e3

Không target nhánh main.

Commit này expose public contract GraspEstimator cùng EstimateResult và worker
socket. Adapter trong repo này chỉ dùng kết quả grasp công khai, không truy cập
PipelineResult/graspgroup nội bộ.

Chuẩn bị và chạy pipeline trong environment của **pipeline repo**:

```bash
git clone https://github.com/jhinezeal123/pipeline_grasppose.git
cd pipeline_grasppose
git checkout 5703506a9d012eaf807387e305cfba4c68d0d6e3
bash scripts/prepare.sh
bash scripts/worker.sh start
```

Worker mặc định tạo socket tại `<pipeline checkout>/.runtime/worker.sock`.
Đặt đường dẫn tuyệt đối tới socket đó trong process 6DoF:

```bash
cd /path/to/6DoF_Grasp
export GRASP_WORKER_SOCKET=/path/to/pipeline_grasppose/.runtime/worker.sock
.venv/bin/python your_perception_client.py
```

Composition root của client 6DoF tạo adapter như sau (hai process phải cùng
máy và có quyền đọc file ảnh tạm):

```python
from m750 import PerceptionRequest
from m750.perception.adapters.grasppose import GraspPosePerceptionAdapter
from m750.perception.adapters.grasppose_worker import WorkerGraspEstimator

perception = GraspPosePerceptionAdapter(WorkerGraspEstimator())
perception.open()  # kiểm tra worker đã sẵn sàng

result = perception.infer(
    PerceptionRequest(
        image=rgb,
        prompt_id="cube",
        camera_matrix=K,
        max_width_m=0.069,
        top=5,
    )
)
perception.close()
```

Client này chỉ gửi ảnh và tham số inference; nó không gửi lệnh robot. Nếu cả hai
repo được cài trong cùng một environment tương thích, có thể bỏ estimator ở
constructor để dùng `grasppose.api.get_estimator()` trong cùng process.

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

Trên KTMT dùng `bash scripts/setup-env.sh` như hướng dẫn ở trên. Manifest
`environment.yml` khai báo runtime 6DoF (pymycobot, Pinocchio, NumPy, SciPy,
OpenCV, MuJoCo) và dependencies kiểm thử; pipeline quản lý runtime của nó trong
repo riêng. GitHub Actions trên x86 dùng extra `.[ci]` trong `pyproject.toml`.
`pip install -e . --no-deps --no-build-isolation` vẫn dùng được nếu các
dependencies đã được cài sẵn trong environment tương thích.

Test kiến trúc:

    .venv/bin/python -m pytest tests/test_feature_first_architecture.py

Public TcpPose dùng mét + quaternion [qx, qy, qz, qw].

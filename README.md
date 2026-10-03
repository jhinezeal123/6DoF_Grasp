# 6DoF_Grasp — myArm M750

Điều khiển robot thật bằng pymycobot, chạy MuJoCo, đồng bộ thật/mô phỏng và
nhận perception từ `pipeline_grasppose` qua contract. Hai repo là hai module
với hai môi trường riêng.

Đọc [bản đồ code](docs/kien-truc-vi.md) và
[review đủ 9 PR](docs/review-pr-vi.md). Hướng dẫn validation ở
[docs/sim-grasp-validation.md](docs/sim-grasp-validation.md).

## Nơi tìm code

| Mục đích | Nơi đọc |
| --- | --- |
| Thông số/calibration | `src/m750/spec.py` |
| Serial pymycobot, controller, FK/IK/safety | `src/m750/robot/` |
| Contract chung và chọn backend | `robot/contracts.py`, `robot/adapters/` |
| Đồng bộ và mapping | `sync/` |
| Perception public API/Unix client | `perception/` |
| Validation simulation, reporting | `sim/` |
| VLA policy runner | `pipeline/` |
| Web UI ROS cũ | `webui/`, `ros/` |

File root `arm.py/control.py/ik.py/...` chỉ giữ đường import cũ. Implementation
duy nhất nằm trong `robot/`. Web UI đã tách HTTP, HTML và process render;
vẫn dùng ROS, chưa được chuyển sang pymycobot. Code điều khiển mới không dùng ROS.

## Cài môi trường của repo này

```bash
bash scripts/setup-env.sh
.venv/bin/python -m pytest -q
```

Script dùng `CONDA_EXE`, conda trên PATH hoặc `~/miniforge3/bin/conda`.
`environment.yml` định nghĩa Python 3.10, Pinocchio, pymycobot, MuJoCo và các
dependency riêng. Không dùng `.venv` của pipeline để điều khiển robot.
Test phần cứng thủ công bị loại khỏi pytest; test backend dùng thiết bị giả.

## CLI ngay tại root

```bash
./robot --help
./robot state
./robot angles
./robot power-on
./robot --speed 15 joint 3 10
./robot gripper 30
./robot stop
```

`joints/joint` dùng độ; `gripper` mặc định mm, hỗ trợ `--unit percent`.
`tcp` dùng mm + Euler XYZ độ. Lệnh move gửi chuyển động thật; CLI không tự bật
nguồn. Xem [đầy đủ lệnh và đơn vị](docs/cli-robot-vi.md).
Có thể dùng `.venv/bin/python robot.py` hoặc `m750-robot` sau cài editable.

## Điều khiển qua OOP

```python
from m750 import RobotControl
from m750.robot.adapters.pymycobot import PymycobotRobotDriver

driver = PymycobotRobotDriver()
app = RobotControl(driver)
try:
    state = app.state()
    print(state)
finally:
    app.close()
```

Thay bằng `MujocoRobotDriver` để dùng cùng use case với simulation.
`RobotDriver` dùng joint **radian**, TCP **mét + quaternion xyzw**, gripper
tổng độ mở **mét**. API vendor/`ArmController` cũ dùng degree và mm/Euler XYZ;
conversion được cô lập ở adapter. Cartesian thật dùng Pinocchio IK rồi gửi
`write_angles`; không dùng firmware `write_coords`.

Power và stop là capability riêng; không ép mọi backend giả làm hardware.
Thêm backend bằng contract, thay calibration đồng bộ bằng mapper, không sửa
use case cho từng implementation.

## Perception qua Unix socket

Integration pin hiện tại:
`jhinezeal123/pipeline_grasppose@666c7eb608c5315ea252fd02b3f5446c39198ee6`.
Refactor không đổi pin. Pipeline main đã có thay đổi depth; đừng coi metadata
pin là bằng chứng process worker đang dùng đúng SHA.

Trong checkout/environment của **pipeline**:

```bash
git checkout 666c7eb608c5315ea252fd02b3f5446c39198ee6
bash scripts/prepare.sh
bash scripts/worker.sh start
```

Trong repo này:

```bash
export GRASP_WORKER_SOCKET=/path/to/pipeline_grasppose/.runtime/worker.sock
```

```python
from m750 import PerceptionRequest
from m750.perception.adapters.grasppose import GraspPosePerceptionAdapter
from m750.perception.adapters.grasppose_worker import WorkerGraspEstimator

perception = GraspPosePerceptionAdapter(WorkerGraspEstimator())
perception.open()
try:
    result = perception.infer(PerceptionRequest(
        image=rgb, prompt_id="cube", camera_matrix=K,
        max_width_m=0.069, top=5,
    ))
finally:
    perception.close()
```

Hai process ở cùng máy và cần đọc được ảnh tạm. Client không import Torch,
TensorRT hay grasppose vào môi trường robot. Grasp ở khung camera; cần extrinsic
và tool calibration trước khi điều khiển trong khung base.

## Refactor và change

Refactor giữ thuật toán, calibration, unit, pin, endpoint và hành vi hiện tại;
test chạy xanh trước/sau. CLI root được thêm ở PR riêng trên nền đã refactor.
Không trộn model/solver/protocol mới trong diff refactor. GitHub CI chạy cả
rendering bằng OSMesa và test Unix socket; offline unit tests không chứng minh
robot thật hay legacy ROS service đã chạy thành công.

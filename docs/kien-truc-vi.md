# Bản đồ code 6DoF

| Trách nhiệm | Nơi đọc |
| --- | --- |
| API public, import nhẹ | `src/m750/__init__.py` |
| CLI pymycobot từ root | `robot`, `robot.py`, `src/m750/robot/cli.py` |
| Danh mục/menu tác vụ | `operator/tasks.py`: `FEATURES` khai báo tên, mô tả và điều kiện; `operator/console.py` chỉ lo menu |
| Thông số robot/calibration/đường model | `spec.py` — `RobotSpec` |
| Contract chung cho robot thật và MuJoCo | `robot/contracts.py`, `robot/types.py` |
| Use case điều khiển qua contract | `robot/application.py` — `RobotControl` |
| Serial, retry, feedback pymycobot | `robot/arm.py` — `MyArmM750` |
| Điều khiển Cartesian + IK + safety | `robot/control.py` — `ArmController` |
| FK, IK, đường đi, viewpoint | `robot/kinematics.py`, `robot/ik.py`, `robot/safety.py`, `robot/viewpoints.py` |
| Adapter chọn backend | `robot/adapters/pymycobot.py`, `robot/adapters/mujoco.py` |
| Đồng bộ thật/mô phỏng và calibration mapping | `sync/application.py`, `sync/mapping.py` |
| Hợp đồng perception | `perception/contracts.py`, `perception/types.py` |
| Pipeline public API và Unix client | `perception/adapters/grasppose.py`, `perception/adapters/grasppose_worker.py` |
| Workflow validation mô phỏng | `sim/application.py` |
| IO/provenance/JSON báo cáo | `sim/reporting.py` |
| Toán camera/grasp và scene | `sim/geometry.py`, `sim/scenario.py` |
| Physics và bridge depth ở venv pipeline | `sim/adapters/` |
| Web UI ROS cũ: HTTP | `webui/http.py` |
| Web UI ROS cũ: trạng thái/use case | `webui/web_control.py` |
| Web UI ROS cũ: HTML thuần | `webui/pages/control.html`, `webui/pages/control.py` |
| Web UI ROS cũ: worker render/stream | `webui/rendering.py`, `webui/streams.py` |
| ROS legacy | `ros/`; không import từ code điều khiển mới |
| VLA orchestration | `pipeline/`; tên này không phải repo perception |

Các file root `arm.py`, `control.py`, `kinematics.py`, `ik.py`, `safety.py`,
`viewpoints.py` là shim cho client cũ. Mỗi tên alias cùng module canonical trong
`robot/`; không có bản implementation thứ hai và monkeypatch cũ vẫn tác động
lên cùng globals. Code mới trong repo dùng đường canonical.

## Đơn vị và tọa độ

`RobotDriver`: sáu joint radian, TCP mét + quaternion **xyzw**, độ mở gripper
tổng tính bằng mét. `ArmController` và vendor pymycobot giữ API cũ: degree,
Cartesian mm + Euler XYZ. Conversion chỉ thực hiện ở adapter/biên CLI.

Perception trả grasp trong camera OpenCV, chưa phải base robot. Phải dùng
extrinsic camera và transform tool0↔gripper đúng trước khi ra lệnh.
Quaternion MuJoCo free joint **wxyz** khác quaternion public; không đảo thứ tự
ngầm. Tool0 offset/rotation có một nguồn ở `spec.py`.

## Điểm mở rộng

Backend mới implement `RobotDriver`; `RobotControl` không cần sửa. Power và stop
là capability riêng; simulator không bắt buộc giả làm phần cứng. Đồng bộ dùng
mapper inject từ ngoài; perception mới implement `PerceptionProvider`.

Không chia file chỉ theo số dòng. File tách ở đây có trách nhiệm riêng: HTTP,
HTML, process render, báo cáo hoặc thuật toán robot. Không thêm base class,
factory registry hay tầng `manager/service/helper` nếu chưa có nhu cầu.

## Hai bước, hai PR

Refactor giữ hành vi, test trước/sau phải xanh. CLI root là change trên nền đó,
nằm trong PR riêng. Thay model, calibration, solver, protocol hoặc migration
ROS/Web UI cũng phải đi theo quy tắc này. Chia commit trong cùng một PR chưa đủ.

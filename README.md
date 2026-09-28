# 6DoF_Grasp — myArm M750

Điều khiển robot tay **myArm M750** (Elephant Robotics) qua 2 đường:
pymycobot trực tiếp (IK Pinocchio) và stack ROS 2 + Web UI.
Kèm pipeline VLA (`Source → Policy → Sink`) để gắn model AI vào.

> Tất cả docstring/comment trong code bằng tiếng Việt không dấu, đậm bài học
> đo trên robot thật — đọc trước khi đổi logic gì.

## Cấu trúc

```
src/m750/             # package chính (pip install -e .)
  spec.py             # RobotSpec: port, baud, FW limits, GRIP_L, URDF — 1 nguồn duy nhất
  arm.py              # MyArmM750: mở port (check port-held), write_joints (retry)
  kinematics.py       # ArmKinematics: FK Pinocchio, test offline được
  ik.py               # IKSolver: IK 6-DOF least_squares + restart
  control.py          # ArmController: gripper_pose / move_gripper_to / state
  safety.py           # SafetyGate: giới hạn khớp + drop-check đường đi
  viewpoints.py       # ring_views: 6 pose GraspNeRF quanh tâm
  camera.py           # MjpegStream (MJPEG HTTP) + capture (chụp 1 khung)
  preview.py          # PreviewServer: web xem trước mô phỏng (render MuJoCo)
  cli.py              # entry points: m750-state / m750-camera / m750-preview
  pipeline/           # VLA harness: Source → Policy → Sink (xem pipeline/README.md)
  ros/                # stack ROS 2: bridge, robot, camera (cần rclpy)
  webui/              # web control đầy đủ (robot thật + fake + camera)
  model/              # URDF + MJCF + scene + meshes (đi theo package)
apps/                 # (chỗ cho app nhỏ sau này)
tools/                # joint_check, check_cameras, benchmark_camera
experiments/vlm/      # thí nghiệm VLM PhysBrain + servo ảnh (P0–P3)
tests/                # test offline (pytest) + test hardware (chạy tay)
docs/                 # tài liệu: hardware-m750, bài học octo, recap, plan
run_web.sh            # khởi động stack ROS 2 + Web UI (server ktmt)
sync.ps1              # đồng bộ local ↔ GitHub ↔ server qua git
```

## Cài đặt (một lần, mỗi máy)

```bash
cd 6DoF_Grasp              # repo root (server: /workspace/6DoF_Grasp/htc)
python -m pip install -e . --no-deps
```

Dependencies (quản lý bởi môi trường, không qua pip ở đây):
`pymycobot`, `pin` (pinocchio), `scipy`, `numpy`, `opencv-python`, `mujoco`,
ROS 2 stack cần thêm `rclpy` + các msg chuẩn.

## Chạy gì

| Việc | Lệnh |
|---|---|
| Xem trạng thái tay (6 khớp, gripper, pose URDF) | `m750-state` |
| Stream MJPEG camera | `m750-camera` (tắt: `touch stop_cam`) |
| Web xem trước mô phỏng + IK + sync thật | `m750-preview` → http://ip:8081/ |
| Stack đầy đủ ROS 2 + Web UI (server) | `bash run_web.sh` → http://ip:8080/ |
| Kiểm tra khớp (P0-A, chạy tay thật) | `python tools/joint_check.py` |
| Test FK/IK offline (không cần robot) | `python -m pytest tests/test_kinematics.py` |
| Test pose hardware (đọc, không ra lệnh) | `python tests/test_gripper_pose_hardware.py` |

API Python chính:

```python
from m750 import MyArmM750, ArmKinematics, IKSolver, ArmController

with MyArmM750() as arm:                # port /dev/ttyACM1, chống 2 tiến trình cùng giữ
    ctrl = ArmController(arm=arm)
    print(ctrl.gripper_pose())          # [x,y,z mm | rx,ry,rz do] hệ URDF
    ctrl.move_gripper_to(400, 0, 250)   # IK + drop-check + retry, trả True/False
    print(ctrl.ring_views(r=150))       # 6 pose GraspNeRF quanh vị trí hiện tại
```

Quy ước pose (đã kiểm chứng đo thật): mm, Euler XYZ độ, hệ URDF,
**rpy=[0,0,0] = gripper chúc thẳng XUỐNG**.

## Đồng bộ local ↔ server

```powershell
.\sync.ps1 status              # so HEAD local vs server
.\sync.ps1 push "message"      # commit + push GitHub, server pull (ff-only)
.\sync.ps1 pull                # local pull
```

Server: `ktmt` (Tailscale), repo tại `/workspace/6DoF_Grasp/htc`.

## Thí nghiệm (lịch sử)

- `docs/lessons-octo-*.md` — vì sao bỏ hướng Octo VLA (finetune mù ngôn ngữ, policy đóng băng)
- `experiments/vlm/` — hướng đang dở: PhysBrain + servo ảnh (P0–P3, P3 chưa có kết quả e2e)
- `docs/plan-vla.md` — chiến lược VLA tiếp theo
- `docs/recap-ktmt.md` — digital twin MuJoCo (simu/ trên server, ngoài git)

## Lint

```bash
ruff check src tests
```

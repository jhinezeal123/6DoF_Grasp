# Đề xuất refactor 6DoF_Grasp theo chuẩn OOP

> Phân tích dựa trên commit `7b645d6` (main) + working tree local `D:\Documents\mujoco\htc`
> (ductocbatdat.py đã phát triển thêm +453 dòng, plan.txt/ktmt/test_gripper_pose.py đã tự xoá local).
>
> Bối cảnh: 43 commit dồn trong 2 ngày (17–18/09) — repo là **vết ước của 2 ngày thí nghiệm tốc độ cao**
> (Octo bỏ → PhysBrain VLM → điều khiển trực tiếp), không phải legacy dài hạn. Đó là lý do cấu trúc chưa kịp
> hình thành. Và cũng vì thế lộ trình refactor dưới đây được thiết kế **nhẹ (2–4 buổi)** — làm nặng hơn thì
> chắc chắn sẽ bị bỏ dở giữa đường như chính các thí nghiệm.

---

## 1. Chẩn đoán: vì sao repo "không làm được cái gì cả"

Repo **không phải một project — mà là 5 project + 1 kho rác trộn lẫn nhau**:

| # | "Project" nằm trong repo | Trạng thái |
|---|---|---|
| 1 | `SDK/` — stack ROS2 + Web UI điều khiển robot (~5.800 dòng Python) | Sống, ổn định |
| 2 | `test_2_vlm/` — pipeline VLM PhysBrain + servo ảnh (P0–P3) | Sống, đang dở |
| 3 | `ductocbatdat.py` — interface điều khiển trực tiếp pymycobot + IK Pinocchio (769 dòng bản local) | Sống, phát triển mạnh nhất (10 commit — nhiều nhất repo) |
| 4 | `test_1_octo/` — thử nghiệm Octo VLA | **ĐÃ CHẾT** (kết luận trong `VI_SAO_TAO_LAO.md`), để lại **128 file / 27.500 dòng code vendored của thư viện Octo** |
| 5 | `simu/` MuJoCo digital twin | Nằm ngoài git, trên server (ktmt/recap.txt) |

Số liệu đo được:

- **78% Python trong repo là code vendored đã chết** (27.5K dòng octo) — code tự viết chỉ ~7.6K dòng / 47 file.
- **Phân bổ code tự viết** (đo theo LOC thực): `SDK/pipeline` 671 (sạch, là "đảo" không phụ thuộc ai), `SDK/program` 1.691 (thư viện thiết bị), `SDK/script` 276, `SDK/user` 3.159 (apps), còn lại rải rác ở root + test_2_vlm.
- **72 MB mesh .obj** commit thẳng vào git → mỗi clone tải cả lịch sử nặng.
- **`SDK/user/web_control.py`: 2.428 dòng**, trong đó class `WebControlApp` 37 method / ~1.850 dòng (dòng 547–2396) và **một method duy nhất `render_html()` dài 1.006 dòng** (45% file — CSS+HTML+JS trong một f-string).
- **27 chỗ `sys.path.insert` hack**, nhiều chỗ hardcode path tuyệt đối của server `/workspace/6DoF_Grasp/htc/...` (`test_2_vlm/run_pipeline.py:27`, `htc/preview.py:8`) và path Windows local (`_probe/probe_cam.py:14`).
- **Không có** README root, `pyproject.toml`, `requirements.txt`, tests, CI.
- **Trùng lặp**: thư mục `robot model/` (có cả dấu cách trong tên!) trùng `SDK/user/robot_model/` — `model.yaml` + `myarm_m750_full.urdf` **giống hệt từng byte**; hằng số firmware `FW_MIN/FW_MAX` khai báo 2 nơi (`ros_bridge.py:92` và `ductocbatdat.py:14`); cổng serial/baud 2 nơi; render MuJoCo + MJPEG serving viết 2 lần độc lập (`camera.py:42-72,145-204` vs `web_control.py:46-135,521-544`).
- **Code mở camera V4L2 viết 4 lần độc lập**: `vla/primary_camera.py`, `octo_pytorch/octo/robot/cameras.py`, `check_cameras.py`, `ductocbatdat.py:305` — lặp lại cùng CAP_V4L2/MJPG/1280×720/buffersize=1 kèm **cùng những comment bài học đo được**.
- **3 driver điều khiển cùng con tay**: `octo/robot/myarm_m750.py` (pymycobot+SE3), `SDK/program/robot/robot.py` (qua ROS), `ductocbatdat.py` (pymycobot+Pinocchio IK) — chưa kể stack bootstrap SDK lặp 2 lần (`run_pipeline.py:27-38` ≈ `calibrate_jacobian.py:54-60`).
- **Artifact commit vào git**: ảnh jpg chụp robot (`test_1_octo/ket_qua_chay_that/`), CSV/PNG trace, benchmark log, ảnh calibration.
- Tên file không tìm được bằng tìm kiếm: `ductocbatdat.py`, `ktmt/`, `VI_SAO_TAO_LAO.md` — hỗn hợp tiếng Việt không dấu / tiếng Anh.

> ⚠️ **Cảnh báo mất dữ liệu ngay cả trước khi refactor**: working tree local `htc` đang xoá `ktmt/recap.txt` + `plan.txt` + `test_gripper_pose.py` nhưng CHƯA commit. `ktmt/recap.txt` là **bản ghi duy nhất** của project digital twin `simu/` (ngoài git, trên Jetson). Nếu `sync.ps1 push` (dùng `git add -A`) trước khi cứu, tài liệu này biến mất khỏi git vĩnh viễn. Bước đầu tiên: đưa 3 file này vào `docs/` rồi mới push.
> ⚠️ **Bẫy sync.ps1**: script hardcode `LocalRoot = D:\Documents\mujoco\htc` (sync.ps1:21) — chạy nó từ clone khác sẽ lặng lẽ thao tác trên clone khác. Cần parametrize theo `$PSScriptRoot`.
> ⚠️ **Repo tồn tại 3 bản sao**: clone sạch + `htc` (dirty, +1.9GB untracked `mujoco_on_kaggle/`) + Jetson `/workspace/6DoF_Grasp/htc` — và 2 bản local khác nhau từng byte vì line-ending (CRLF vs LF).

### 1.1 Kho dead code (kiểm chứng bằng grep toàn repo — không ai import/gọi)

| Vị trí | Nội dung chết | Bằng chứng |
|---|---|---|
| `SDK/program/model/` toàn bộ | Abstract `Model` + __init__ | 0 importers toàn repo |
| `SDK/script/scene.py` (49–220) | get_objects/set_object_pose/randomize_objects/reset_scene | 0 runtime callers; `FakeRobot.reset_scene` tự viết lại thay vì gọi |
| `camera.py:31-72,170-184,283-414` | ThreadedHTTPServer + CameraStreamHandler + stream/stop_stream/reset_renderer + pixel_to_3d_base (+ position/rotation_matrix/distribution chỉ phục vụ nó) | không được gọi |
| `robot.py:201,376,498` | `distribute` (alias), `set_joint_deg`, `FollowTrajector` (typo alias) | không được gọi |
| `web_control.py:13,180-181` | `import sys` unused, `_send_json_header` no-op | — |
| `pipeline/adapters/sdk_model.py` | ModelPolicy | chỉ mock_run dùng |
| `test_1_octo/octo_pytorch/` | Framework Octo vendored (95 file) + artifacts/duplicate docs | **~80% số file / ~88% dung lượng test_1_octo là chết**; giữ lại: 2 file .md bài học, check_cameras.py, dataset tooling (nếu còn hướng thu data) |

**Hệ quả thực tế**: muốn chạy bất cứ thứ gì phải thuộc lòng path bí mật, thứ tự `pkill`, stack nào còn sống. Đó chính là cảm giác "không làm được cái gì cả".

Điểm mấu chốt: **code lõi không tệ**. `SDK/pipeline/` (Source→Policy→Sink, types frozen dataclass, runner có thread-owner + deadline + NaN check) là OOP sạch. `ductocbatdat.py` chứa kiến thức hardware cực giá trị trong docstring (firmware nuốt lệnh 1/10, servo dừng non, over-current chốt firmware...). Vấn đề là **tổ chức, không phải chất lượng logic**.

---

## 2. Nguyên tắc refactor (quan trọng nhất)

1. **MOVE, không REWRITE.** Mọi dòng logic đã kiểm chứng trên robot thật được di chuyển nguyên vẹn, kèm docstring bài học. Refactor = đổi *nơi code sống*, không đổi *cách code chạy*.
2. **Xoá trước, tách sau.** Phase 1 chỉ xoá rác — không đụng dòng code sống nào. Mỗi phase commit riêng, chạy được ngay sau mỗi commit.
3. **Kiểm chứng bằng số trước & sau.** FK/IK là hàm thuần — test numeric round-trip trước refactor, sau refactor phải ra **cùng kết quả từng bit** (seed đã cố định `default_rng(0)`).
4. **Ponytail/YAGNI**: không thêm framework, không thêm interface chỉ có 1 implementation, không abstract factory. Pattern có sẵn của chính repo (`SDK/pipeline`) là chuẩn để noi.

---

## 3. Cấu trúc đích

```text
6DoF_Grasp/
├── README.md                  # MỚI: repo là gì, lệnh chạy từng việc
├── pyproject.toml             # MỚI: package `m750` + console-scripts
├── .gitattributes             # MỚI: LFS cho *.obj
├── .gitignore                 # SỬA: chặn artifact (jpg/png/csv/log) sinh ra khi chạy
├── assets/robot_m750/         # GỘP "robot model/" + SDK/user/robot_model → 1 bản duy nhất
├── docs/
│   ├── hardware-m750.md       # ← SDK/information.txt (303 dòng spec robot — tài sản)
│   ├── lessons-octo.md        # ← test_1_octo/VI_SAO_TAO_LAO.md + TOI_UU_OCTO.md
│   └── vlm-plan.md            # ← plan.txt (nếu muốn giữ; local đã xoá)
├── src/m750/
│   ├── spec.py                # RobotSpec: cổng, baud, FW limits, GRIP_L, path URDF
│   ├── arm.py                 # MyArmM750: mở/đóng port, đọc góc, write_joints (retry)
│   ├── kinematics.py          # ArmKinematics: Pinocchio FK, limits
│   ├── ik.py                  # IKSolver + IKResult
│   ├── control.py             # ArmController: get/set pose (quy ước URDF)
│   ├── safety.py              # SafetyGate: giới hạn khớp + drop-check đường đi
│   ├── viewpoints.py          # ring_views() — 6 pose GraspNeRF, thuần toán
│   ├── camera.py              # MjpegStream, capture
│   ├── preview.py             # PreviewServer (HTML tách file riêng)
│   ├── pipeline/              # ← SDK/pipeline NGUYÊN VẸN (đảo không phụ thuộc ai — giữ nguyên)
│   └── ros/                   # ← SDK/program: ros_bridge (RosBridge+OffsetCalibration), robot/ (Robot+RealRobot), camera/ (Camera+USBCamera)
├── apps/
│   ├── web_control.py         # ← SDK/user/web_control.py
│   ├── teleop.py              # ← SDK/user/teleop.py
│   ├── fake_robot.py          # ← SDK/user/fake_robot
│   ├── run_camera.py          # ← htc/camera.py
│   ├── run_preview.py         # ← htc/preview.py
│   └── calibrate_*.py         # ← test_2_vlm/scripts/*.py
├── tools/                      # check_cameras.py, benchmark_camera/ (script + kết quả đo)
├── experiments/vlm/           # ← test_2_vlm (vla/, configs/, P*_KET_QUA.md)
└── tests/                     # MỚI: test FK/IK round-trip, spec, viewpoints
```

Vì sao `src/` layout: chống chính lỗi mà `run_web.sh` đã phải khắc phục bằng docs — *"phải dùng `python -m user.web_control` chứ không phải `python user/web_control.py`"*. Cài package một lần (`pip install -e .`), mọi script import `m750.*` từ bất cứ đâu, hết 27 chỗ `sys.path.insert`.

Vì sao gộp model vào `assets/`: 2 bản URDF trùng từng byte, `robot model/` có dấu cách trong tên path (phá script), chỉ 1 consumer mỗi bản. `RobotSpec.urdf_path` trỏ tới `assets/robot_m750/myarm_m750_full.urdf` qua `importlib.resources` — chạy được cả khi cài package.

Vì sao tách `experiments/` khỏi `src/`: code thử nghiệm không cần import vào package chính; nó *import* package chính. Muốn xoá sau này xoá cả thư mục, không sót.

---

## 4. Thiết kế OOP chi tiết — class hoá `ductocbatdat.py`

File này là mục tiêu số 1 (đang phát triển mạnh nhất, và là "god module" procedural). Bản local 769 dòng tách thành 6 class + 3 module thuần:

### 4.1 `RobotSpec` — một nguồn sự thật cho hằng số hardware

```python
@dataclass(frozen=True)
class RobotSpec:
    port: str = "/dev/ttyACM1"
    baudrate: int = 1_000_000     # 115200 mặc định KHÔNG nối được — giữ comment gốc
    camera: str = "/dev/v4l/by-id/usb-046d_..."
    fw_min_deg: tuple = (-165, -80, -100, -160, -90, -180)
    fw_max_deg: tuple = (165, 100, 80, 160, 120, 180)
    grip_l_mm: float = 87.0       # tool0 → tâm gap, lấy từ URDF
    urdf: Path = ...              # assets/robot_m750/myarm_m750_full.urdf
```

`ros_bridge.py` import lại từ đây → **xoá 2 bản FW_MIN/FW_MAX, GRIP_L rải rác**. Đổi hằng số = sửa 1 chỗ.

### 4.2 `MyArmM750` — sở hữu port, context manager thay global

```python
class MyArmM750:
    """Một instance = một kết nối serial. `with` đảm bảo mở/đóng đúng."""
    def __init__(self, spec: RobotSpec): ...
    def __enter__(self) / __exit__(...)          # thay `arm` global + _open()
    def open(self)      # giữ nguyên _held() check — mo chồng không báo lỗi nhưng lệnh hỏng
    def close(self)
    @property
    def q_deg(self) -> tuple | None              # = _q_now() + _ok6() validation
    def power_on(self, wait_s=2.5) -> bool       # giữ docstring "power_on trả -1 kể cả thành công"
    def release_all_servos(self) -> None
    def temperatures(self) -> list | None
    def write_joints(self, q_deg, speed=20, timeout_s=30, tol=1.0, tries=4) -> bool
        # = _move(): retry khi firmware nuốt lệnh, bù khi dừng non — GIỮ NGUYÊN
    def wait_settled(self, timeout_s=30, quiet_s=0.5, start_s=2.0) -> bool
        # = _wait_stop(): KHÔNG dùng is_moving() — docstring lý do giữ nguyên
```

### 4.3 `ArmKinematics` + `IKSolver` — tách khỏi hardware để test được

```python
class ArmKinematics:
    """Pinocchio, nạp URDF đúng 1 lần trong __init__ — thay hack _model.__dict__."""
    def __init__(self, urdf_path: Path): ...
    def fk_tool0(self, q_deg) -> pin.SE3         # = _fk()
    def gripper_mm(self, M: pin.SE3) -> ndarray  # = _grip(), dùng spec.grip_l_mm
    @property
    def limits_deg(self) -> Limits               # lo/hi từ URDF (nguồn chính, mục 7.5 lab)

@dataclass(frozen=True)
class IKResult:
    q_deg: tuple | None
    pos_err_mm: float
    rot_err_deg: float
    @property
    def reached(self) -> bool: ...

class IKSolver:
    """IK 6-DOF least_squares + restart. Không mở port → test offline (docstring gốc)."""
    def __init__(self, kin: ArmKinematics, n_restart: int = 15): ...
    def solve(self, T_target: pin.SE3, q_seed) -> IKResult
```

### 4.4 `SafetyGate` — policy an toàn tách khỏi luồng lệnh

```python
class SafetyGate:
    """2 kiểm tra hiện đang nằm lẫn trong set_gripper_pose(): giới hạn khớp + drop-check."""
    def __init__(self, kin: ArmKinematics, drop_max_mm: float = 30.0): ...
    def check_path(self, q_from, q_to) -> bool   # nội suy 11 điểm, z không tụt quá drop_max
```

### 4.5 `ArmController` — use-case API, giữ nguyên quy ước

```python
class ArmController:
    """Compose: arm + kin + ik + safety. Quy ước pose giữ NGUYÊN vẹn:
    rpy=[0,0,0] = chúc thẳng XUống, mm, hệ URDF — đã kiểm chứng trên tay thật."""
    def gripper_pose(self) -> Pose6 | None                     # = get_gripper_pose()
    def move_gripper_to(self, pose: Pose6, speed=20, ...) -> MoveResult  # = set_gripper_pose()
    def ring_views(self, r=150, theta=30, center=None) -> list[Pose6]    # = views_6() (GraspNeRF IV-B)
    def state(self) -> ArmState                                # = state()
```

### 4.6 `MjpegStream` + `PreviewServer` — hết monkey-patch

```python
class MjpegStream:
    """Camera thread + HTTP. .stop() là METHOD thật (join thread, shutdown, close)
    — thay `srv.stop = lambda: ...` hack. Giữ comment core-dump OpenCV."""
class PreviewServer:
    """Trang preview (HTML tách ra file .html riêng, không nhúng string 138 dòng)."""
```

### 4.7 Bảng mapping đầy đủ (để thực thi không sót)

| Hiện tại | Đích | Chú ý |
|---|---|---|
| `arm`, `_open`, `_held` | `MyArmM750.open/ close/ __enter__` | giữ check port-held |
| `_q_now`, `_ok6` | `MyArmM750.q_deg` | validation -1 giữ |
| `_wait_stop`, `_move` | `write_joints`, `wait_settled` | docstring bài học giữ nguyên |
| `_model` (cache hack) | `ArmKinematics.__init__` | nạp 1 lần |
| `_fk`, `_grip` | `ArmKinematics.fk_tool0/ gripper_mm` | |
| `_solve_ik` | `IKSolver.solve` → `IKResult` | seed rng(0) giữ |
| `set_tcp_pose/ set_gripper_pose` | `ArmController.move_to` + `SafetyGate` | drop-check tách |
| `get_gripper_pose`, `state` | `ArmController.gripper_pose/ state` | |
| `views_6` | `ArmController.ring_views` | thuần toán |
| `capture` | `Camera.capture` | |
| `camera_stream` | `MjpegStream` | stop() chuẩn |
| `_PAGE` + `preview` | `PreviewServer` + file HTML | |

### 4.8 Sửa 4 lỗi OOP cụ thể trong SDK (không đụng logic hardware)

**a) `ros_bridge.py` — globals + import side effect.** `OFFSETS_DEG/_OFFSETS/MODEL_MIN_RAD/MODEL_MAX_RAD` bị mutate qua `global` trong `_apply_offsets_deg` (ros_bridge.py:111-124); import-time chạy file I/O đọc `offsets.json` (dòng 189); singleton ẩn `_bridge`/`get_bridge()`. Hệ quả đã tự ghi nhận trong code: *phải import module chứ không import name* (web_control.py:262-263, calibrate_jacobian.py:60). Sửa: đưa 4 biến thành instance của `class OffsetCalibration` (1 object, 1 trạng thái, load trong `__init__` chứ không phải lúc import).

**b) `robot.py` — class-level mutable state.** `_cmd = np.zeros(6)`, `_cmd_grip = 0.0` là **thuộc tính class** (robot.py:501-502), bị shadow per-instance khi ghi; `ctrl` trả về mặc định dùng chung trước lệnh đầu tiên. Sửa: chuyển 2 biến vào `__init__` — 2 dòng diff, 0 rủi ro logic.

**c) `fake_robot.py` — kế thừa sai hướng (LSP violation).** `FakeRobot(Robot)` là *gương hiển thị* nhưng kế thừa toàn bộ lệnh phát ROS (`change/set_joint/_go/_wait_until_ready/FollowTrajectory`) mà nó không bao giờ được phép chạy — invariant "display không publish" hiện chỉ tồn tại trong docstring, không nằm trong cấu trúc. `web_control` đang lợi dụng sự trùng hợp: `apply_preset` gọi `set_pose` trên real HAY fake, cả hai đều publish `/myarm/command/joint_goal` (web_control.py:1334-1360). Sửa: tách `Robot` thành base không publish + `RealRobot(Robot)` giữ lệnh ROS; `FakeRobot` chỉ giữ phần hiển thị. Đây là thay đổi cấu trúc duy nhất phase 4 cần làm cẩn thận (kiểm chứng bằng: fake không còn method publish).

**d) `script/__init__.py` — coupling ngầm.** Import `script.compressor` (chỉ cần cv2) kéo theo `scene` → `mujoco` (script/__init__.py:9-11). Sửa: xoá import scene khỏi `__init__`, ai cần thì `from script.scene import ...` trực tiếp. 1 dòng diff.

**e) `print` thay logging.** Chỉ `pipeline/runner.py` dùng `logging`. Không bắt buộc sửa ngay — chuẩn hoá dần khi từng file được chuyển vào `src/m750/`.

---

## 5. `web_control.py` (2.428 dòng) — phase sau, chiến lược "mỏng dần"

God class `WebControlApp` (37 methods) đang gộp ≥6 trách nhiệm: session/cookie store, state machine mode theo phiên, teleop keyboard logic, farm render-process MuJoCo (spawn + pipes + threads), lệnh an toàn robot, business logic offset/preset/IK-preview, và 1.006 dòng HTML/CSS/JS trong `render_html()`.

Không tách ngay (rủi ro cao, đang chạy tốt). Tách theo thứ tự lợi nhuận giảm dần — mỗi bước độc lập, commit riêng:

1. **Template ra khỏi Python**: 1.006 dòng `render_html()` → file `static/index.html` + `app.js` + `style.css` thật (serve từ disk). JS lintable được, diff-able được, không re-render mỗi request. `teleop.py` đã đi trước đúng pattern này. *(Bỏ ~1.000 dòng)*
2. **`SessionStore`** class — thay dict juggling sessions/mode/tab/teleop-target + eviction (web_control.py:645-711).
3. **Route table** — dict `path → handler` thay 2 chuỗi if/elif 164/153 dòng trong do_GET/do_POST (:193-356, :357-509).
4. **`RenderService`** — sở hữu worker spawn, pipes, JPEG cache, sync loop (:814-964 + worker fn) và **quan trọng nhất: giấu `_mj_lock` khỏi handler**. Hợp đồng thứ tự khoá `_lock → _mj_lock` (2 lần deadlock ABBA đã ghi trong comment :226-234, :587-597) từ "comment phải nhớ" thành **bất biến cấu trúc** — chỉ RenderService được chạm mj_lock.
5. **`TeleopService`** — target accumulation + clamps + quaternion cố định (:740-812).
6. **`RobotCommandService`** — stop/power/rearm/presets/apply_to_real (:1086-1145).
7. **Đưa toán offset về `ros_bridge`** — logic calibration đang nằm trong route `/api/offsets/save` (web_control.py:428-459) → chuyển cạnh `save_offsets_deg`; handler còn 3 dòng.
8. **Encoder numpy→JSON dùng chung** — thay serializer inline (:241-249).

Kết quả: web_control còn ~300 dòng wiring + 6 module nhỏ, mỗi module test được riêng. Bước 4 quan trọng nhất về an toàn (deadlock đã xảy ra 2 lần thật).

---

## 6. Lộ trình thực thi (4 phase, mỗi phase commit riêng, chạy được ngay)

### Phase 0 — Cứu dữ liệu + commit việc đang dở (30 phút)
- **Trước hết**: chuyển `ktmt/recap.txt` + `plan.txt` + `test_gripper_pose.py` (đang bị xoá ở working tree local) vào `docs/` rồi commit — tránh `git add -A` của sync.ps1 xoá sạch khỏi git.
- Commit +453 dòng ductocbatdat.py đang dở + file mới (run.py, camera.py, preview.py).
- Thêm `mujoco_on_kaggle/`, `*.jpg`, `stop_cam`, `cap_img.jpg` vào .gitignore (tarball + checkpoint 1.9GB không thuộc repo).
- Sửa `sync.ps1:21`: `$LocalRoot = $PSScriptRoot` thay vì hardcode — hết bẫy "chạy ở clone này, tác động clone kia".

### Phase 1 — Dọn kho (1 buổi, chỉ XOÁ/GỜI, không sửa code sống)
- `git rm -r test_1_octo/octo_pytorch/octo/{data,model,utils}/ test_1_octo/octo_pytorch/scripts/{train,finetune}*.py test_1_octo/octo_pytorch/scripts/jax_pt/ test_1_octo/octo_pytorch/scripts/configs/ test_1_octo/octo_pytorch/{setup.py,pyproject.toml,requirements_torch.txt}` — **~95 file framework vendored** (tái tạo được từ repo "đàn anh" octo-pytorch-infer v0_2_5 trên Jetson, 2 file có patch của user đã được ghi chép trong TOI_UU_OCTO.md)
- `git rm -r test_1_octo/ket_qua_chay_that/ test_1_octo/octo_pytorch/artifacts/ test_1_octo/tai_lieu_dan_anh/README.md` (ảnh kết quả + trace + bản README trùng từng byte; số liệu đã nằm trong 2 file .md)
- `git rm -r "robot model/"` (bản trùng; `myarm_m750.xml` ở root còn trỏ `meshdir="meshes/visual/"` không tồn tại — bản hỏng; ductocbatdat.py:20 trỏ sang `SDK/user/robot_model/` — 1 dòng diff)
- **Quyết định để người dùng chọn — nhóm `octo_pytorch/octo/robot/` + script dataset (33 file)**: nếu còn định thu dataset Stage-A thì giữ (MOVE vào `experiments/`); nếu không thì xoá luôn — code nằm trên Jetson của đàn anh, moi lại được. Mặc định đề xuất: **xoá** (không wired vào stack nào đang sống).
- Giữ lại từ test_1_octo: `TOI_UU_OCTO.md`, `VI_SAO_TAO_LAO.md` (→ `docs/lessons-octo.md`), `CHANGELOG.md` + `myarm_octo_pipeline.md` (→ `docs/`), `check_cameras.py` (→ `tools/`), `generate_gaussian_pick_blue_waypoints.py` + `datasets/myarm*.json` (→ `experiments/` nếu chọn hướng giữ dataset)
- Dead code trong SDK (xác minh ở bảng §1.1, an toàn vì 0 callers): `program/model/`, `script/scene.py` (hoặc tách khỏi `__init__`), method chết trong `camera.py`/`robot.py`/`web_control.py`
- Chuyển docs: `SDK/information.txt → docs/hardware-m750.md`; `ktmt/recap.txt → docs/` (Phase 0 đã làm); `benchmark_camera/` giữ nguyên cả script + kết quả (bench.log được recap.txt:206-209 trích dẫn — không phải rác)
- `.gitattributes`: `*.obj filter=lfs diff=lfs merge=lfs` + `git lfs migrate import` (72MB → repo nhẹ)
- Kiểm chứng: `git ls-files | wc -l` giảm ~130-160 file; stack web + ductocbatdat vẫn chạy như cũ (chỉ xoá code không ai gọi).

### Phase 2 — Package hoá (1–2 buổi)
- `pyproject.toml` (package `m750`, deps: pymycobot, pinocchio, scipy, numpy, opencv, mujoco — pin đúng version Jetson)
- `git mv SDK/pipeline src/m750/pipeline`; `git mv SDK/program src/m750/ros` (giữ nguyên nội dung từng file trong phase này — chỉ đổi chỗ + import path)
- Class hoá `ductocbatdat.py` → `src/m750/{spec,arm,kinematics,ik,control,safety,viewpoints,camera,preview}.py` theo bảng mapping §4.7
- **Hợp nhất 4 bản code mở camera V4L2** (primary_camera.py, check_cameras.py, camera_stream, USBCamera): 1 class `V4L2Camera` duy nhất trong `m750/camera.py` giữ đúng thứ tự set CAP_V4L2 → MJPG → size → buffersize=1 kèm comment bài học — các consumer khác nhau chỉ thêm phần riêng của mình
- Entry points: `m750-camera`, `m750-preview`, `m750-teleop`, `m750-state` (thay `python3 run.py` + sys.path hack)
- Cập nhật import cho các consumer ngoài: `test_2_vlm` đang import `pipeline.adapters.*`, `program.robot.robot` qua sys.path hack — chuyển sang `from m750.pipeline...`, `from m750.ros...`; `run_web.sh` giữ `python -m` nhưng trỏ `m750.apps.web_control`
- `tests/test_fk_ik.py`: FK round-trip, IK giải lại pose chính nó (ep<0.05mm), views_6 shape — chạy KHÔNG cần robot
- Kiểm chứng: kết quả IK trước/sau refactor khớp từng bit (seed cố định `default_rng(0)`); chạy `m750-state` trên server đọc đúng góc khớp.
- Xoá `ductocbatdat.py` gốc khi apps đã gọi `from m750 import ...`.

### Phase 3 — Thí nghiệm VLM vào đúng chỗ (1 buổi)
- `git mv test_2_vlm experiments/vlm`; xoá path hack `/workspace/...` trong `run_pipeline.py:27-38` và `calibrate_jacobian.py:54-60` (2 chỗ bootstrap SDK bị lặp) → `from m750.pipeline...`, `from m750.ros...`
- **Commit 2 config kết quả còn thiếu từ server** (đang chỉ tồn tại trên Jetson): `configs/servo_jacobian.json` (run_pipeline cần) + `configs/camera_intrinsics.json` (calibrate_camera sinh ra) — không commit là mất khi server đổi
- Cập nhật README của experiments/vlm (đang ghi "Mới tạo. Chưa có nội dung" — stale), ghi rõ P0–P2 đã verify, P3 chưa có doc kết quả
- `scripts/calibrate_*.py → apps/`; các script này dùng `m750.ros.Robot` thay sys.path
- P0–P3 docs giữ trong `experiments/vlm/` (chúng là log thí nghiệm hợp lệ)
- Lưu ý: `vla/` đã là OOP tốt (class nhỏ, đúng trách nhiệm, error taxonomy rõ, duck-type vào pipeline slot) — **chỉ sửa path, không refactor**

### Phase 4 — web_control mỏng dần (tuỳ chọn, sau này)
Như §5. Không làm nếu chưa cần — nó đang chạy.

**Sau mỗi phase: `sync.ps1 push "refactor phase N: ..."` — server pull và smoke-test ngay.**

---

## 7. Cái gì KHÔNG làm (đừng làm)

- **Không merge 2 stack điều khiển** (ROS `Robot` vs `MyArmM750` direct) thành một interface chung — chúng phục vụ mục đích khác nhau, tranh chấp thiết bị là ràng buộc phần cứng (2 stack không chạy đồng thời theo README test_2_vlm). Gộp khi có nhu cầu thật. Ponytail: interface 2 implementation mà chỉ 1 được dùng cùng lúc = abstraction rỗng.
- **Không rewrite SDK/pipeline** — nó là phần chuẩn OOP nhất repo, giữ nguyên vẹn.
- **Không thêm framework** (ros2 CLI, FastAPI, pydantic, DI container...). http.server + stdlib đang đủ.
- **Không đổi convention pose** (rpy=0 chúc xuống, mm, hệ URDF) — đã kiểm chứng bằng đo thật; refactor chỉ *ghi lại* convention, không *đổi* nó.
- **Không đụng simu/ digital twin** (ngoài git, trên server) trong phase này.

---

## 8. Kết quả mong đợi

| Chỉ số | Trước | Sau |
|---|---|---|
| File track | 175 | ~60 |
| Dòng Python | ~35K | ~8K |
| Clone size | ~80MB+ | <5MB (LFS mesh) |
| File lớn nhất | web_control.py 2.428 dòng (render_html 1.006 dòng) | mỗi file < 500 dòng |
| Chạy state robot | cần nhớ `python3 ductocbatdat.py` + import tay | `m750-state` |
| Chỗ sys.path hack | 27 | 0 |
| Hằng số FW limits | 2 bản | 1 (`RobotSpec`) |
| Dead code | ~30K dòng (vendored + method chết) | 0 (moi ra lại từ upstream/git khi cần) |
| Test | 0 | FK/IK/viewpoints chạy offline |
| Tìm code điều khiển arm | `ductocbatdat.py` (???) | `m750/arm.py` |
| Import name vs module (offsets) | bẫy đã ghi trong comment | không còn — object hoá |

Repo trở thành đúng nghĩa: **một package Python cài được + apps + experiments + docs** — thay vì 5 project trộn trong một root.

**Ba module "xương sống" phải hoạt động y nguyên sau refactor** (kiểm chứng bằng grep import ngoài SDK): `pipeline` core (runner/types/interfaces/adapters trừ sdk_model), `program/robot/robot.py`, `program/ros_bridge.py` — test_2_vlm và run_web.sh đang đứng trên chúng.

---

> **TRẠNG THÁI THỰC THI (2026-10): ĐÃ LÀM XONG.** Phases 0–3 đã commit (xem git
> log: docs cuu tai lieu → don kho 175 file → package m750 → experiments/vlm).
> Điểm khác so với kế hoạch dưới đây (cập nhật thực tế):
> - Model robot nằm trong package (src/m750/model/) thay vì ssets/ ở root —
>   để importlib.resources chạy được cả khi pip install.
> - git lfs migrate KHÔNG chạy (viết lại lịch sử → force-push → 3 clone phải
>   cài lại; rủi ro cao với server đang chạy). Meshes 61MB vẫn trong git history.
>   Làm sau khi cần, có phối hợp.
> - Phase 4 (tách web_control 2.4K dòng) CHƯA làm — cần verify trên robot thật.
>   Kế hoạch 8 bước giữ nguyên ở §5.
> - Thêm 	ools/verify_refactor.py: so khớp FK/IK code cũ (git blob) vs mới —
>   chạy trên server để chứng minh refactor không đổi kết quả tính toán.

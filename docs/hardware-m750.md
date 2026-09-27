THÔNG SỐ myArm M750 — SDK này (htc/SDK)
=======================================

Tài liệu này ghi các con số SDK đang THỰC SỰ dùng, mỗi mục kèm NGUỒN, để khi sửa
thì biết phải sửa ở đâu và chỗ nào lệch theo.

Bản cũ của file này ghi offset [0, +52.2, -63.8, 0, +9.5, 0] khai báo tại
`MyArmHardware.OFFSETS_DEG` trong `robot/hardware.py`. Cả hai đều không còn đúng:
file đó đã bị xoá từ lúc refactor, và bộ số đó không phải giá trị đang chạy. Phần
động học khớp (mục 2) và tọa độ camera (mục 6) của bản cũ thì vẫn đúng, đã kiểm lại.


------------------------------------------------------------------
1. CẤU HÌNH CỐ ĐỊNH
------------------------------------------------------------------

Nguồn: `program/ros_bridge.py` (hằng ở đầu file)

  ROS_DOMAIN_ID   10
  SERIAL_PORT     /dev/ttyACM1     (baud 1000000, driver mở)
  CAMERA_DEVICE   /dev/v4l/by-id/usb-046d_Logitech_Webcam_C925e_3F4C8F2F-video-index0

Ảnh camera đi qua ROS, SDK không mở V4L trực tiếp:

  TOPIC_CAMERA_IMAGE = /myarm/cameras/cam01/image_raw

Hai device này cố định nên hardcode trong file, không truyền qua tham số.


------------------------------------------------------------------
2. SÁU KHỚP — ĐỘNG HỌC
------------------------------------------------------------------

Nguồn: `/robot_description` do `robot_state_publisher` phát lúc chạy (đọc lại bằng
subscriber QoS TRANSIENT_LOCAL). Đây là bản có thẩm quyền.

  Khớp  Tên URDF / ROS 2       Trục   Origin (x, y, z) m        Giới hạn URDF
  ----  ---------------------  -----  ----------------------  -----------------------
  q1    shoulder_pan_joint     +Z     0.0000   0.0000  0.0979  ±165.0°  (±2.8798 rad)
  q2    shoulder_lift_joint    +Y     0.0000   0.0000  0.0760  ±90.0°   (±1.5708 rad)
  q3    elbow_flex_joint       +Y     0.0570   0.0000  0.3030  ±90.0°   (±1.5708 rad)
  q4    forearm_roll_joint     +X     0.0788   0.0000  0.0000  ±160.0°  (±2.7925 rad)
  q5    wrist_flex_joint       +Y     0.2491   0.0000  0.0000  -90.0° .. +120.0°
  q6    wrist_roll_joint       +X     0.0790   0.0000  0.0000  ±180.0°  (±3.1416 rad)

Origin tính từ link trước (chuỗi 6 bậc tự do nối tiếp, trục chuẩn URDF/PoE).

Khớp cố định nối tiếp:

  flange_to_tool0_joint    origin (0.118, 0, 0)
  gripper_mount_joint      origin (0.031, 0, 0)

Gripper — hai khớp tịnh tiến đối xứng:

  left_gripper_joint       origin (0, +0.008, 0)   trục +Z   0.0 .. 0.04 m
  right_gripper_joint      origin (0, -0.008, 0)   trục -Z   0.0 .. 0.04 m

`right` là MIMIC của `left` (multiplier 1.0, offset 0.0), nên chỉ có 7 bậc tự do —
khớp với việc driver publish 7 joint.


------------------------------------------------------------------
3. OFFSET HIỆU CHUẨN PHẦN CỨNG
------------------------------------------------------------------

Nguồn: `OFFSETS_DEG` trong `program/ros_bridge.py` — NGUỒN DUY NHẤT trong code.

  DEFAULT_OFFSETS_DEG = (-10.1, 28.58, -7.42, 2.1, -21.19, 2.54)

Giá trị ĐANG chạy KHÔNG cố định: chỉnh được ngay trên UI (chế độ chỉnh offset),
bấm Lưu thì ghi ra `program/offsets.json`. Lúc import, module đọc file đó ghi đè
lên DEFAULT — dòng cuối ros_bridge.py:

  _apply_offsets_deg(_saved_offsets_deg())

`_apply_offsets_deg()` GÁN LẠI OFFSETS_DEG, _OFFSETS, MODEL_MIN_RAD, MODEL_MAX_RAD.
Vì là gán lại biến, mọi chỗ cần offset phải đọc qua module:

  ros_bridge.OFFSETS_DEG                          # ĐÚNG
  from program.ros_bridge import OFFSETS_DEG      # SAI — giữ bản sao, mãi là số cũ

Quy ước chiều (giống `myarm_m750_robot_arm*.yaml` của lab):

  sim = real - offset

  real = giá trị trên /myarm/state/joint_state;  sim = thứ MuJoCo vẽ ra.

  raw_deg(q_model) = degrees(q) + OFFSETS_DEG       # chỉ để đọc/đối chiếu
  model_rad(q_raw) = q_raw - radians(OFFSETS_DEG)   # đường lệnh/feedback đi qua

Nút Lưu trên UI tính `offset_moi = real - sim`, nên bấm Lưu lần hai không đổi gì
(idempotent) và robot đang vẽ không nhảy. Chặn biên: MAX_ABS_OFFSET_DEG = 180.0.

Calibration đang lưu trên máy này:

  offsets_deg = [-9.5, -13.26, -7.42, -7.3, 30.5, -13.08]

Gốc servo firmware lệch nhiều so với zero kinematic URDF. Các số trên đo bằng cách
đối chiếu render MuJoCo với ảnh chụp robot thật.

CẢNH BÁO: `myarm_m750_robot_arm_acm0.yaml` của lab từng ghi [0, 10, -10, 0, 0, 0].
Bộ số đó KHÔNG được đo, mà giải ngược từ "offset = giới hạn firmware - giới hạn
URDF", và nó làm FK/IK của `myarm_kinematics` sai. Sửa bên nào phải sửa cả bên kia.


------------------------------------------------------------------
4. GIỚI HẠN FIRMWARE — VÌ SAO KHÁC URDF
------------------------------------------------------------------

Nguồn: `FW_MIN_DEG`, `FW_MAX_DEG`, `FW_SAFE_MARGIN_DEG` trong `program/ros_bridge.py`

  FW_MIN_DEG = (-165.0, -80.0, -100.0, -160.0,  -90.0, -180.0)
  FW_MAX_DEG = ( 165.0, 100.0,   80.0,  160.0,  120.0,  180.0)
  FW_SAFE_MARGIN_DEG = 1.5

Firmware KHÁC URDF ở q2 và q3:

  q2 chạy tới +100°   (URDF chỉ cho +90°)
  q3 lùi tới -100°    (URDF chỉ cho -90°)

Phần dư đó là hành trình thật của servo mà URDF không khai báo.

Đổi sang không gian model để so được với `/myarm/state/joint_state`:

  MODEL_MIN/MAX = FW_MIN/MAX ± FW_SAFE_MARGIN - OFFSETS

  min = (-153.40, -107.08, -91.08, -160.60, -67.31, -181.04)°
  max = ( 173.60,   69.92,  85.92,  156.40, 139.69,  175.96)°

Lệch 1.5° là CÓ CHỦ Ý: ép sát đúng bằng giới hạn thì lệnh vẫn được nhận nhưng tay
không nhúc nhích (servo gồng lại chậm cơ). Đã đo trên máy thật: khớp 2 nghỉ ở 106°
trong khi giới hạn 100 — kẹp về 100 thì đứng yên, kẹp về 99 thì chạy.

`Robot.rad_min` / `Robot.rad_max` là GIAO của giới hạn URDF và đường bao firmware
trên. Driver chỉ validate theo URDF, nên một đích nằm trong URDF vẫn có thể vượt
firmware — khi đó firmware huỷ NGUYÊN gói lệnh và robot đứng yên mà không báo lỗi.
Giao này mới là miền thực sự gửi được.


------------------------------------------------------------------
5. GRIPPER — QUY ƯỚC ĐỘ MỞ, VÀ MỘT CHỖ ĐANG LỆCH
------------------------------------------------------------------

SDK dùng ĐỘ MỞ TỔNG giữa hai đầu ngón, đơn vị mét:

  MAX_OPENING_M = 0.08        (nguồn: `program/ros_bridge.py`)

Khác quy ước SDK cũ (tọa độ MỘT ngón, 0 .. 0.0345). Driver publish vị trí một ngón,
nên bridge nhân đôi khi đọc (`_on_gripper`) và chia đôi khi gửi.

CHỖ ĐANG LỆCH — phải đo trên robot thật rồi chốt trước khi đồng bộ:

  Nguồn                                    Hành trình/ngón   Độ mở tổng
  ---------------------------------------  ----------------  -----------
  /robot_description (lab, đang chạy)      0 .. 0.04 m       80 mm
  MAX_OPENING_M trong ros_bridge.py        —                 80 mm
  myarm_m750_mujoco.xml (model MuJoCo)     0 .. 0.0345 m     69 mm
  myarm_m750_full.urdf (bản SDK giữ)       0 .. 0.0345 m     69 mm

SDK ra lệnh tới 80 mm trong khi model mô phỏng chỉ mở được 69 mm, nên phần vượt bị
bão hoà trong sim. Chưa xác định bên nào đúng.

Một điểm liên quan, ghi trong `myarm_m750_mujoco.xml`: TRỤC hai khớp kẹp ở model
MuJoCo đã bị ĐẢO có chủ ý so với URDF (URDF: left +Z, right -Z; model: ngược lại).
Lý do đã đo bằng AABB mesh — với trục của URDF thì tăng q làm hai ngón chạy vào nhau
rồi xuyên qua nhau, nửa dải [0, 0.0161] bất khả thi về hình học; với trục đảo thì
khe hở = 2*q, đơn điệu. File model ghi chú: khi lên lại ktmt phải đối chiếu và xác
nhận chiều mở của kẹp thật.


------------------------------------------------------------------
6. CAMERA GẮN CỔ TAY
------------------------------------------------------------------

Có HAI định nghĩa camera khác nhau, đừng lẫn:

6.1. Chuỗi TF trong mô tả robot của lab

  Nguồn: `myarm_description/config/camera_profiles/logitech_c925e_wrist_v1.measurement.yaml`
         (calibration_id logitech_c925e_wrist_v1_20260602, status CALIBRATED)

  parent_frame          = gripper_base_link
  camera_body_frame     = wrist_camera_link
  camera_optical_frame  = wrist_camera_optical_frame

  Từng chặng (translation_m, rpy_rad):

    gripper_base_link -> logitech_c925e_wrist_mount_link
        [-0.0215, 0.000, -0.0214]     [0, -1.5707963, 0]
    mount_link -> wrist_camera_link
        [0.00209, 0.000, 0.025]       [0, 0, 0]
    wrist_camera_link -> wrist_camera_optical_frame
        [0.04, 0.0, 0.03]             [-1.5707963, 0, -1.5707963]

  Ghép lại, biểu diễn trong `gripper_base_link`:

    wrist_camera_link             t = [-0.04650, 0.00000, -0.01931] m
                                  rpy = [0°, -90°, 0°]
    wrist_camera_optical_frame    t = [-0.07650, 0.00000, +0.02069] m
                                  rpy = [0°, 0°, -90°]

  Khung optical có trục Z nhìn về phía trước. Các số này đã kiểm lại bằng cách ghép
  ma trận transform, khớp đúng từng chữ số.

  LƯU Ý: `/robot_description` đang chạy KHÔNG chứa link camera nào — chuỗi trên chỉ
  được nạp khi build mô tả có gắn sensor (myarm_m750_neugrasp.urdf.xacro). SDK này
  cũng không dùng chuỗi đó.

6.2. Camera trong model mô phỏng của SDK

  Nguồn: `user/robot_model/myarm_m750_mujoco.xml`

    <camera name="wrist_cam" pos="-0.04650 0 0.02069"
            quat="0 0.7071068 -0.7071068 0" fovy="70"/>

  Camera này nằm trong frame `gripper_base_link`. Quy ước trục của MuJoCo: camera
  nhìn theo -z của chính nó, +y là "lên" trong ảnh, +x là "sang phải". Với quat
  trên thì: ảnh-phải = -y_body, ảnh-lên = -x_body, tia nhìn = +z_body.

  Chọn frame `gripper_base_link` vì +z của nó trùng trực tiếp cần kẹp và chỉ phụ
  thuộc wrist_flex/wrist_roll, nên hướng nhìn ổn định.

  Quat cũ "0 0 1 0" cho ảnh-lên = +y_body — SAI: trục y của gripper_base_link nằm
  ngang, nên ảnh bị roll đúng 90° và hai ngón hiện lên trên/dưới thay vì trái/phải.

  Vị trí đặt để hai ngón kẹp nằm ở ĐÁY khung hình, không che vùng làm việc. Đo bằng
  phép chiếu 5 vật thể: (-0.045, 0, 0.020) cho 4/5 vật trong khung, tốt nhất trong
  các phương án đã thử.

  Góc nhìn thứ ba (xem toàn cảnh, không phải camera thật) — `user/robot_model/scene_vla.xml`:

    <camera name="view_cam" pos="0.35 -1.6314 0.8607"
            quat="0.843391 0.5373 0 0" fovy="62"/>

  Đừng nhầm 6.2 với 6.1: cùng trỏ vào frame gripper_base_link nhưng hai định nghĩa
  khác nhau (x của 6.2 trùng x của camera BODY, z của 6.2 trùng z của khung OPTICAL).


------------------------------------------------------------------
7. TRA NHANH — SỬA SỐ THÌ SỬA Ở ĐÂU
------------------------------------------------------------------

  Cần đổi                          Sửa tại
  -------------------------------  --------------------------------------------
  Cổng serial, device camera       program/ros_bridge.py (hằng đầu file)
  Offset hiệu chuẩn                UI -> program/offsets.json (gốc: ros_bridge.py)
  Giới hạn firmware                program/ros_bridge.py: FW_MIN/MAX_DEG
  Độ mở gripper tối đa             program/ros_bridge.py: MAX_OPENING_M
  Động học khớp (nguồn thật)       URDF của lab, qua myarm_robot_driver
  Chuỗi TF camera (lab)            myarm_description/config/camera_profiles/
  Hình dáng + camera trong sim     user/robot_model/*.xml

  Bật/tắt chuyển động thật         <LAB>/.../service/config/services.yaml
                                   (mục 8 — KHÔNG nằm trong SDK)


------------------------------------------------------------------
8. CỔNG BẬT CHUYỂN ĐỘNG THẬT — phía LAB, không nằm trong SDK
------------------------------------------------------------------

Nguồn: <LAB>/myarm_sdk/pycore/src/myarm_sdk/service/config/services.yaml
        <LAB>/myarm_sdk/pycore/src/myarm_sdk/service/robot_arm.py (nhánh chọn
        adapter, đoạn gán accepts_execution_setpoints)

TRIỆU CHỨNG khi cổng đóng — dễ chẩn đoán sai thành lỗi web:

  - Bấm nút trên web, API trả {"status": "ok"}
  - SDK gửi ĐÚNG lệnh lên /myarm/command/joint_goal
  - executor nhận và stream setpoint
  - nhưng tay ĐỨNG YÊN tuyệt đối
  - rồi mọi lệnh sau đó bị từ chối: "joint goal rejected; call reset after holding"

Vì sao có dòng từ chối đó: executor bám quỹ đạo, tay không nhích nên sai số bám
vượt ngưỡng, nó tự chốt HOLDING. HOLDING chỉ thoát bằng
/myarm/motion_execution/reset — nút "🔓 KHÔI PHỤC" gọi đúng service này.

Gốc rễ, trong robot_arm.py:

    accepts_execution_setpoints = accept_internal_setpoints and allow_physical_motion

Cờ tắt thì myarm_robot_driver KHÔNG tạo subscription cho
/myarm/internal/driver_joint_setpoint — setpoint rơi vào khoảng không. Lúc khởi
động driver log đúng một dòng, đây là dấu hiệu nhận biết nhanh nhất:

    "Internal execution setpoints are disabled by robot transport policy."

Hai cờ phải bật trong services.yaml, mục services.robot_arm:

    transport:
      accept_internal_setpoints: true
      allow_physical_motion: true      # mặc định FALSE, phải bật có chủ đích
    gripper:
      allow_physical_actuation: true   # cờ RIÊNG cho kẹp, cũng mặc định FALSE

Comment của chính lab ghi: "Physical motion remains an explicit opt-in." Nghĩa là
mặc định đóng là CỐ Ý, không phải lỗi. Sửa xong phải KHỞI ĐỘNG LẠI stack vì
driver đọc cờ ngay lúc khởi tạo.

CÒN MỘT CỬA NỮA: sau mỗi lần restart stack, driver ở trạng thái `disarmed`, mà
driver chỉ áp setpoint khi đã armed. Phải gọi /myarm/robot/rearm (nút
"🔓 KHÔI PHỤC") TRƯỚC khi gửi lệnh, nếu không lệnh vẫn bị nuốt im lặng.

Quyền điều khiển trên web tự nhả sau 120s không thao tác (LOCK_TIMEOUT_S), nhưng
nếu tab của người giữ quyền còn mở và còn poll thì giữ tới trần
120 * HOLDER_KEEPALIVE_FACTOR = 300s. Mất quyền thì mọi lệnh trả
"Ban chua giu quyen dieu khien" — bấm "✋ Giành quyền điều khiển" để lấy lại.

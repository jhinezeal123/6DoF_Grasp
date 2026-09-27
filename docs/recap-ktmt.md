================================================================================
RECAP - MyArm M750 Digital Twin (MuJoCo)
Cập nhật: khi ktmt OFFLINE (Tailscale: "offline, last seen 10h ago")
================================================================================

MÁY:      ktmt = 100.92.121.52 (Tailscale), user ktmt-agx-xv
MÃ NGUỒN: /workspace/6DoF_Grasp/htc/simu/   (TẤT CẢ nằm trên ktmt)
NGUYÊN TẮC: không viết file lên máy user, trừ connect.ps1


================================================================================
1. MỤC TIÊU BAN ĐẦU (7 bước, theo thứ tự)
================================================================================

Bước 1-3 là nền móng: "sai ở đây thì hỏng hết"

  1. tools/dae2obj.py        - convert COLLADA .dae -> .obj
  2. tools/build_model.py    - URDF -> MJCF
  3. tools/verify_fk.py      - kiểm chứng FK vs Pinocchio
  4. sim/engine.py           - engine mô phỏng
  5. web/server.py + index.html - MJPEG stream + UI
  6. realtime/ + sim/ros_bridge.py + run_sim.py --follow
  7. connect.ps1             - script kết nối từ máy user

  ==> CẢ 7 BƯỚC ĐÃ XONG VÀ ĐÃ KIỂM CHỨNG


================================================================================
2. YÊU CẦU MỞ RỘNG (đã thống nhất, đang làm dở)
================================================================================

  A. Áp dụng hiệu chuẩn (calibration offsets)
  B. Thêm khả năng thay đổi góc nhìn (tịnh tiến / zoom / quay)
  C. Hai mode: control và simulator
     - control   : điều khiển/mirror robot THẬT, read-only, placeholder.
                   Tác dụng duy nhất: chuyển pose hiện tại vào mirror.
     - simulator : thêm API ROS để thử nghiệm điều khiển robot GIẢ LẬP.
                   Chạy driver giả lập, KHÔNG đụng robot thật.
  D. Tab camera trong UI, có toggle on/off, stream ảnh về web
     (mục đích: sau này test model vision)

  ==> A, B, D: XONG VÀ ĐÃ KIỂM CHỨNG
  ==> C: XONG NHƯNG CẦN SỬA LẠI (xem mục 4)


================================================================================
3. TRẠNG THÁI HIỆN TẠI - VIỆC NÀO XONG, VIỆC NÀO CHƯA
================================================================================

[XONG - đã đo/kiểm chứng bằng số]

  [x] Bước 1-7 mục tiêu ban đầu
      - 9 mesh .obj, 61 MB
      - MJCF: 13 bodies, 8 joints, 7 actuators
      - FK vs Pinocchio: sai số 1.422e-16 m
      - engine: sai số xác lập 0.00000 rad

  [x] A. Hiệu chuẩn
      - realtime/calibration.py
      - Công thức lấy từ SDK (pycore/src/adapters/joint_mapping.py):
            core_deg = (firmware_deg - offset_degree) / direction
      - Số đo THẬT từ tay máy xác nhận:
            thô :  84.5   74.5   -23.1  -87.1  -59.1  143.4
            chuẩn: 84.5   64.5   -13.1  -87.1  -59.1  143.4
            lệch :  0.0  -10.0   +10.0    0.0    0.0    0.0
      - Mặc định BẬT; cờ --no-offset để xem số thô

  [x] B. Điều khiển góc nhìn
      - 6 phép biến đổi cho 6 hash ảnh KHÁC NHAU
      - {"auto":true} khôi phục ĐÚNG hash gốc (chứng minh auto tất định)
      - UI: kéo chuột = quay, lăn = zoom, shift/chuột phải = tịnh tiến,
            nháy đúp = về auto

  [x] C. Hai mode (phần chính)
      - mode.py: lock dùng flock, từ chối chạy 2 mode cùng lúc
      - serve.py: --mode control|simulator|--stop|--status
      - ĐÃ ĐO: mode simulator -> fuser /dev/ttyACM0 = "none" (không đụng robot)
      - ĐÃ ĐO: driver log "hardware remains closed"
      - ĐÃ ĐO: gửi trajectory -> sim nhảy tới ĐÚNG pose
               [55.0, -27.5, 48.13, -0.0, -34.38, 20.63]

  [x] D. Tab camera
      - realtime/camera.py, thông số lấy từ benchmark đã đo sẵn (không đoán)
      - Camera thật: Logitech C925e, /dev/video0, ảnh 1280x720, 83 KB
      - Toggle ON -> giữ /dev/video0; OFF -> nhả (đã đo bằng fuser)
      - Mặc định OFF (tránh tranh USB với cổng serial)


[CHƯA XONG - cần làm khi ktmt lên lại]

  [ ] 1. CHẠY LẠI selftest.py  <-- VIỆC ĐẦU TIÊN
         Lần chạy trước: 9/10 pass, chỉ "view control" FAIL.
         Lỗi đó ĐÃ SỬA nhưng CHƯA VERIFY được (mất kết nối giữa chừng).
         Lệnh:
           ssh ktmt 'cd /workspace/6DoF_Grasp/htc/simu && python3 selftest.py'

  [ ] 2. XÓA BRIDGE  <-- TASK MỚI, user yêu cầu
         (chi tiết ở mục 4)

  [ ] 3. Chưa từng chạy thử mode control SAU KHI thêm calibration
         (đã test calibration đọc số, nhưng chưa test mirror đầy đủ vào web)


================================================================================
4. TASK MỚI: XÓA BRIDGE (chưa làm)
================================================================================

LÝ DO - do user chỉ ra, và user ĐÚNG:

  ROS bản chất là công cụ truyền dữ liệu. Nếu đã có topic /joint_states
  thì web server subscribe TRỰC TIẾP là xong. Không cần ai đứng giữa.

  Cách ĐANG LÀM ở mode simulator (THỪA 1 tầng):
      driver -> /joint_states -> ros_bridge -> HTTP POST -> web server -> sim

  Cách ĐÚNG:
      driver -> /joint_states -> web server (subscribe trực tiếp) -> sim

  Đây là lỗi thiết kế của tôi: tôi tái dùng sim/run_sim.py cũ (viết hồi
  CHƯA có web server) mà không kiểm tra ai đang sở hữu engine. Kết quả là
  có 2 bản sim; tôi vá bằng HTTP thay vì sửa gốc.

  LƯU Ý QUAN TRỌNG VỀ PHẠM VI:
  - Mode CONTROL vốn KHÔNG có bridge (chỉ real_arm_reader đọc cáp -> /set).
    Ở mode control hiện tại đã đúng và gọn: tay máy -> reader -> sim.
  - Task này CHỈ áp dụng cho mode SIMULATOR.

CẦN LÀM:
  - XÓA:  sim/ros_bridge.py
  - XÓA:  sim/run_sim.py
  - THÊM: ~60 dòng vào web/server.py
            + subscribe /joint_states  -> cap nhat sim
            + ActionClient gui FollowJointTrajectory cho driver
  - SỬA:  serve.py (bỏ việc spawn bridge)
  - SỬA:  sim/engine.py giữ nguyên (đã có target())
  - GIỮ:  realtime/real_arm_reader.py (mode control vẫn cần)

KẾT QUẢ MONG ĐỢI:
  - mode simulator: 4 tầng -> 3 tầng
  - xóa 2 file
  - PHẢI TEST LẠI mode simulator sau khi sửa


================================================================================
5. LỖI ĐÃ GẶP VÀ CÁCH SỬA (để không mắc lại)
================================================================================

  [ĐÃ SỬA] rclpy không có trên python3 thường
           -> phải source /opt/ros/foxy/setup.bash

  [ĐÃ SỬA] cv2 PHẢI import TRƯỚC mujoco, nếu không:
             ImportError: libgomp.so.1: cannot allocate memory in static TLS block
           -> đã thêm comment "do not reorder" trong web/server.py
           -> LỖI NÀY SẼ LÀM TAB CAMERA HỎNG HOÀN TOÀN nếu đảo thứ tự

  [ĐÃ SỬA] selftest tìm EOI sai: find(b"\xff\xd9", s) từ vị trí header
           trả về frame 2 byte -> phải tìm từ soi+2 và yêu cầu >1000 byte

  [ĐÃ SỬA] selftest đọc MJPEG vô tận bằng urlopen().read() -> treo mãi
           -> đọc có giới hạn rồi close()

  [ĐÃ SỬA] mode.py: nội dung file lock sống lâu hơn flock
           -> current() phải kiểm tra chính flock, không chỉ đọc JSON

  [ĐÃ SỬA] mode.py: assert sai vì tiến trình con thừa hưởng flock
           -> kiểm tra bằng fork() cho đúng cross-process

  [ĐÃ SỬA] camera mặc định quá tối (exposure_auto_priority=1 khoá gain=0)
           đo được: mặc định 18.5/255 | gain 255 -> 106.8 | exp400+gain200 -> 165.3
           -> đặt DEFAULT_GAIN=200, DEFAULT_EXPOSURE=400

  [ĐÃ SỬA] đặt sai tên action -> tên thật:
             /myarm_m750/follow_joint_trajectory  (driver_node.py:150)

  [ĐÃ SỬA] tay máy xuyên sàn ở tư thế thật (đo 401 tư thế: thấp nhất -0.359 m)
           -> thêm MOUNT_Z=0.40, verify_fk trừ đi offset này

  [ĐÃ SỬA] camera nhìn sai chỗ -> ảnh đen
           -> camera tự căn theo tay máy thay vì lookat cố định

  [ĐÃ SỬA] mirror chết khi web server restart -> bỏ qua lỗi, tự kết nối lại


================================================================================
6. SỐ LIỆU ĐO ĐƯỢC (dùng để đối chiếu, không phải suy đoán)
================================================================================

  Tay máy thật:
    - Cổng: /dev/ttyACM0, QinHeng USB Single Serial, ID 1a86_USB_Single_Serial_5B09024867
    - Baud: 1000000
    - Đơn vị góc trên dây: CENTI-DEGREE (raw/100 = độ). raw/10 cho 844 độ -> vô lý
    - Lệnh duy nhất được gửi: 0x20 (GET_ANGLES) - chỉ HỎI, không chuyển động
    - Gói: FE FE 02 20 FA
    - Pose hiện tại (ổn định qua 30 lần đọc):
        84.5  74.5  -23.1  -87.1  -59.1  143.4  (thô)
    - Nhóm dialout: ĐÃ thêm user vào (cần cho quyền cổng serial)

  Tư thế robot thật khi mirror: tay máy CHÚC XUỐNG, đầu tay ở z=-0.09 m

  Mode simulator:
    - Topic: /joint_states (5 Hz), /diagnostics
    - Action: /myarm_m750/follow_joint_trajectory
    - Safety: max_joint_step_rad = 0.08 (gửi 0.5 rad 1 điểm -> ABORTED)
    - Action name: xem driver_node.py:150

  Camera (từ benchmark_camera/benchmark_report.json):
    - MJPG 1280x720@30 (v4l2):  30.1 fps, p99 37 ms, CPU 32%   <-- ĐANG DÙNG
    - MJPG  640x480@30 (v4l2):  30.0 fps, p99 37 ms, CPU 11%
    - YUYV  640x480@30 (v4l2):  30.0 fps, p99 37 ms, CPU 88%  (tốn CPU gấp 3)

  PHÁT HIỆN QUAN TRỌNG:
    robot_mock.yaml và robot_real.yaml GIỐNG HỆT NHAU từng byte.
    => Driver giả lập ĐÃ dùng đúng thông số robot thật.
    => Việc "lấy thông số cho driver giả lập" KHÔNG CẦN LÀM GÌ.


================================================================================
7. KIẾN TRÚC CODE HIỆN TẠI (sau khi xóa bridge sẽ khác - xem mục 4)
================================================================================

  simu/
  ├── mode.py                    [MỚI]  quản lý mode + flock lock
  ├── serve.py                   [MỚI]  entry point: --mode control|simulator
  ├── selftest.py                [SỬA]  10 check
  ├── assets/
  │   ├── myarm_m750.xml         (MJCF, 148 dòng)
  │   └── meshes/*.obj           (9 file, 61 MB)
  ├── tools/
  │   ├── dae2obj.py
  │   ├── build_model.py         (MOUNT_Z=0.40)
  │   ├── verify_fk.py           (trừ mount offset)
  │   └── mesh_stats.py
  ├── sim/
  │   ├── engine.py              [SỬA]  + target()
  │   ├── ros_bridge.py          [SỬA]  <-- SẼ XÓA
  │   └── run_sim.py             [SỬA]  <-- SẼ XÓA
  ├── realtime/
  │   ├── calibration.py         [MỚI]  offset + đổi đơn vị
  │   ├── camera.py              [MỚI]  đọc camera, gain/exposure
  │   ├── real_arm_reader.py     [SỬA]  dùng calibration + mode guard
  │   ├── fake_robot.py
  │   ├── ros_joint_state.py
  │   ├── rate_probe.py
  │   └── stamp_probe.py
  └── web/
      ├── server.py              [SỬA]  + /view, /camera/*, cv2 import trước
      └── index.html             [SỬA]  + 2 tab, điều khiển chuột

  D:\Documents\mujoco\connect.ps1   (file DUY NHẤT trên máy user)


================================================================================
8. LỆNH THƯỜNG DÙNG
================================================================================

  # Kiểm tra toàn bộ (CHẠY ĐẦU TIÊN khi ktmt lên)
  ssh ktmt 'cd /workspace/6DoF_Grasp/htc/simu && python3 selftest.py'

  # Chạy mode control (mirror tay máy thật)
  ssh ktmt 'cd /workspace/6DoF_Grasp/htc/simu && python3 serve.py --mode control'

  # Chạy mode simulator
  ssh ktmt 'cd /workspace/6DoF_Grasp/htc/simu && python3 serve.py --mode simulator'

  # Dừng / xem trạng thái
  ssh ktmt 'cd /workspace/6DoF_Grasp/htc/simu && python3 serve.py --stop'
  ssh ktmt 'cd /workspace/6DoF_Grasp/htc/simu && python3 serve.py --status'

  # UI
  http://100.92.121.52:8080/

  # Kiểm tra riêng calibration
  ssh ktmt 'cd /workspace/6DoF_Grasp/htc/simu && python3 realtime/calibration.py --check'

  # Kiểm tra riêng camera
  ssh ktmt 'cd /workspace/6DoF_Grasp/htc/simu && python3 realtime/camera.py'


================================================================================
9. LƯU Ý KHI LÀM TIẾP
================================================================================

  - TUYỆT ĐỐI không gửi lệnh chuyển động ra tay máy thật.
    real_arm_reader chỉ gửi 1 loại gói: 0x20 (GET_ANGLES) = câu HỎI.
    Không có lệnh chuyển động nào trong code.

  - 2 mode KHÔNG được chạy cùng lúc (đã có lock chặn).

  - Mode simulator PHẢI giữ /dev/ttyACM0 đóng. Đã kiểm chứng bằng fuser.

  - Mass/inertia trong MJCF là GIẢ ĐỊNH (thể tích mesh x 900 kg/m3,
    tổng 1.4027 kg) vì URDF ghi rõ "No inertial data are available".
    => Mô-men trọng lực KHÔNG đáng tin.

  - connect.ps1 vẫn đang khởi động driver giả lập -> cần cập nhật để
    dùng serve.py --mode.

  - SSH tới ktmt luôn in cảnh báo ra stderr (hostfile_replace_entries:
    mkstemp: Permission denied). Đây là NHIỄU, không phải lỗi.
    Vì vậy lệnh scp hay trả exit code 1 một cách giả tạo.

  - Không dùng heredoc trong PowerShell. Cách đang dùng:
    base64 hoá script rồi: ssh ktmt "echo <b64> | base64 -d | bash"

================================================================================

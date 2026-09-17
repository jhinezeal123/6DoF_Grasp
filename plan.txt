PLAN: BIẾN PHYSBRAIN 1.5 THÀNH VLA CHO M750 TRÊN SERVER KTMT
================================================================

Mục tiêu
--------
Tích hợp DeepCybo/PhysBrain1.5 vào pipeline của HTC SDK để nhận:

  ảnh camera + instruction + robot state + action history

và tạo:

  end-effector action chunk -> action adapter -> TCP pose/gripper của M750.

Lượt lập plan này chỉ nghiên cứu/read-only. Không sửa SDK, không tải model và không
gửi lệnh tới robot.

KẾT LUẬN KỸ THUẬT
-----------------
PhysBrain 1.5 đã được huấn luyện để sinh action token, nhưng checkpoint HF hiện chưa
phải VLA plug-and-play cho M750.

Có hai hướng:

1. Native ActionPiece:
   ảnh + instruction + action history -> ActionPiece token -> trajectory.
   Hướng này gần với thiết kế gốc và có thể zero-shot nếu có đủ decoder/codebook,
   nhưng artifact ActionPiece công khai hiện chưa đủ để decode an toàn.

2. Continuous action head:
   dùng PhysBrain 1.5 làm VLM backbone, thêm projector + action head sinh action
   liên tục, rồi fine-tune trên dữ liệu M750. Đây là hướng khả thi để có VLA chạy được
   trên robot.

Khuyến nghị: kiểm tra 2B trước, chạy 8B trên GPU server riêng, và phát triển
continuous action head song song với việc chờ ActionPiece decoder chính thức.

TÀI LIỆU GỐC
------------
Model card:
  https://huggingface.co/DeepCybo/PhysBrain1.5-8B

Checkpoint 2B:
  https://huggingface.co/DeepCybo/PhysBrain1.5-2B

Technical report:
  https://arxiv.org/html/2609.14973

PhysBrain 1.5 repository:
  https://github.com/DeepCybo-PhysAI/PhysBrain-1.5

PhysBrainEvalKit:
  https://github.com/DeepCybo-PhysAI/PhysBrainEvalKit

ActionPiece repository:
  https://github.com/DeepCybo-PhysAI/ActionPiece

PhysBrain 1.0 VLA reference:
  https://github.com/Phys-Brain/PhysBrain-VLA

KIỂM TRA SERVER KTMT
--------------------
Kết quả read-only hiện tại:

- hostname: ktmt-agx-xv
- kiến trúc: Linux aarch64 / Jetson
- Python: 3.8.10
- PyTorch: 2.1.0a0+41361538.nv23.06
- CUDA: khả dụng
- RAM: khoảng 30 GiB tổng, khoảng 25 GiB available
- disk: khoảng 64 GiB còn trống
- transformers: chưa cài
- SDK hiện chưa có artifact PhysBrain hoặc ActionPiece

Model 8B được HF ghi là 9B parameters, BF16; model tree có khoảng 17.8 GB weights.
Không nên coi 30 GiB shared memory của Jetson là đủ cho inference 8B ổn định khi
ROS, camera và robot cũng chạy cùng máy. Không train model trên ktmt.

PhysBrainEvalKit yêu cầu Python 3.10+, Python 3.11 được khuyến nghị. Môi trường model
phải tách khỏi Python 3.8 và runtime ROS hiện tại.

KIẾN TRÚC ĐÍCH
--------------
Mô hình triển khai:

  ktmt Jetson
    - ROS camera
    - robot feedback
    - SDK pipeline
    - safety/watchdog
    - gửi request đến model server

  GPU server
    - PhysBrain1.5-2B hoặc 8B
    - processor/tokenizer
    - action head hoặc ActionPiece decoder
    - policy service

Luồng dữ liệu:

  Camera + Robot state
        -> Source
        -> Remote PhysBrain Policy
        -> ActionChunk Adapter
        -> DryRunSink
        -> RobotSink tùy biến

Không để model process gửi lệnh ROS trực tiếp. SDK giữ quyền kiểm tra deadline,
sequence, giới hạn action và emergency stop.

API dự kiến giữa ktmt và GPU server:

Request:
  {
    "rgb": image,
    "instruction": "...",
    "qpos": [6 values],
    "tcp_pos": [x, y, z],
    "tcp_quat": [qx, qy, qz, qw],
    "gripper": opening_m,
    "previous_action": ...,
    "frequency_hz": ...,
    "embodiment": "M750"
  }

Response:
  {
    "action_format": "eef_relative_10d",
    "horizon": 16,
    "action": [[...], ...],
    "frame": "...",
    "normalization_version": "...",
    "model_version": "...",
    "latency_ms": ...
  }

PHYSBRAIN ACTION REPRESENTATION
--------------------------------
Theo technical report, mỗi wrist dự đoán H=16 bước. Mỗi bước có 10 chiều:

  [dx, dy, dz, rotation_6d[6], gripper]

Trong đó:

- dx, dy, dz là translation tương đối so với pose hiện tại;
- rotation_6d biểu diễn rotation tương đối;
- gripper là giá trị đóng tuyệt đối trong [0, 1];
- một wrist chunk dùng 32 action token;
- ActionPiece dùng vocabulary/codebook 512 token;
- action history trước đó có thể được dùng làm local motion context;
- source native axes, motion scale và frequency được giữ lại, không tự động
  canonicalize thành frame M750.

Các điểm trong một chunk đều có cùng anchor hiện tại:

  p_target[k] = p_current + undo_scale(delta_position[k])
  R_target[k] = R_current @ R_delta[k]

Không cộng delta theo kiểu stepwise.

NATIVE ACTIONPIECE ROUTE
------------------------
Mục tiêu:

  model.generate(...)
      -> action token IDs
      -> ActionPiece decoder
      -> 16 x 10 continuous action
      -> M750 frame
      -> TCP/gripper

Các điều kiện bắt buộc:

1. ActionPiece decoder chính thức.
2. 512-codebook checkpoint hoặc artifact tương ứng.
3. Hàm encode/decode để kiểm tra round-trip.
4. Translation normalization và physical units.
5. Prompt/serialization format cho action task.
6. Quy ước wrist, embodiment và frequency.
7. Mapping native axes sang frame base_link của M750.
8. Calibration gripper và TCP.

Checkpoint HF có các action token, nhưng token ID riêng lẻ không chứa mapping từ
codebook sang trajectory. Public PhysBrain 1.5 repository chỉ chứa tài liệu; public
ActionPiece repository hiện chưa có package decoder/checkpoint codebook hoàn chỉnh.

Vì vậy:

- chưa được tự gán nghĩa cho token ID;
- chưa được gửi token ID trực tiếp xuống robot;
- chưa được gọi là native zero-shot VLA nếu chưa decode được trajectory;
- phải dừng native route nếu không lấy được ActionPiece artifact chính thức.

CONTINUOUS ACTION HEAD ROUTE
----------------------------
Đây là route được khuyến nghị để làm VLA cho M750.

Backbone:

  PhysBrain 1.5 -> image/instruction/state processing
                 -> selected hidden states

Head:

  VLM hidden state
    + proprioception encoder
    + previous action encoder
    -> projector
    -> flow-matching/diffusion/Transformer action head
    -> 16 x 10 continuous action

Khởi đầu nên:

- freeze toàn bộ PhysBrain backbone;
- train projector, proprioception encoder và action head;
- chỉ dùng một bản VLM;
- sau khi head hội tụ mới thử LoRA hoặc partial unfreeze.

Không dùng TwinBrain dual 8B trên Jetson. PhysBrain 1.0 VLA có flow-matching action
head và policy server làm reference, nhưng checkpoint/code của 1.0 không load trực
tiếp vào 1.5.

DỮ LIỆU CẦN THU TỪ M750
-----------------------
Mỗi sample phải đồng bộ:

- RGB image tại thời điểm t;
- instruction;
- timestamp;
- qpos 6 khớp;
- TCP position;
- TCP quaternion;
- gripper opening;
- 16 TCP target states kế tiếp;
- action frequency;
- frame_id và calibration metadata.

Chuyển label về:

  action[t, k] = [
      p[t+k] - p[t],
      rotation6d(R[t].T @ R[t+k]),
      gripper[t+k]
  ]

Cần xác định action log là commanded target hay observed state. Không trộn hai loại
trong cùng dataset.

Giai đoạn đầu:

- một arm;
- một camera;
- một task pick/place;
- một frequency cố định;
- 100-500 demonstrations để kiểm tra pipeline;
- train/validation split theo episode;
- lưu normalization statistics cùng checkpoint head.

PHASE 0 - COMPATIBILITY SPIKE, CHƯA DÙNG ROBOT
------------------------------------------------
Mục tiêu: xác nhận model có thể load trong environment riêng.

Trên GPU server:

1. Dùng Python 3.10 hoặc 3.11.
2. Tạo venv/container riêng cho model.
3. Dùng PyTorch phù hợp CUDA.
4. Cài Transformers version hỗ trợ Qwen3-VL/PhysBrain config.
5. Cài Accelerate, safetensors, Pillow và service dependencies.
6. Load PhysBrain1.5-2B bằng AutoProcessor và AutoModelForMultimodalLM.
7. Chạy một image + instruction.
8. Chạy output_hidden_states=True để kiểm tra feature.
9. Đo load time, inference latency, peak memory.
10. Chỉ sau khi 2B ổn định mới thử 8B.

Trên ktmt:

- giữ Python/Torch ROS hiện tại;
- không nâng cấp global environment;
- chỉ thử 2B hoặc làm client;
- 8B local chỉ là thử nghiệm offline nếu có backend tương thích, không dùng
  cho closed-loop robot nếu latency/memory chưa được chứng minh.

PHASE 1 - POLICY SERVICE
------------------------
Tạo model service trên GPU server:

- model được load một lần và giữ trong VRAM;
- request có sequence number và timestamp;
- bỏ request cũ khi queue bị trễ;
- timeout thì không trả action hợp lệ;
- trả latency và model version;
- health endpoint;
- giới hạn kích thước ảnh;
- logging request/response metadata;
- không để network timeout làm treo vòng ROS.

Giai đoạn đầu dùng service để trả hidden features hoặc raw continuous head output.
Native action token service chỉ bật sau khi có decoder.

PHASE 2 - INTEGRATE VÀO PIPELINE SDK
------------------------------------
SDK pipeline hiện có interface:

  Source -> Policy -> Sink

Các module dự kiến:

1. M750Source
   - đọc camera ROS;
   - đọc qpos;
   - đọc tcp_pos, tcp_quat và gripper;
   - giữ timestamp đồng bộ;
   - thêm instruction và action history.

2. PhysBrainRemotePolicy
   - resize/format RGB;
   - gửi request đến GPU server;
   - giữ connection;
   - nhận chunk;
   - kiểm tra shape và finite values.

3. PhysBrainActionChunkAdapter
   - decode continuous 10D;
   - reconstruct rotation;
   - undo normalization;
   - transform frame;
   - tạo absolute TCP waypoint;
   - map gripper;
   - giữ buffer 16 waypoint;
   - phát từng waypoint theo frequency.

4. DryRunSink
   - lưu action;
   - không gọi robot;
   - dùng cho replay và plotting.

5. M750Sink tùy biến
   - gửi TCP pose và gripper;
   - reject NaN/infinity;
   - reject workspace violation;
   - có hold/stop callback.

Các file SDK liên quan:

  D:\Documents\mujoco\htc\SDK\pipeline\README.md
  D:\Documents\mujoco\htc\SDK\pipeline\adapters\sdk_source.py
  D:\Documents\mujoco\htc\SDK\pipeline\adapters\sdk_model.py
  D:\Documents\mujoco\htc\SDK\pipeline\adapters\sdk_sink.py
  D:\Documents\mujoco\htc\SDK\program\robot\robot.py
  D:\Documents\mujoco\htc\SDK\program\camera\camera.py
  D:\Documents\mujoco\htc\SDK\program\ros_bridge.py

Lưu ý SDK hiện tại:

- CameraRobotSource mặc định chỉ đưa qpos vào state;
- Robot.set_tcp_pose() yêu cầu pose tuyệt đối [x,y,z,qx,qy,qz,qw];
- RobotSink tcp_pose hiện nhận đúng 7 giá trị;
- PhysBrain action chunk là relative 10D và chứa gripper;
- cần custom chunk scheduler và writer;
- Robot facade có tcp_pos nhưng cần xác nhận/expose tcp_quat cho adapter;
- tcp pose của SDK được gửi trong frame base_link.

PHASE 3 - ACTION ADAPTER VÀ CALIBRATION
---------------------------------------
Trên mỗi action chunk:

1. Kiểm tra shape [16, 10].
2. Kiểm tra finite values.
3. Reconstruct rotation matrix từ rotation_6d bằng Gram-Schmidt.
4. Undo translation normalization.
5. Chuyển native frame sang base_link của M750.
6. Tính absolute target TCP.
7. Chuẩn hóa quaternion.
8. Đổi gripper closure về opening_m.
9. Clamp workspace, delta và tốc độ.
10. Chọn waypoint tiếp theo theo receding horizon.

Nếu dùng quy ước PhysBrain:

  gripper = 0 -> mở
  gripper = 1 -> đóng

thì M750 SDK có thể dùng gần đúng:

  opening_m = (1 - gripper) * 0.08

Phải xác nhận thực nghiệm trước khi cho phép robot chạy.

PHASE 4 - TRAIN CONTINUOUS ACTION HEAD
--------------------------------------
Stage 1:

- backbone PhysBrain frozen;
- dùng M750 dataset;
- train action head trên chunk 16 bước;
- dùng loss flow matching hoặc diffusion;
- lưu normalization/config/action schema;
- đánh giá held-out episodes.

Stage 2 nếu cần:

- LoRA trên một phần backbone;
- giữ language/visual capability;
- so sánh latency và memory;
- kiểm tra overfit vào camera/background.

Không huấn luyện native ActionPiece bằng codebook tự tạo rồi coi đó là checkpoint gốc.
Nếu muốn native token training, phải có ActionPiece tokenizer chính thức.

PHASE 5 - OFFLINE VÀ SIMULATION
-------------------------------
Thứ tự:

  recorded image/state
      -> PhysBrain
      -> action head
      -> adapter
      -> DryRunSink

Kiểm tra:

- position error;
- rotation error;
- gripper error;
- chunk boundary continuity;
- predicted trajectory visualization;
- normalization;
- latency;
- stale frame;
- network timeout;
- workspace violation;
- NaN/infinity;
- hướng gripper;
- frame transform.

Sau đó replay action qua MuJoCo trước khi gửi lệnh robot thật.

PHASE 6 - ROBOT THẬT
--------------------
Chỉ bật robot sau khi dry-run và simulation đạt yêu cầu.

Thứ tự:

1. Không có vật thể trong workspace.
2. Tốc độ TCP thấp.
3. Delta mỗi bước nhỏ.
4. Workspace giới hạn chặt.
5. Chỉ thực thi waypoint đầu tiên trong chunk.
6. Có watchdog và timeout mạng.
7. Có emergency stop riêng.
8. Có cơ chế hold khi model lỗi.
9. Ghi image, state, action, latency và safety event.
10. Tăng dần tốc độ/horizon sau khi replay ổn định.

TIÊU CHÍ GO / NO-GO
-------------------
GO khi:

- model load ổn định trong environment riêng;
- inference trả đúng shape;
- mọi giá trị đều finite;
- latency đáp ứng tần số thử nghiệm;
- action adapter được kiểm tra round-trip;
- TCP frame và gripper direction đã xác nhận;
- held-out offline metrics chấp nhận được;
- DryRunSink không có lệnh vượt giới hạn;
- simulation chạy ổn định;
- robot có watchdog, hold và emergency stop.

NO-GO khi:

- không có ActionPiece decoder nhưng vẫn cố chạy native token;
- 8B chạy thiếu memory hoặc latency không ổn định;
- chỉ có CPU nhưng yêu cầu closed-loop realtime;
- chưa biết normalization/action convention;
- chưa có TCP quaternion;
- chưa calibration frame base_link;
- gripper bị đảo;
- frame/state bị stale;
- action có NaN/infinity hoặc nhảy vượt giới hạn;
- chưa qua DryRunSink và simulation.

THỨ TỰ ƯU TIÊN THỰC HIỆN
------------------------
P0. Smoke test PhysBrain1.5-2B, không nối robot.
P1. Dựng model server GPU và client policy trên ktmt.
P2. Kiểm tra hidden state và latency của 8B trên GPU server.
P3. Thu dataset M750 có camera + TCP pose + gripper.
P4. Train continuous 10D action head.
P5. Tích hợp action chunk adapter.
P6. Replay/DryRun/MuJoCo.
P7. Chạy robot thật với giới hạn chặt.
P8. Khi ActionPiece decoder chính thức có sẵn, thử native route và so sánh với head riêng.

ĐỊNH NGHĨA KẾT QUẢ
------------------
"VLA chạy được" nghĩa là hệ thống đã có:

  observation + instruction
      -> policy inference
      -> action chunk
      -> frame/calibration transform
      -> safety validation
      -> robot command

"Zero-shot native PhysBrain 1.5" chỉ được xác nhận khi action token được decode bằng
ActionPiece artifact chính thức và trajectory sau decode đã được calibration cho M750.

Continuous action head cần fine-tune trên dữ liệu M750; không được gọi là zero-shot.



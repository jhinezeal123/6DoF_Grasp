# Pipeline VLA

`pipeline` là một lớp ghép tùy chọn nằm trong SDK. Nó không thay đổi
`program/`, `user/` hoặc vòng đời ROS hiện có. Một thử nghiệm chọn ba module:

```text
Source → preprocess → Policy → postprocess → guard → Sink
```

`Source` trả `Observation`, `Policy` trả `Action`, còn `Sink` nhận lệnh. Mỗi
module có `open()`, `reset()` và `close()` mặc định no-op nên có thể thay riêng
từng phần. `Pipeline` kiểm tra deadline của observation, sequence tăng dần,
ActionSpec, NaN/infinity và giới hạn tần số vòng lặp.

## Chạy thử offline

Từ thư mục `htc/SDK`:

```bash
python -m pipeline.examples.mock_run
```

## Nối model SDK

`ModelPolicy` bọc object có `inference(input)` trong
`program/model/base.py`. Payload mặc định là mapping gồm `images`, `state`,
`stamps_ns`, `instruction` và `observation`. Model có input/output riêng thì
truyền `input_builder` và `output_parser`.

```python
spec = ActionSpec(
    "joint_position", tuple(robot.JOINT_NAMES), ("rad",) * 6, "joint"
)
policy = ModelPolicy(
    model,
    spec,
    input_builder=lambda obs, text: {"rgb": obs.images["front"], "prompt": text},
    output_parser=lambda values: Action(tuple(values), spec),
)
```

## Nối Camera và Robot hiện có

`CameraRobotSource(camera, robot)` đọc `camera.photo()` và `robot.qpos`. Nó chờ
feedback khớp hữu hạn và tạo sequence mới cho mỗi snapshot. Timestamp trong
adapter này là thời điểm đọc cache; nếu thử nghiệm cần đồng bộ ROS chính xác
giữa nhiều sensor, hãy viết Source riêng dùng bundle đã đồng bộ từ callback ROS.

`RobotSink` hỗ trợ sẵn các mode `joint_position`, `tcp_pose` và `gripper`. Với
mode hoặc controller khác, truyền `write_callback`. `halt_callback` mặc định
không gọi `Robot.stop()` vì đó là dừng khẩn cấp có chốt lỗi; ứng dụng phải chọn
rõ hành vi cancel/hold phù hợp controller.

```python
source = CameraRobotSource(camera, robot)
sink = RobotSink(robot, spec, halt_callback=lambda r, reason: None)
pipeline = Pipeline(source=source, policy=policy, sink=sink, rate_hz=5)

with pipeline:
    pipeline.reset("nhặt khối đỏ")
    result = pipeline.step()
```

Để chạy an toàn trước, thay `RobotSink` bằng `DryRunSink(spec)`; các action sẽ
được lưu ở `sink.actions` để kiểm tra.

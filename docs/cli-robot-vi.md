# Điều khiển pymycobot từ root

Sau `bash scripts/setup-env.sh`, dùng `./robot`. Wrapper luôn chọn Python của
6DoF. Có thể dùng `.venv/bin/python robot.py` hoặc console command `m750-robot`
sau khi cài/cập nhật editable package. Không cần tìm file sâu trong `src/`.

```bash
./robot --help
./robot state
./robot angles
./robot power-on
./robot --speed 15 joint 3 10
./robot --speed 15 joints 0 0 0 0 0 0
./robot gripper 30
./robot gripper 50 --unit percent
./robot tcp 300 0 250 0 0 0
./robot stop
./robot power-off
```

Các lệnh joint/gripper/tcp gửi chuyển động thật. `power-off` tắt servo, cần đỡ
tay để tránh rơi. Ví dụ tọa độ không bảo đảm reachable cho mọi cấu hình robot.

| Input CLI | Ý nghĩa |
| --- | --- |
| `joints Q1 ... Q6` | Sáu góc tuyệt đối, độ; kiểm tra giới hạn firmware có margin |
| `joint ID DEG` | ID 1..6; đọc feedback và giữ năm joint còn lại |
| `gripper VALUE` | Tổng độ mở hai ngón, mm; mặc định tối đa 69 mm |
| `gripper VALUE --unit percent` | Vendor scale 0..100, đổi qua độ mở thực của driver |
| `tcp X Y Z RX RY RZ` | Pose tool0 trong base/URDF, mm + Euler XYZ độ; IK của repo |
| `state` | Snapshot contract: joint radian, TCP mét/quaternion xyzw, gripper mét |
| `angles` | Chỉ in góc joint theo độ |

Option toàn cục đứng trước lệnh: `--port /dev/ttyACM1 --baud 1000000`,
`--speed 1..100`, `--timeout SECONDS`. Có thể chỉ định interpreter bằng
`M750_PYTHON=/path/to/python ./robot state`.

Input không hữu hạn, ngoài range hoặc sai số phần tử bị chặn **trước khi tạo
driver**. CLI không tự bật nguồn để chạy một lệnh move. Exit code: 0 thành công,
1 thất bại của driver/runtime, 2 input lỗi. Driver được close khi command kết thúc.
Ctrl-C thử gửi stop nếu backend có capability này, luôn thử close driver và
trả code 130. Nếu stop hoặc close ném exception, CLI ghi lỗi ra stderr mà
không che mất Ctrl-C. Lệnh thường trả code 1 nếu cleanup thất bại.
Lệnh nhận/gửi đều đi qua `RobotControl`/capabilities và adapter pymycobot;
không nhân bản logic retry, IK, conversion hay truy cập firmware Cartesian.

PR này chỉ thêm CLI trên nền refactor; không sửa solver/calibration/driver.
Test dùng fake driver, không mở serial hoặc thử chuyển động trên robot thật.

## Giới hạn của stop và smoke trên phần cứng

`./robot stop` là lệnh vendor gửi qua **cùng serial port**, không phải hardware
E-stop hoặc kênh dừng độc lập. Nếu process khác đang giữ port, `MyArmM750.open()`
từ chối mở và CLI trả lỗi; lệnh này không dừng được robot trong tình huống đó.
CLI không kill process, giành port hoặc bỏ guard. Ctrl-C chỉ gửi stop best-effort
qua driver của chính process đang bị ngắt; serial hỏng thì stop cũng có thể lỗi.

Khi nghiệm thu trên robot, kiểm riêng từng lệnh trong tư thế được đỡ và có
hardware E-stop sẵn. Chỉ một process sở hữu serial tại mỗi thời điểm:

1. `./robot state`: kiểm feedback và trạng thái kết nối.
2. `./robot power-on`, rồi `./robot state`: kiểm servo bật bằng feedback/thực tế.
3. `./robot stop`: kiểm phản ứng thực của firmware, gồm khi đang chạy một chuyển
   động thử trong chính process điều khiển qua Ctrl-C. CLI riêng sẽ bị chặn nếu
   process điều khiển còn sở hữu port; kiểm và ghi nhận lỗi đó, không bypass.
4. Đỡ tay rồi `./robot power-off`, sau đó `./robot state`: kiểm servo đã tắt.

Ghi model/firmware, version pymycobot, port, phản hồi và quan sát chuyển động.
`ok: true` cho stop chỉ là kết quả adapter hiện có; không chứng nhận chức năng
E-stop, thời gian dừng hay loại bỏ mọi chuyển động. Các test giả lập của PR chỉ
chứng minh routing, error handling và cleanup; hardware smoke chưa được chạy.

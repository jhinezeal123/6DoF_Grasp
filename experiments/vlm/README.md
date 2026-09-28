# experiments/vlm — PhysBrain + servo ảnh

Thí nghiệm 2: dùng VLM (vision-language model) thay cho Octo.
Source → `m750.pipeline` (đã `pip install -e .` ở repo root).

## Vì sao dừng hướng Octo — kết luận test 1

Chi tiết đầy đủ ở `docs/lessons-octo-vi-sao-tao-lao.md`. Ba lý do:

1. **Checkpoint finetune bị mù ngôn ngữ.** Đổi câu lệnh từ `"pick up the blue block"`
   sang `"fly the rocket to mars immediately"` chỉ làm action đổi **0,002%** (2,15e-05).
   Model pretrained đổi 5,05e-03 — tức finetune yếu hơn **235 lần**. Nguyên nhân: dataset
   train chỉ có **một câu lệnh duy nhất**, nên gradient tỉa sạch đường ngôn ngữ.
   Không thể bảo nó gắp cái gì.
2. **Policy sập về điểm bất động.** Sau ~10 bước, chính policy ra lệnh đứng yên
   (target đóng băng: 108/119 bước dịch < 1 mm). Tay không nhúc nhích → ảnh không đổi
   → model lại ra lệnh đứng yên. Điểm bất động tự nhất quán, không tự thoát ra được.
3. **Bám vị trí thiếu 13 mm ở trục z** (target z=0.0967, measured z=0.0834). Lỗi điều
   khiển, độc lập với model, và vẫn còn nguyên.

## Cái gì mang sang từ test 1 (đã kiểm chứng, dùng lại được)

| hạng mục | giá trị đúng |
|---|---|
| Camera primary | `/dev/video2` — SPCA2650, index **2**, 1280x720 MJPG @30 |
| Camera wrist | `/dev/video0` — Logitech C925e, index **0**, 1280x720 MJPG @10 |
| Cổng serial | `/dev/ttyACM1` (⚠ `/dev/ttyACM0` **không tồn tại** trên máy này) |
| Mở camera OpenCV | phải đặt `api_preference = 200` (`CAP_V4L2`); mặc định `0` = `CAP_ANY` sẽ khiến OpenCV chọn GStreamer và chết |
| Sau khi connect robot | phải gọi `configure_for_policy()` ngay, **trước** mọi lệnh chuyển động |
| Home pose | `[0.3, 0.0, 0.1, 0.0, 3.14, 0.0]` (m, rad) |

**Stack web (m750/webui) và pipeline AI không chạy cùng lúc** — cả hai giành
`/dev/video0`, `/dev/video2` và `/dev/ttyACM1`. Phải `pkill` trước, chạy xong bật lại:

```bash
pkill -9 -f run_web.sh; pkill -9 -f web_control; pkill -9 -f "ros2 launch"
pkill -9 -f myarm_robot_driver_node; pkill -9 -f myarm_motion_execution; pkill -9 -f kinematics_node
sleep 6
# ... chay thi nghiem ...
cd /workspace/6DoF_Grasp/htc && setsid nohup bash run_web.sh > /tmp/myarm_ui.log 2>&1 < /dev/null &
```

## Trạng thái (cập nhật sau refactor 2026-10)

- **P0–P2: đã làm, đã đo** — xem `P0_KET_QUA.md`, `P1_KET_QUA.md`, `P2_KET_QUA.md`
  (build llama.cpp CUDA + PhysBrain1.5-8B, grounding, hiệu chuẩn Jacobian).
- **P3 (run_pipeline.py): đã ghép, chưa có doc kết quả e2e** — chạy với `--dry-run`
  trước, xem README root để biết lệnh.
- Hai config kết quả **chưa commit** (đang chỉ tồn tại trên Jetson, mục
  `configs/`): `servo_jacobian.json` (run_pipeline cần), `camera_intrinsics.json`
  (calibrate_camera sinh). Copy từ server khi cần khôi phục.
- Hướng chiến lược tiếp theo xem `docs/plan-vla.md` (PhysBrain làm backbone +
  action head liên tục).

## Cấu trúc

```
vla/                 # Policy + guard (cắm vào m750.pipeline), client PhysBrain,
                     # servo ảnh, OnlineJacobian (Broyden), camera primary
scripts/             # calibrate_camera / calibrate_jacobian / check_target
                     # + serve_llamacpp (build + serve model trên Jetson)
configs/             # grasp_myarm_m750.json (P3 runtime config)
run_pipeline.py      # P3: lắp Source→Policy→guard→Sink
P0..P2_KET_QUA.md    # kết quả đo được
```

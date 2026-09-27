# Vì sao model chạy tào lao — chuỗi bằng chứng

Ngày 17/9. Prompt dùng: `"pick the yellow ball"`. Checkpoint:
`octo_weights/finetuned_pytorch/experiment_20260420_024612`, backend robot thật,
`run_octo_robot_eval.py`.

## 1. Hiện tượng đo được

120/120 bước, 132 s, `infer ≈ 215 ms`/bước, kết thúc `reset_to_home()` sạch.

```
step    x       y       z      rz(deg)  grip   err(mm)
   0   0.290   0.000   0.088      0.2   1.00    30.3
  10   0.239   0.133   0.056     30.9   0.00    21.6
  20   0.202   0.207   0.050     44.7   1.00    21.8
  30   0.150   0.286   0.010     74.7   1.00    14.5   <- da ket clamp
 119   0.150   0.269   0.010     72.7   0.00    11.9   <- van y nguyen
```

| chỉ số | giá trị |
|---|---|
| 83/119 bước dịch chuyển < 1 mm | 90 giây cuối (75% episode) đứng yên |
| đường đi 0,393 m / dịch chuyển thuần 0,312 m | đi thẳng tới góc clamp rồi dừng |
| kẹp đổi trạng thái 11 lần (bước 9,17,31,32,33,43,47,51,84,87,93) | rung, không phải trình tự gắp |
| `rz` 0,2° → 72,4° rồi đứng | quay một mạch rồi bỏ |
| `trans_err` 9,9–31,8 mm (trung vị 13,4) | chưa bao giờ bám tốt |

Quỹ đạo đi một mạch −x, +y, −z, +rz cho tới khi đụng giới hạn — dấu hiệu
**action gần như hằng số**, tức policy chạy như vòng hở.

Ảnh wrist lúc kết thúc trông như đang ở sát quả bóng, nhưng đó **chỉ vì khối
`finally` gọi `reset_to_home()`**, không phải vì policy đi tới đó.

## 2. Loại trừ: không phải lỗi bản tối ưu

Chạy cùng ảnh, cùng câu lệnh qua code gốc chưa sửa và code đã tối ưu:

```
goc : [1.9854e-02, -8.1845e-03, -4.2801e-03, 0, 0, -1.0791e-01, 1.0030e+00]
toi : [1.9854e-02, -8.1845e-03, -4.2801e-03, 0, 0, -1.0791e-01, 1.0030e+00]
```

Giống tới từng chữ số. Các tối ưu (cache tokenizer, hoist mask) **không** làm hỏng model.

## 3. Nguyên nhân gốc: ngôn ngữ đã bị vô hiệu hoá

`scripts/diagnose_language.py` — nạp model một lần, hai câu lệnh khác hẳn nhau,
cùng seed, cùng ảnh:

```
input_ids A "pick up the blue block"           : [1432, 95, 8, 1692, 2463, 1, 0, ...]
input_ids B "fly the rocket to mars immediately": [3971, 8, 15721, 12, 8113, 2017, 1, 0, ...]
pad_mask_dict/language_instruction = 1.0        <- token DUOC attention
```

| checkpoint | tỉ lệ action đổi khi đổi câu lệnh |
|---|---|
| `octo-base-1.5-torch` (pretrained) | **5,05e-03** → ngôn ngữ CÓ tác dụng |
| `experiment_20260420_024612` (finetune) | **2,15e-05** → VÔ HIỆU, yếu hơn **235 lần** |

Token được tokenize đúng, mask hợp lệ, vẫn vào transformer — nhưng **không ảnh hưởng gì**.

### Vì sao

Toàn bộ dataset train chỉ có **một câu lệnh duy nhất**:

```
"instruction": "pick up the blue block"
```

Đầu vào ngôn ngữ không mang thông tin nào, nên finetune fit được dữ liệu hoàn hảo
trong khi bỏ qua nó, và gradient tỉa sạch đường ngôn ngữ. Bản port torch **không có lỗi**
(model pretrained vẫn phản ứng với câu lệnh).

Hệ quả: checkpoint finetune là **policy đơn nhiệm, mù ngôn ngữ**. Hỏi nó câu khác
là chuyện vô nghĩa — nó không nghe thấy.

## 4. Sai entrypoint (bài học phụ)

`log.txt` 100 bước của đàn anh — cái có action biến thiên đẹp — do
`runner_realtime_camera.py` sinh ra. Script đó **chỉ mở camera, gọi model, in action**;
không hề kết nối tay máy. Nên **không có bằng chứng nào cho thấy rollout trên robot thật
từng chạy được**. CHANGELOG v0.2.x toàn thêm dụng cụ chẩn đoán cho rollout
(`control_period_s`, resync, `hardware_read_settle_s`, A/B `measured_compose`/`target_only`)
— dấu hiệu của một vấn đề chưa giải quyết.

## 5. Hai lỗi phải sửa mới chạy nổi (đã sửa)

| lỗi | hậu quả |
|---|---|
| `api_preference` mặc định `0` = `CAP_ANY` | OpenCV chọn GStreamer → `Failed to open camera 'primary' on index 2`. Đặt `200` (`CAP_V4L2`) |
| `configure_for_policy()` chỉ gọi trong `finally` | cả rollout chạy với `fresh_mode`/`movement_type`/`end_type`/`gripper_enabled` tuỳ tiện, và `reset_to_home()` ở `env.reset()` xảy ra **trước** khi cấu hình |

## 6. Hướng đi tiếp

Model pretrained **có** phản ứng với ngôn ngữ, và bản `octo-base-1.5-torch` trong thư mục
đàn anh **đã được thêm sẵn thống kê `myarm_stage_a_dataset`** (27 key, gồm cả key này).
Đó là con đường zero-shot đúng:

```bash
python scripts/run_octo_robot_eval.py \
  --config scripts/robot_configs/myarm_m750_primary_wrist_720p_mjpg.json \
  --checkpoint_dir /home/ktmt-agx-xv/Data/khoanhd/Octo_Lab/octo_weights/pretrained_pytorch/octo-base-1.5-torch \
  --dataset_key myarm_stage_a_dataset \
  --text "pick the yellow ball"
```

Muốn finetune nghe được câu lệnh thì dataset **phải có nhiều câu lệnh khác nhau**
cho cùng một cảnh, nếu không thì mọi lần train lại đều lặp lại lỗi này.

# P0 — Build llama.cpp + nạp PhysBrain1.5-8B

Ngày chạy: 17/09/2026. Máy: Jetson AGX Xavier (`ktmt-agx-xv@192.168.6.173`).

## Cổng chặn P0: "Server trả lời 1 câu văn bản" — ĐẠT

---

## 1. Việc đã làm

| | kết quả |
|---|---|
| Sửa ghi chú sai về pymycobot | `SDK/program/robot/robot.py`, `SDK/program/robot/__init__.py` |
| Build llama.cpp bật CUDA | `KET QUA: CUDA BUILD OK`, 60 phút (15:26 → 16:26) |
| Tải weights | `PhysBrain1.5-8B.i1-Q4_K_M.gguf` — 5.122.456.928 byte (4,77 GiB) |
| Tải mmproj | `PhysBrain1.5-8B.mmproj-f16.gguf` — 1.159.030.304 byte (1,08 GiB) |
| Chạy llama-server | cổng **8081** (8080 là web UI của SDK) |

## 2. Bẫy cmake — tốn 5 phút, đã ghi vào script

Lượt build CUDA đầu tiên hỏng, trông y như lỗi CUDA:

```
CMake Error at ggml/src/ggml-cuda/CMakeLists.txt:1 (cmake_minimum_required):
  CMake 3.18 or higher is required.  You are running version 3.16.3
```

**Không phải lỗi CUDA.** `llama.cpp/CMakeLists.txt` gốc ghi cần **3.14**, nên cmake 3.16.3
của JetPack trông đủ. Nhưng file con `ggml/src/ggml-cuda/CMakeLists.txt` cần **3.18**, và nó
**chỉ được đọc khi `GGML_CUDA=ON`** — nên phần CPU build bình thường, chỉ CUDA mới lộ.

Xử lý: tải cmake 3.31.6 về `$BASE/tools/` (không cần sudo). `CUDA 11.4.315` và `sm_72`
hoàn toàn không có vấn đề gì.

## 3. Kiểm chứng CUDA

```
$ llama-server --list-devices
Available devices:
  CUDA0: Xavier (30991 MiB, 17323 MiB free)
```

`libggml-cuda.so.0.24.0` = 143 MB.

Bằng chứng GPU thật sự chạy, đo trong lúc sinh 128 token:

```
GR3D_FREQ:  99% x 11 mau,  85% x 1,  80% x 1     (trong 13,77 s)
nhiet GPU:  35C -> 36C
```

Ghi chú phương pháp: lần đo đầu bằng một request 16 token cho `GR3D_FREQ 0%` — **kết luận
sai**. `GR3D_FREQ` là mẫu tức thời mỗi ~1 giây, suy luận 0,59 s lọt giữa hai mẫu. Phải
sinh đủ dài mới đo được.

## 4. Kiểm chứng vision — model nhìn được thật

Gửi `03_dataset_dan_anh_primary.jpg`, hỏi mô tả:

> "The image shows a robotic arm with red-tipped fingers positioned above a blue
> cylindrical object on a wooden surface."

Đúng tay robot, đúng ngón kẹp đỏ, đúng vật xanh.

Log xác nhận mmproj đã nạp:
```
load_model: loaded multimodal model, '.../PhysBrain1.5-8B.mmproj-f16.gguf'
```

## 5. Số đo độ trễ — VƯỢT CỔNG P1

Cùng một request ảnh 720p:

```
prompt_tokens:      1100        (anh + cau hoi)
prompt_ms:          19138       <- PREFILL
prompt_per_second:  57,5 tok/s
completion_tokens:  23
predicted_per_second: 8,7 tok/s (decode)
TONG:               22,27 s     ->  VUOT nguong 20 s
```

Prefill chiếm **19,1 / 22,3 s**. Nút thắt là xử lý ảnh, không phải sinh chữ.

### Flash attention không phải lever — đã đo

Nghi FA chưa bật nên prefill chậm. Bật `-fa on`, chạy lại y hệt:

| | prompt_ms |
|---|---|
| không `-fa` | 19138 |
| có `-fa on` | 19732 |

Không cải thiện (chênh trong nhiễu).

### Đây là giới hạn phần cứng, không phải cấu hình sai

8B × 2 FLOP/tham số × 1100 token ≈ **17,6 TFLOP**. Xavier có 512 nhân CUDA @ ~1,1 GHz
≈ 1,4 TFLOPS FP32 đỉnh; thực tế đạt được thấp hơn nhiều vì phải giải lượng tử hoá Q4.
19 giây **nằm sát trần phần cứng**. Không có cờ cấu hình nào cứu được.

## 6. `--image-min-tokens 1024` là BẮT BUỘC

llama.cpp cảnh báo lúc nạp model:

```
Qwen-VL models require at minimum 1024 image tokens to function correctly
on grounding tasks
if you encounter problems with accuracy, try adding --image-min-tokens 1024
(ggml-org/llama.cpp issue 16842)
```

Test 2 cần chính xác **grounding** (model trả `point_2d`) → cờ này là điều kiện bắt buộc.
Thiếu nó thì model vẫn chạy, vẫn trả lời, nhưng toạ độ trả về sẽ lệch — **sai im lặng**,
đúng loại lỗi khó phát hiện nhất.

Hệ quả: **hạ độ phân giải ảnh không giảm được token**, vì cờ này áp sàn 1024.

## 7. Phát hiện mở ra hướng thiết kế mới: ngón kẹp đỏ bắt được bằng màu

Chạy thử phát hiện đỏ trên một khung primary thật (`episode_000002/frames/image_primary/000000.jpg`,
1280×720):

```
so blob do >=40px: 2
   dien tich 2664 px   tam (741, 267)  = 57.9% x 37.1% anh
   dien tich 2252 px   tam (609, 220)  = 47.6% x 30.5% anh
tong dien tich do: 0.53% anh
```

Đúng **2 blob** — hai ngón kẹp. Sạch, tách biệt rõ, không lẫn nhiễu.

Nghĩa là `gripper_point_2d` **không cần model**. Chỉ `target_point_2d` mới cần grounding
— và vật thì đứng yên.

Chi phí: vài mili giây, thay vì 19 giây.

## 8. Đề xuất cho P1

| phương án | độ trễ mỗi bước | đánh giá |
|---|---|---|
| Gọi model mỗi bước (thiết kế cũ) | ~22 s | 25 bước ≈ 9 phút |
| **Gọi model 1 lần/vật + bắt ngón kẹp bằng màu** | ~1 s | **25 bước ≈ 20 s + 25 s ≈ 45 s** |
| Lùi về PhysBrain1.5-2B | ~5 s | mất chất lượng grounding, chưa đo |
| Giảm độ phân giải ảnh | không giúp | cờ `--image-min-tokens` áp sàn 1024 |

Đề xuất: **phương án 2**. Nó dùng model đúng chỗ model giỏi (grounding ngữ nghĩa: "khối
xanh" là chỗ nào) và để thị giác cổ điển lo việc nó giỏi (bám một vật thể có màu xác định
trong mọi khung hình). Vẫn gọi lại model định kỳ để hỏi `can_grasp`.

## 9. Việc chưa làm

- Chưa đo chất lượng `point_2d` (đó là P1)
- Chưa so sánh với Space demo của tác giả
- Chưa thử mmproj `Q8_0`
- Server repo vẫn kẹt ở `d6da21e`, commit `a4a242c` mới chỉ lên GitHub

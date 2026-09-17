# Tối ưu Octo trên Jetson AGX Xavier — 4 chỗ

Máy: `ktmt` (Jetson AGX Xavier, L4T R35.6.4, CUDA 11.4, torch `2.1.0a0+41361538.nv23.06`,
device name `Xavier`). Venv của đàn anh: `Octo_Lab/Octo_env`.

## 0. Vì sao phải copy code

`/home/ktmt-agx-xv/Data/khoanhd/Octo_Lab/octo-pytorch-infer_v0_2_5/` **không nằm trong git**
(`fatal: not a git repository`) — sửa thẳng vào đó là không thể hoàn tác. Nên toàn bộ
`octo/`, `scripts/`, `assets/` đã được copy sang `htc/test_1_octo/octo_pytorch/` (1,6 MB,
125 file .py) và mọi thay đổi dưới đây nằm trong repo của mình. Bản gốc còn nguyên.

Cách đo: `scripts/runner_inference_sample.py --profile`, checkpoint zero-shot
`octo-base-1.5-torch`, dataset `bridge_dataset`, `--window 2`, ảnh tĩnh
`assets/obs_primary.png`, câu lệnh `pick up the red cube`.

## 1. Phát hiện lớn nhất: config inference TẮT wrist trong khi model được train CÓ wrist

### 1.1 Camera nào là camera nào (đo hôm nay)

| | `/dev/video0` — Logitech C925e | `/dev/video2` — SPCA2650 |
|---|---|---|
| ảnh chụp 1280x720 | nhìn thẳng xuống mặt bàn, bóng vàng ở góc dưới | toàn cảnh phòng: tay robot, thảm, bóng, trụ xanh, khối trắng, bàn ghế |
| vai trò thật | **WRIST** (gắn trên tay) | **PRIMARY** (camera cảnh) |
| định dạng | YUYV + MJPG, tới 1920x1080 | MJPG + YUYV, tới 2560x1440 |
| 1280x720 MJPG đo được | **10 fps** | **30 fps** |

Config của đàn anh ghi `primary.index = 0`, `wrist.index = 2`; đo hôm nay thì ngược lại, nên
bản copy đã sửa thành `primary.index = 2`, `wrist.index = 0`, `wrist.enabled = true`.

**Nhưng đàn anh KHÔNG hề gán sai nhãn.** Dataset của họ
(`collect_raw_datasets/pick_up_the_blue_block/raw/episode_000002/frames/`) có cả hai thư mục
`image_primary` và `image_wrist`, và ảnh cho thấy nhãn của họ **đúng theo nghĩa**:

- `image_primary` = góc nhìn thứ ba từ trên xuống (thấy thân tay robot, ngón kẹp đỏ/đen, trụ xanh)
- `image_wrist` = nhìn thẳng xuống, **thấy chính hai ngón kẹp ở hai góc dưới** → eye-in-hand thật

Tức là tháng 4/2026 index 0 là **camera cảnh**; còn bây giờ index 0 (Logitech) đang nằm ở
**tay kẹp**. Hai camera đã được **đổi chỗ vật lý** trong khoảng giữa. Con số index cũ không
sai — nó chỉ ứng với cách lắp cũ.

### 1.2 Lỗi thật: train có wrist, inference tắt wrist

`config.json` của checkpoint finetune `experiment_20260420_024612` khai báo **hai**
observation tokenizer:

    observation_tokenizers:  primary -> image_primary
                             wrist   -> image_wrist

và `example_batch.pickle` của chính nó xác nhận model mong đợi cả hai:

    image_primary               (32, 2, 3, 256, 256)  uint8
    image_wrist                 (32, 2, 3, 128, 128)  uint8
    pad_mask_dict/image_wrist   (32, 2)

Nhưng config inference của đàn anh để `wrist.enabled = false`. Khi đó
`build_observation_from_example()` lấp `image_wrist` bằng **số 0** và đặt
`pad_mask_dict/image_wrist = False` — model chạy với modality wrist bị **zero + mask**,
trong khi nó được train bằng ảnh wrist thật. Đây là **lệch train/inference thật sự**.

Bật `wrist.enabled = true` với Logitech (index 0) **không** phá vỡ checkpoint cũ — nó
**sửa đúng cái lỗi đó**. Ảnh wrist còn có mặt trong cả `task` (`task/image_wrist`), nên
nhánh goal cũng cần.

## 2. Kết quả đo

| chỉ số | trước | sau | chênh |
|---|---|---|---|
| **e2e `sample_actions`** | **850,70 ms** | **608,11 ms** | **−242,6 ms (−28,5%)** |
| tốc độ suy luận | ~1,18 Hz | **~1,64 Hz** | +39% |
| tổng các module đo được | 511,15 ms | 501,84 ms | −9,3 ms |
| **phần không giải thích được** | **339,55 ms (39,9%)** | **106,27 ms (17,5%)** | −233,3 ms |

Chi tiết breakdown (trung bình 5 lần):

| module | trước | sau |
|---|---|---|
| `transformer_backbone` | 241,64 ms | 248,44 ms |
| `action_head` (20 bước diffusion) | 176,25 ms | 170,37 ms |
| `task_tokenizer.language` | 68,75 ms | 59,39 ms (*) |
| `obs_tokenizer.wrist` | 11,58 ms | 10,56 ms |
| `obs_tokenizer.primary` | 11,50 ms | 12,21 ms |
| `readout.action` | 1,43 ms | 0,87 ms |

(*) Con số này **đo cao hơn thực tế** ở lần chạy sau: breakdown gọi thẳng tokenizer
(`octo_t.task_tokenizers[name](...)`) nên **đi vòng qua cache**; còn đường e2e thật thì
trúng cache. Đây là điểm profiler cũ chưa phản ánh đúng.

## 3. Bốn chỗ tối ưu

### 3.1 Cache embedding câu lệnh — `octo/model/octo_module_pt.py`

`LanguageTokenizerPt` chỉ dùng `tasks`, **không dùng `observations`**, và câu lệnh không
đổi trong suốt episode. Vậy mà t5-base vẫn chạy lại mỗi bước: 68,75 ms/bước, ~8% e2e.

Thêm `_run_task_tokenizer()` + cache trong `OctoTransformerPt`:
- khoá cache theo **nội dung** (`_content_digest`, blake2b trên bytes của tensor), không
  theo `id()` — tránh trường hợp tensor bị giải phóng rồi cấp lại địa chỉ làm cache trả
  nhầm kết quả của câu lệnh khác;
- chỉ cache khi `train=False` (lúc train có dropout và cần graph autograd).

**Lỗi đã gặp và sửa:** lần chạy đầu tiên nổ
`RuntimeError: Inference tensors cannot be saved for backward` tại
`task_tokens = self.task_projections[...]`. Nguyên nhân: warmup chạy trong
`torch.inference_mode()` nên tensor bị cache mang cờ "inference", mà e2e của profiler lại
chạy ở chế độ grad thường. Sửa bằng cách clone trong `torch.inference_mode(False)` để có
tensor thường (đã kiểm chứng: `clone()` trong `inference_mode(False)` cho
`is_inference() == False`, và `Linear` chạy được trên nó).

**Đã kiểm chứng** bằng `scripts/verify_lang_cache.py`:

| kiểm tra | kết quả |
|---|---|
| lần chạy đầu tiên | `{'hit': 0, 'miss': 1}` |
| sau 5 lần chạy | `{'hit': 4, 'miss': 1}` |
| có cache (median) | **456,13 ms** |
| không cache (median) | 505,55 ms |
| **tiết kiệm mỗi bước** | **49,42 ms** |
| action có/không cache | **giống hệt nhau**, sai khác tuyệt đối max `0.000e+00` |

Lưu ý về phép đo: script này chạy trong `torch.inference_mode()` — **đúng như rollout
thật** — nên cho 456 ms, thấp hơn con số 608 ms của profiler. Profiler gọi e2e ở chế độ
grad thường (không bọc `inference_mode`), tức là **đo cao hơn thực tế**. Con số đáng tin
cho rollout thật là ~456 ms ≈ **2,2 Hz**.

### 3.2 Camera wrist thật — `scripts/robot_configs/*.json`

Xem mục 1. Thêm `htc/test_1_octo/check_cameras.py` để kiểm tra thật: mở từng camera theo
đúng config, đọc 10 khung, in ra độ phân giải/fourcc **thực tế được thiết lập** (không tin
giá trị yêu cầu) và ghi ảnh ra `/tmp` để xem bằng mắt.

Phải chạy khi SDK đã tắt — xem mục 5.

### 3.3 Mask attention — cảnh báo SAI của build, vấn đề thật nằm chỗ khác

Cảnh báo `Converting mask without torch.bool dtype to bool` xuất hiện ở
`transformer_pt.py:362` (`attn_mask=attention_mask`). Nhưng:

1. **Đo trực tiếp dtype** (monkeypatch `Encoder1DBlockPt.forward`) cho thấy mask **đã là
   `torch.bool`**, shape `(12, 690, 690)`, trên `cuda:0`.
2. **Thử trên torch trần:** `MultiheadAttention` với `attn_mask` bool, `attn_mask` float,
   và `key_padding_mask` bool **đều cho đúng 1 cảnh báo**; không truyền mask thì 0 cảnh báo.

Kết luận: trên build `2.1.0a0+41361538.nv23.06` này, `_native_multi_head_attention` phát
cảnh báo đó bất kể dtype thật. **Không có gì để sửa phía ta.**

Nhưng đọc kỹ `transformer_pt.py` thì lộ ra vấn đề thật, và nó **lớn hơn nhiều**: khối
"mở lại đường chéo cho hàng bị mask hết" nằm trong `Encoder1DBlockPt.forward`, mà hàm này
được gọi **12 lần** mỗi lần inference (12 lớp transformer). Mỗi lần:

- `attention_mask.clone()` — tensor `(12, 690, 690)` bool = **5,7 MB**;
- `am.all(dim=-1)` — reduce trên 5,7M phần tử;
- và nếu có hàng bị mask hết thì vòng lặp Python gọi `nonzero()` **từng hàng một** — mỗi
  `nonzero()` là một lần đồng bộ GPU. 12 hàng × 12 lớp = **144 lần đồng bộ mỗi inference**.

Mask thì **y hệt nhau** giữa các lớp, nên kết quả cũng y hệt: làm một lần là đủ.

Đã sửa: hoist lên `_ensure_some_key_unmasked()` và gọi **một lần** trong
`TransformerPt.forward`, đồng thời vector hoá vòng lặp
(`rows, cols = fully_masked.nonzero(as_tuple=True); am[rows, cols, cols] = False` — tương
đương bản cũ nhưng chỉ gọi `nonzero()` một lần), và **kiểm tra trước khi clone** để tránh
copy 5,7 MB khi không cần.

Đây chính là phần lớn của 243 ms: tổng các module gần như không đổi (511 → 502 ms) trong
khi e2e giảm 243 ms — tức thời gian nằm ở phần "keo keo" mà profiler cũ không đo.

### 3.4 Truy 340 ms không giải thích được

Phương pháp: thêm `scripts/profile_split_e2e.py`, chia đúng những gì e2e làm thành 3 phần
(A. `module.forward`, B. `action_head`, C. hậu xử lý) rồi đối chiếu với e2e. Profiler cũ
đo từng module **riêng lẻ** nên không thấy được chi phí ở phần nối.

Kết quả: phần không giải thích được **339,55 ms → 106,27 ms**. Phần còn lại chưa truy hết —
ứng viên là chi phí glue trong `OctoTransformerPt.forward` (projections, positional
embedding, `repeat_interleave` mask thành `(12, 690, 690)`, ghép token) mà profiler không đo.

## 4. Cách chạy lại

```bash
cd /workspace/6DoF_Grasp/htc/test_1_octo/octo_pytorch
PY=/home/ktmt-agx-xv/Data/khoanhd/Octo_Lab/Octo_env/bin/python

# Đo hiệu năng
LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1 $PY scripts/runner_inference_sample.py \
  --ckpt /home/ktmt-agx-xv/Data/khoanhd/Octo_Lab/octo_weights/pretrained_pytorch/octo-base-1.5-torch \
  --obs_primary assets/obs_primary.png --text "pick up the red cube" \
  --window 2 --dataset bridge_dataset --profile

# Chứng minh cache đúng + nhanh
LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1 $PY scripts/verify_lang_cache.py --ckpt <dir>

# Chia e2e để truy phần thời gian còn lại
LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1 $PY scripts/profile_split_e2e.py --ckpt <dir> \
  --obs_primary assets/obs_primary.png --text "pick up the red cube" --dataset bridge_dataset
```

## 5. Cảnh báo vận hành

- **SDK và Octo không thể chạy cùng lúc.** `web_control.py` đang giữ cả `/dev/video0` và
  `/dev/video2`, và driver ROS 2 giữ `/dev/ttyACM1`; pipeline Octo mở camera trực tiếp và
  dùng `pymycobot` trên serial. Chỉ **một** tiến trình được stream V4L2 và **một** tiến
  trình được giữ serial. Muốn chạy Octo thật thì phải tắt SDK trước:
  `pkill -9 -f run_web.sh; pkill -9 -f web_control`
- `check_cameras.py` phải chạy lúc SDK đã tắt.
- Ảnh `primary` trong lần đo hiện tại là **ảnh tĩnh** (`assets/obs_primary.png`), không
  phải camera thật — nên kết quả đo là của phần suy luận thuần, không dính camera.

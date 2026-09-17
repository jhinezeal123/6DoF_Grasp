# P1 — Đo chất lượng grounding của PhysBrain1.5-8B

Ngày chạy: 17/09/2026. Config: **giữ nguyên `--image-min-tokens 1024`** (bắt buộc cho
grounding, không đánh đổi bằng tốc độ — quyết định của chủ dự án).

Ảnh thử: khung primary thật `episode_000002/frames/image_primary/000000.jpg` (1280×720).

---

## 1. Định dạng prompt chính xác — lấy từ chính nhóm tác giả

Đọc `app.py` của Space `DeepCybo/physbrain1-5-8b-demo`, có comment ghi rõ:

```python
# Prompt suffix used by the authors' evaluation harness (EmbodiedEvalKit) for
# Qwen3-VL-backboned models. Coordinates come back normalised to 0-1000.
POINT_SUFFIX = 'The answer should be presented in JSON format as follows: [{"point_2d": [x, y]}].'
```

Chế độ point: `instruction + "\n" + POINT_SUFFIX`

Ba chi tiết bắt buộc phải theo, sai là hỏng im lặng:

| | giá trị | nguồn |
|---|---|---|
| Toạ độ | chuẩn hoá **0–1000**; pixel = `x/1000*width` | `to_pixels()` |
| Nhiệt độ | **0 (greedy)** — cấu hình tác giả dùng để đánh giá | `temperature=0.0` |
| Ảnh tối đa | 1 megapixel (1280×720 = 0,92 MP nằm dưới ngưỡng) | `MAX_PIXELS` |

## 2. Grounding CHÍNH XÁC — đã kiểm chứng bằng mắt

| câu hỏi | model trả | pixel | kết quả |
|---|---|---|---|
| "Please point out the blue block." | `[{"point_2d":[615,445]}]` | (787, 320) | ✅ **nằm trên khối trụ xanh** |
| "free space to the right of the blue block" | `[{"point_2d":[700,450]}]` | (896, 324) | ✅ **mặt bàn trống, lệch phải khối xanh** |

Ảnh kiểm chứng: `artifacts/p1_A_point.jpg`, `artifacts/p1_C_free.jpg`

Ba quan sát:

1. **Vật thật là khối TRỤ xanh, không phải khối hộp** — hỏi "blue block" model vẫn
   tìm đúng. Nó hiểu ngữ nghĩa, không bám từ khoá.
2. **Hai ngón kẹp đỏ hình chữ V hiện rất rõ** trong ảnh → phát hiện bằng màu khả thi.
3. Model hiểu **quan hệ không gian** ("to the right of"), không chỉ định vị vật.

### Điểm cần lưu ý

Điểm model trả về nằm ở **mép trên** khối trụ, không phải tâm. Với kẹp thì cần tâm vật.
Đây là việc của tầng servo (`J`), không phải lỗi model.

## 3. Chế độ TRACE THẤT BẠI — không dùng được

Prompt trace chính chủ (`app.py`, `MODE_TRACE`), hỏi 8 điểm cho "pick up the blue block":

```
[{"point_2d":[532,363]},{"point_2d":[532,363]},{"point_2d":[532,330]},
 {"point_2d":[532,290]},{"point_2d":[532,250]},{"point_2d":[532,210]},
 {"point_2d":[532,170]},{"point_2d":[532,130]}]
```

**Toàn bộ 8 điểm có cùng x=532**, y giảm đều 363→130. Đó là **một đường thẳng đứng**,
và ảnh `artifacts/p1_B_trace.jpg` xác nhận: đường đi thẳng lên từ ngón kẹp,
**không hề tiến về khối xanh**.

Đây là quỹ đạo suy biến, không mang thông tin. **Kết luận: bỏ chế độ trace.**

## 4. Prompt cache — đo được, và nó quyết định thiết kế

```
=== CUNG anh + CUNG prompt, 4 lan lien tiep ===
  lan 1:  2,7 s   prompt_n=1115  cached=1084
  lan 2:  2,3 s   prompt_n=1115  cached=1114
  lan 3:  2,3 s   prompt_n=1115  cached=1114
  lan 4:  2,3 s   prompt_n=1115  cached=1114

=== ANH KHAC NHAU, cung prompt ===
  000000.jpg   2,5 s   cached=1114
  000001.jpg  17,5 s   cached=0      <- VO
  000002.jpg  16,9 s   cached=0      <- VO
  000003.jpg  15,7 s   cached=0      <- VO
=== quay lai anh DAU TIEN ===
  000000.jpg   2,5 s   cached=1114   <- cache VAN GIU
```

Ba kết luận:

1. **Cùng ảnh → 21,0 s giảm còn 2,3 s (nhanh 9×).** Toàn bộ 1114/1115 token được cache,
   tức **bỏ hẳn phần encode ảnh ~11,5 s**.
2. **Ảnh mới → vỡ cache hoàn toàn → 15,7–17,5 s.**
3. **Cache giữ được nhiều ảnh**, không chỉ ảnh gần nhất.

## 5. Đáp án ổn định

Với `temperature=0`, 4 lần gửi cùng ảnh cho **đúng một đáp án** `[{"point_2d":[615,445]}]`.
Qua 4 khung hình khác nhau (tay di chuyển, vật đứng yên): `(615,445)`, `(615,450)`,
`(615,450)`, `(615,445)` — gần như bất biến.

**Vật đứng yên thì toạ độ vật đứng yên, bất kể tay ở đâu.**

## 6. Hệ quả cho thiết kế test 2

| | chi phí |
|---|---|
| Gọi model, ảnh MỚI | ~16–17 s |
| Gọi model, ảnh ĐÃ GỬI | ~2,5 s |
| Sinh chữ | ~8,7 token/s |
| Phát hiện ngón kẹp đỏ | ~1 ms |

Camera sống → mỗi khung hình là ảnh mới → **~17 s mỗi bước**, 25 bước ≈ **7 phút**.

Nhưng vì toạ độ vật bất biến và ngón kẹp bắt được bằng màu (§7 của `P0_KET_QUA.md`:
đúng 2 blob đỏ, tâm (741,267) và (609,220)):

```
Goi model 1 lan  ->  target_point_2d          ~17 s
Moi buoc servo   ->  ngon kep tu blob do      ~1 ms
```

**Tổng một episode ≈ 17 s + thời gian tay di chuyển**, thay vì 7 phút.

Vẫn gọi lại model khi cần `can_grasp` — mỗi lần ~17 s, nên chỉ gọi ở vài mốc quyết định.

## 7. Việc chưa làm

- Chưa thử trên ảnh chụp từ camera primary hiện tại (mới dùng ảnh dataset)
- Chưa đo `point_2d` khi có vật cản hoặc nhiều vật cùng màu
- Chưa thử mmproj `Q8_0`
- Chưa kiểm tra model có trả nhiều điểm khi hỏi "all blue objects" không

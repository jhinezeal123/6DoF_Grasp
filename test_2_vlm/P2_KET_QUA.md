# P2 — Hiệu chuẩn Jacobian command → image

Ngày chạy: 17/09/2026. Script: `scripts/calibrate_jacobian.py`.
Kết quả: `configs/servo_jacobian.json`.

---

## 1. Kết quả

```
g0 = (697.6, 414.0) px   tcp = [0.3530  0.0826  0.2007]
g1 = (659.2, 428.4) px   dich anh = (-38.40, +14.40) px   dich TCP = [+19.1, +1.1, -3.3] mm
gb = (697.6, 414.0) px   (ve goc)
g2 = (697.6, 421.2) px   dich anh = (  0.00,  +7.20) px   dich TCP = [+0.2, +14.7, -5.5] mm

J = [[-1920.0,    0.0]     pixel/met      det(J) = -6.91e5
     [  720.0,  360.0]]                    cond(J) = 6.1   OK

1 mm theo x  ->  2.05 px
1 px sai so  ->  0.49 mm
```

**Vì sao tin được con số này:**

1. **Tay thật sự di chuyển.** Delta TCP đo được là +19.1 mm và +14.7 mm cho hai lệnh
   20 mm — không phải tay đứng im (xem §2, lỗi 2).
2. **Model lặp lại được.** Lần quay về gốc, model trả về **đúng chuỗi `[545,575]`
   y hệt g0**. Vừa xác nhận tay về đúng tư thế, vừa xác nhận model không nhiễu.
3. **Tín hiệu lớn hơn nhiễu nhiều.** Bước x cho 38.4 px dịch chuyển, trong khi
   nhiễu nền đo được ở chế độ dry-run chỉ ~2–3 px.
4. **đã kiểm chứng bằng mắt** (`artifacts/p2_verify_g0.jpg`, `p2_verify_g1.jpg`):
   dấu ngắm nằm trên ngón kẹp đỏ hình chữ V, lệch đúng chiều như số đo.

**Điểm yếu cần biết:** `J[0][1] = 0.0` đúng bằng 0 vì bước y không làm ngón kẹp
dịch theo x. Nhưng canvas 0–1000 của model có độ phân giải 1 đơn vị = 1.28 px
theo x, nên giá trị thật nằm trong khoảng **±64 px/m**. So với số hạng trội
1920 px/m thì sai số này ~3%. Chấp nhận được.

---

## 2. Ba lỗi chặn đường — và cách sửa

Cả ba đều thuộc loại "hỏng im lặng": mọi thứ trông bình thường nhưng kết quả vô nghĩa.

### Lỗi 1 — DDS discovery chập chờn (lỗi môi trường, không phải lỗi SDK)

`tcp_pos` trả `[nan nan nan]` và `is_real_connected = False` một cách ngẫu nhiên,
dù publisher phát đều. Cùng topic, cùng QoS, node này nhận node kia không.

**Nguyên nhân gốc: 197 file shared-memory FastRTPS mồ côi trong `/dev/shm`.**
Mỗi script chạy tạo một DDS participant; participant chết để lại xác. Khi tích tụ
~200 xác, việc khớp publisher–subscriber trở thành may rủi.

Đây **cũng chính là lời giải cho bí ẩn từ test 1**: `tcp_pos` trả NaN mà không ai
hiểu vì sao. Không phải lỗi SDK — là môi trường DDS bị nhiễm.

**Cách chữa tận gốc:** tắt hẳn transport shared-memory, chỉ dùng UDP loopback.
Robot nằm cùng máy nên độ trễ không đáng kể, mà xung đột `/dev/shm` biến mất.

```xml
<!-- /tmp/udp_only.xml -->
<transport_descriptors>
  <transport_descriptor>
    <transport_id>udp_only</transport_id><type>UDPv4</type>
  </transport_descriptor>
</transport_descriptors>
<participant profile_name="udp_only_participant" is_default_profile="true">
  <rtps><userTransports><transport_id>udp_only</transport_id></userTransports>
        <useBuiltinTransports>false</useBuiltinTransports></rtps>
</participant>
```

`export FASTRTPS_DEFAULT_PROFILES_FILE=/tmp/udp_only.xml` cho **mọi** tiến trình
(cả stack lẫn SDK).

Kết quả đo: **5/5 process mới liên tiếp đều đạt**, `/dev/shm` còn **0 file**.
Trước đó tỉ lệ hỏng ước tính 1/3–1/2.

Lưu ý: `pkill -f 'myarm_sdk/ros2_ws/install'` **không** khớp `robot_state_publisher`
(binary hệ thống) — phải kill riêng, không thì stack cũ còn sống sót.

### Lỗi 2 — Executor từ chối lệnh nhưng `send_tcp_pose` vẫn trả `True`

```
motion_state = ('idle', 'joint goal execution rejected: driver safety gate
                        is disarmed; operator re-arm is required', 1)
```

Tay đang `disarmed` thì executor **từ chối mọi lệnh**, nhưng `send_tcp_pose()`
trong `ros_bridge.py` chỉ publish rồi `return True` vô điều kiện. Script tưởng
thành công, tay đứng im, và J đo được **toàn là nhiễu** — mà `cond` vẫn ra 2.1
"đẹp" nên rất dễ tin nhầm.

**Sửa:** gọi `robot.rearm()` trước mọi chuyển động, và kiểm tra `is_armed` sau đó.
`rearm()` phải gọi **cả hai** service (`/myarm/robot/rearm` +
`/myarm/motion_execution/reset`) — docstring của SDK đã cảnh báo đúng.

**Chốt chặn thêm:** script tự abort nếu TCP dịch < 20% độ lớn lệnh.

### Lỗi 3 — Chờ sai trạng thái kết thúc

Executor báo **`succeeded`**, không phải `idle`. Hàm chờ đầu tiên của tôi chỉ chấp
nhận `idle` nên quay vòng vô ích 15 s rồi báo lỗi oan, trong khi tay đã đứng yên
từ giây thứ 4.

Đo được: `executing` → `succeeded` ở t=3.0 s → tốc độ 0.00 mm/s từ t=4.0 s và
**không nhúc nhích suốt 25 s sau đó**.

**Sửa:** chờ cho trạng thái **không còn `executing`/`pending`**, rồi xác nhận TCP
đứng yên 4 mẫu liên tiếp (ngưỡng 0.5 mm). Cũng bỏ luôn `sleep(1.2)` cố định.

---

## 3. Nhận diện ngón kẹp: bỏ cách dò màu, chuyển sang hỏi model

Cách dò màu đỏ **thất bại về nguyên tắc**, không phải do ngưỡng.

Đo trên cảnh thật: **11 vật đỏ** — ghế đỏ, áo đỏ, vật đỏ trên bàn, người đi lại.
Không vật nào là ngón kẹp. Bộ lọc khoanh vào mặt người, nút e-stop, ghế nền.
Tệ hơn, auto-exposure đổi khi có người vào khung làm ngón kẹp tối đi và **rớt
khỏi ngưỡng** (ngón sáng V≈110, ngón tối V≈80).

Đã thử hai hướng vá — chọn theo cặp, rồi giao mặt nạ đỏ với vùng ảnh đã thay đổi —
nhưng chủ dự án quyết định: **dùng model dò ngón kẹp liên tục, chấp nhận chậm.**

```
GRIPPER_Q = "Point to the red gripper fingers at the tip of the robot arm."
```

Dùng đúng định dạng tác giả (xem `P1_KET_QUA.md` §1): prompt + `POINT_SUFFIX`,
toạ độ chuẩn hoá 0–1000, `temperature=0`.

**Chi phí thực đo: 14.8–15.2 s mỗi lần gọi** (ảnh mới, cache vỡ). Cả P2 mất
khoảng 90 s cho 4 lần gọi.

**Đã kiểm chứng bằng mắt** (`artifacts/mp_g0.jpg`): dấu ngắm nằm chính xác trên
ngón kẹp đỏ, **không** nhầm sang nắp nhựa đen của cụm kẹp, cũng **không** nhầm
sang quả bóng vàng ngay phía sau.

---

## 4. Điều chỉnh so với kế hoạch

Kế hoạch ban đầu (`P1_KET_QUA.md` §6) tính dùng model **một lần** rồi theo dõi
ngón kẹp bằng màu ~1 ms mỗi bước, để một episode chỉ tốn ~17 s. Cách đó **không
dùng được** vì lý do ở §3.

Hệ quả cho tầng servo: mỗi bước servo cần một lần gọi model, **~15 s/bước**.
Với `max_steps = 25` thì một episode ≈ **6–7 phút**. Chủ dự án đã chấp nhận
đánh đổi này.

---

## 5. Việc chưa làm

- Chưa đo lại J ở tư thế khác để kiểm tra J có ổn định theo cấu hình không
  (J phụ thuộc tư thế — đây là điều đã biết của servo ảnh, không phải bất ngờ)
- Chưa đo độ lặp lại của J qua nhiều lần chạy
- Chưa thử `--step` lớn hơn (4 cm) để tăng tín hiệu trên nhiễu
- `J[0][1]` chỉ đo được một bước; muốn chắc thì nên đo 3 lần rồi lấy trung vị

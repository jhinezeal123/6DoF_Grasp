# Review PR — 6DoF_Grasp

Rà soát ngày 03/10/2026 tại `main@e99d6ca9a3334ffb357b653b4fc1a793cd24c7cf`.
Đã đối chiếu diff, lịch sử merge, CI và chạy test offline/MuJoCo. Chưa truy cập
checkout server hoặc gửi lệnh tới robot thật. Các kết quả Jetson ghi trong PR
là bằng chứng của tác giả; không phải đo lại trong đợt review này.

| PR | Trạng thái | Quyết định | Lý do |
| --- | --- | --- | --- |
| [#1](https://github.com/jhinezeal123/6DoF_Grasp/pull/1) | Đã merge | Giữ contracts/backend; refactor tiếp | RobotDriver, optional power/stop, conversion degree/radian và perception public API là nền tốt. PR trộn migration backend với refactor; lớp `arm/control` và Web UI cũ còn xen nhau. |
| [#2](https://github.com/jhinezeal123/6DoF_Grasp/pull/2) | Đã merge | Giữ | Conda riêng và worker client giúp 6DoF không kéo Torch/TensorRT vào process robot. Protocol có guards cho timeout/JSON; không dùng chung venv với pipeline. |
| [#3](https://github.com/jhinezeal123/6DoF_Grasp/pull/3) | Đóng, chưa merge | Bỏ nhánh cũ | Đã được chứa trong #5. Không merge lại và không revert PR chưa merge. |
| [#4](https://github.com/jhinezeal123/6DoF_Grasp/pull/4) | Đóng, chưa merge | Bỏ nhánh cũ | Harness đã được đưa vào #5 với cấu trúc mới. |
| [#5](https://github.com/jhinezeal123/6DoF_Grasp/pull/5) | Đã merge | Giữ; làm dễ đọc | Feature `sim/`, depth injection và guard simulator là hữu ích. Nhiều hàm bị nén nhiều câu trên một dòng; refactor hiện tại mở lại format và tách reporting. |
| [#6](https://github.com/jhinezeal123/6DoF_Grasp/pull/6) | Đã merge | Giữ | OSMesa/PyOpenGL chạy rendering thật trên CI, không skip test cần thiết. |
| [#7](https://github.com/jhinezeal123/6DoF_Grasp/pull/7) | Đã merge | Giữ | Một nguồn tool0 offset/rotation và joint names; có test hai MuJoCo backend đồng ý với nhau. |
| [#8](https://github.com/jhinezeal123/6DoF_Grasp/pull/8) | Đã merge | Giữ | Dùng SciPy thống nhất và bỏ import helper riêng xuyên feature. Rotation tương đương; JSON quaternion có thể đổi dấu q/−q, nên không gọi đó là byte-identical refactor. |
| [#9](https://github.com/jhinezeal123/6DoF_Grasp/pull/9) | Đã merge | Giữ phép đo như diagnostic | Box signed distance và cube 25 mm đúng với scene. Sai số tới surface không tự chứng minh grasp tốt: pose origin của gripper khác điểm tiếp xúc; lift/contact/collision vẫn là kiểm chứng cần thiết. |

## Điểm cần sửa ngoài refactor

**P1 — pin trong metadata không phải SHA worker thực tế.**
`perception/adapters/grasppose.py` luôn gắn `GRASPPOSE_COMMIT=666c7eb` vào grasp,
trong khi socket client không kiểm chứng SHA của process bên kia. Pipeline `main`
đã ở `c417fd0` và dùng DA3. Vì thế có thể ghi metadata Lite-Mono pin cho kết quả
được tạo bởi DA3. Chưa tự đổi pin trong refactor; cần rollout/provenance riêng.

**P1 — pipeline #13 cần rollback riêng lựa chọn depth mặc định.**
Composition mặc định chọn DA3, nhưng `prepare.sh`/requirements vẫn chuẩn bị
Lite-Mono; scale DA3 chưa được kiểm chứng trên camera thật. Không tự cập nhật
server theo `main`; phải biết SHA và calibration đang chạy.
Xem [review pipeline](https://github.com/jhinezeal123/pipeline_grasppose/blob/refactor/readable-perception-vi/docs/review-pr-vi.md).

**P2 — ROS/Web UI và pymycobot là hai đường triển khai khác nhau.**
Web UI cũ vẫn dùng ROS, không phải UI mới cho `RobotDriver`. Refactor tách HTTP,
template và render ra nhưng giữ endpoint và hành vi. Migration UI sang
`RobotControl` sẽ là change riêng; không xóa ROS hay giả vờ migration đã xong.

**P2 — #9 không phải tiêu chí thành công.**
Không dùng `surface_mm≈0` làm mục tiêu hiệu chỉnh grasp. VGN định nghĩa pose
gripper tại voxel, quality gắn với pose đó, không định nghĩa origin luôn là điểm
trên mặt vật. Giữ tên field/report hiện tại để không phá dữ liệu đã ghi.
Nguồn: [VGN, mục 3–4 và hình 2c](https://proceedings.mlr.press/v155/breyer21a/breyer21a.pdf).

Không thấy căn cứ để revert toàn bộ PR đã merge của 6DoF. Nên giữ các contract,
guard và calibration; tách những sửa chức năng vừa nêu khỏi refactor.

## Kiểm chứng

- CI gốc `e99d6ca`: [success](https://github.com/jhinezeal123/6DoF_Grasp/actions/runs/37033747730).
- Nền: 65 passed, 2 ca Unix socket bị host này chặn.
- Thêm 4 test serial giả và 2 golden tests HTML, chạy xanh trước refactor.
- Sau refactor: 71 passed; chỉ deselect 2 ca socket bị chặn. Không thêm skip/xfail.
- AST toàn bộ thân class/hàm arm, controller, FK, IK, safety, viewpoints và
  workflow/adapter mô phỏng được đối chiếu với main; toán học không đổi.
- HTML của hai cấu hình calibration giữ nguyên SHA-256. Wheel phải chứa template.
- Không chạy legacy ROS service do không có ROS trong môi trường review;
  golden tests kiểm tra trang tĩnh, không chứng minh ROS/HTTP runtime hoàn chỉnh.
- CLI mới được kiểm thử trên fake driver; không điều khiển robot để kiểm tra.

## Thứ tự áp dụng

Merge PR refactor trước. PR CLI root lấy refactor làm base và chỉ bổ sung giao
diện dòng lệnh. Chờ full CI trên GitHub; sau đó chạy test/integration bằng venv
riêng trên server. Không đổi calibration, pin hay firmware cùng lần refactor.

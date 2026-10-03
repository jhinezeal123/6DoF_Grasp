"""Menu theo công việc; FEATURES dùng chung cho menu và --list."""

import argparse
import os
from pathlib import Path

from .config import ProfileStore
from .tasks import Tasks
from .terminal import Back, CommandRunner, Terminal


FEATURES = (
    (
        "status",
        "Kiểm tra và đọc trạng thái robot",
        "Xem môi trường, thiết bị và feedback; không gửi chuyển động.",
    ),
    (
        "robot",
        "Điều khiển robot thật",
        "Khớp / kẹp / TCP / servo / stop; hiện lệnh và đơn vị trước khi gửi.",
    ),
    ("camera", "Xem camera / chụp ảnh", "Mở camera V4L2 hoặc chụp một ảnh riêng tư."),
    (
        "simulation",
        "Thử gắp 10 cảnh mô phỏng",
        "Preset DA3 + confidence 0.05 + gravity; tâm volume dùng ground-truth.",
    ),
    (
        "perception",
        "Thử perception: ảnh → pose gắp",
        "Mở menu module con; dùng environment riêng của pipeline.",
    ),
    (
        "advanced",
        "Công cụ nâng cao: ROS / VLA / preview",
        "ROS cần SDK riêng; VLA mock là thử nghiệm, preview cần scene máy.",
    ),
    (
        "configure",
        "Thiết lập robot / camera / perception",
        "Lưu port, tốc độ, camera và đường dẫn module con một lần.",
    ),
    (
        "setup",
        "Cài môi trường robot",
        "Dùng script Conda hiện có để chuẩn bị Python và dependency.",
    ),
)


def main(argv=None, terminal=None):
    parser = argparse.ArgumentParser(
        description="Menu myArm M750 tiếng Việt. Chạy ./start; :q để quay lại khi nhập dữ liệu."
    )
    parser.add_argument("action", nargs="?", choices=[f[0] for f in FEATURES])
    parser.add_argument("--list", action="store_true", help="xem tính năng, không mở serial")
    parser.add_argument("--dry-run", action="store_true", help="xem lệnh, không chạy tác vụ")
    parser.add_argument("--config", type=Path, help="file profile riêng của menu")
    args = parser.parse_args(argv)
    view = terminal or Terminal()
    root = Path(__file__).resolve().parents[3]
    if args.list:
        view.say("6DOF — các công việc có thể làm")
        for key, title, summary in FEATURES:
            view.say("./start %s — %s\n  %s" % (key, title, summary))
        return 0
    try:
        store = ProfileStore(root, args.config)
        tasks = Tasks(root, store, view, CommandRunner(root, view, args.dry_run))
        while True:
            view.say("\nMYARM M750 — chọn công việc")
            view.say(
                "Serial: %s | tốc độ: %d | Python robot: %s"
                % (
                    tasks.profile.port,
                    tasks.profile.speed,
                    "đã có"
                    if os.access(
                        os.environ.get("M750_PYTHON", str(root / ".venv/bin/python")), os.X_OK
                    )
                    else "chưa cài",
                )
            )
            view.say("Nhập :q để quay lại; '-' để xóa giá trị đã lưu.")
            if args.action:
                key = args.action
            else:
                selected = view.choose(
                    "Bạn muốn làm gì?",
                    {
                        str(i + 1): "%s [%s]" % (item[1], tasks.availability(item[0]))
                        for i, item in enumerate(FEATURES)
                    },
                )
                if selected is None:
                    return 0
                key = FEATURES[int(selected) - 1][0]
            try:
                view.say(next(f[2] for f in FEATURES if f[0] == key))
                getattr(tasks, key)()
            except Back:
                view.say("Đã quay lại.")
            except (OSError, RuntimeError, ValueError) as exc:
                view.say("Chưa chạy được: %s" % exc)
                view.say("Chọn Thiết lập để sửa cấu hình, hoặc Cài môi trường nếu thiếu Python.")
                if args.action:
                    return 1
            if args.action:
                return 0
    except (EOFError, Back):
        view.say("\nĐã thoát menu.")
        return 0
    except KeyboardInterrupt:
        view.say("\nĐã ngắt tác vụ.")
        return 130
    except (OSError, ValueError, TypeError) as exc:
        view.say("Không đọc được cấu hình: %s. Dùng --config để chọn file khác." % exc)
        return 1

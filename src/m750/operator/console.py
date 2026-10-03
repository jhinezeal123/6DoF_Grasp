"""Menu theo công việc; FEATURES dùng chung cho menu và --list."""

import argparse
import os
from pathlib import Path

from .config import ProfileStore
from .tasks import FEATURES, Tasks
from .terminal import Back, CommandRunner, Terminal


def main(argv=None, terminal=None):
    parser = argparse.ArgumentParser(
        description="Menu myArm M750 tiếng Việt. Chạy ./start; :q để quay lại khi nhập dữ liệu."
    )
    parser.add_argument("action", nargs="?", choices=list(FEATURES))
    parser.add_argument("--list", action="store_true", help="xem tính năng, không mở serial")
    parser.add_argument("--dry-run", action="store_true", help="xem lệnh, không chạy tác vụ")
    parser.add_argument("--config", type=Path, help="file profile riêng của menu")
    args = parser.parse_args(argv)
    view = terminal or Terminal()
    root = Path(__file__).resolve().parents[3]
    if args.list:
        view.say("6DOF — các công việc có thể làm")
        for key, (title, summary, _) in FEATURES.items():
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
                        str(i + 1): "%s [%s]" % (title, tasks.availability(action))
                        for i, (action, (title, _, _)) in enumerate(FEATURES.items())
                    },
                )
                if selected is None:
                    return 0
                key = list(FEATURES)[int(selected) - 1]
            try:
                view.say(FEATURES[key][1])
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

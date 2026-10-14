"""Ghép tác vụ bằng lệnh/use case hiện có; serial chỉ mở trong ./robot."""

import json
import math
import os
import time
from dataclasses import replace
from pathlib import Path


# Một nơi khai báo action → (tên, mô tả, điều kiện sẵn sàng).
FEATURES = {
    "status": (
        "Kiểm tra và đọc trạng thái robot",
        "Xem môi trường, thiết bị và feedback; không gửi chuyển động.",
        "serial",
    ),
    "robot": (
        "Điều khiển robot thật",
        "Khớp / kẹp / TCP / servo / stop; hiện lệnh và đơn vị trước khi gửi.",
        "serial",
    ),
    "camera": (
        "Xem camera / chụp ảnh",
        "Mở camera V4L2 hoặc chụp một ảnh riêng tư.",
        "camera",
    ),
    "simulation": (
        "Thử gắp 10 cảnh mô phỏng",
        "Preset DA3 + confidence 0.05 + gravity; tâm volume dùng ground-truth.",
        "simulation",
    ),
    "perception": (
        "Thử perception: ảnh → pose gắp",
        "Mở menu module con; dùng environment riêng của pipeline.",
        "perception",
    ),
    "advanced": (
        "Công cụ nâng cao: ROS / VLA / preview",
        "ROS cần SDK riêng; VLA mock là thử nghiệm, preview cần scene máy.",
        "ROS cần SDK; VLA thử nghiệm",
    ),
    "configure": (
        "Thiết lập robot / camera / perception",
        "Lưu port, tốc độ, camera và đường dẫn module con một lần.",
        "có thể mở",
    ),
    "setup": (
        "Cài môi trường robot",
        "Dùng script Conda hiện có để chuẩn bị Python và dependency.",
        "cần Conda",
    ),
}


class Tasks:
    def __init__(self, root, store, view, runner):
        self.root, self.store, self.view, self.runner = root, store, view, runner
        self.profile = store.load()

    def availability(self, action):
        needs = FEATURES.get(action, (None, None, "python"))[2]
        if needs not in ("serial", "camera", "simulation", "perception", "python"):
            return needs
        if needs == "perception":
            return (
                "có menu con"
                if (Path(self.profile.pipeline_repo) / "start").is_file()
                else "cần cập nhật module con"
            )
        python = Path(os.environ.get("M750_PYTHON", str(self.root / ".venv/bin/python")))
        if not os.access(str(python), os.X_OK):
            return "cần môi trường"
        if needs == "serial" and not Path(self.profile.port).exists():
            return "cần thiết bị serial"
        if needs == "camera" and not Path(self.profile.camera_device).exists():
            return "cần camera"
        return "volume ground-truth" if needs == "simulation" else "có thể mở"

    def save(self, profile):
        profile.validate()
        if not self.runner.dry_run:
            self.store.save(profile)
        self.profile = profile
        self.view.say(
            "Cấu hình tạm cho dry-run; chưa ghi file."
            if self.runner.dry_run
            else "Đã lưu cấu hình; lần sau dùng lại."
        )

    def python(self):
        path = Path(os.environ.get("M750_PYTHON", str(self.root / ".venv/bin/python")))
        if not self.runner.dry_run and not os.access(str(path), os.X_OK):
            raise RuntimeError("Thiếu Python robot. Chọn mục Cài môi trường robot.")
        return path

    def pipeline(self):
        root = Path(self.profile.pipeline_repo).expanduser().resolve()
        if not (root / "scripts/worker.sh").is_file():
            raise RuntimeError(
                "Chưa tìm thấy module perception. Chọn Thiết lập → Perception, trỏ tới pipeline_grasppose."
            )
        return root

    def feedback(self, result):
        if self.runner.dry_run:
            return
        response = next(
            (
                json.loads(line)
                for line in reversed(result.stdout.splitlines())
                if line.startswith("{")
            ),
            None,
        )
        if not response:
            self.view.say(result.stdout.rstrip())
            return
        self.view.say("Kết quả: " + ("thành công" if response.get("ok") else "chưa thành công"))
        state = response.get("state", {})
        joints = state.get("joints_rad")
        if joints:
            self.view.say(
                "Góc Q1..Q6 (độ): " + " / ".join("%.1f" % math.degrees(v) for v in joints)
            )
        opening = state.get("gripper_opening_m")
        if opening is not None:
            self.view.say("Độ mở kẹp: %.1f mm" % (opening * 1000))
        pose = state.get("tcp_pose")
        if pose:
            self.view.say(
                "TCP trong base (mm): "
                + " / ".join("%.1f" % (v * 1000) for v in pose["position_m"])
            )

    def robot_command(self, command, motion=False):
        self.python()
        argv = [
            self.root / "robot",
            "--port",
            self.profile.port,
            "--baud",
            str(self.profile.baud),
            "--speed",
            str(self.profile.speed),
            *command,
        ]
        result = self.runner.run(
            argv, capture=True, confirm="Gửi lệnh này tới ROBOT THẬT?" if motion else None
        )
        self.feedback(result)

    def status(self):
        self.view.say("Profile: %s" % self.store.path)
        self.view.say("Python robot: %s" % self.python())
        self.view.say("Serial: %s | baud %d" % (self.profile.port, self.profile.baud))
        if not self.runner.dry_run and not Path(self.profile.port).exists():
            raise RuntimeError(
                "Chưa thấy thiết bị serial. Cắm robot hoặc sửa port trong Thiết lập → Robot."
            )
        self.robot_command(["state"])

    def numbers(self, label, count):
        values = self.view.ask(label).replace(",", " ").split()
        if len(values) != count:
            raise ValueError("Cần đúng %d số, cách nhau bằng dấu cách." % count)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("Các số phải hữu hạn.")
        return values

    def robot(self):
        selected = self.view.choose(
            "Điều khiển robot thật",
            {
                "1": "Đặt một khớp (độ)",
                "2": "Đặt sáu khớp (độ)",
                "3": "Đặt độ mở kẹp (mm)",
                "4": "Di chuyển TCP (mm + Euler XYZ độ)",
                "5": "Bật servo",
                "6": "Tắt servo",
                "7": "Stop qua serial",
                "8": "Đọc trạng thái",
            },
        )
        if selected is None:
            return
        if selected == "1":
            joint = int(self.view.ask("Khớp 1..6"))
            if not 1 <= joint <= 6:
                raise ValueError("Khớp phải trong 1..6.")
            degree = self.numbers("Góc tuyệt đối của khớp %d (độ)" % joint, 1)
            command = ["joint", str(joint), *degree]
        elif selected == "2":
            command = ["joints", *self.numbers("Góc tuyệt đối Q1 Q2 Q3 Q4 Q5 Q6 (độ)", 6)]
        elif selected == "3":
            command = ["gripper", *self.numbers("Tổng độ mở hai ngón (mm, 0..69)", 1)]
        elif selected == "4":
            command = [
                "tcp",
                *self.numbers("Tool0 trong base: X Y Z (mm) RX RY RZ (độ Euler XYZ)", 6),
            ]
        else:
            command = [{"5": "power-on", "6": "power-off", "7": "stop", "8": "state"}[selected]]
        if selected == "6":
            self.view.say("Tắt servo: đỡ tay robot trước khi gửi.")
        if selected == "7":
            self.view.say(
                "Stop dùng cùng serial port; không phải hardware E-stop. Process khác giữ port thì lệnh sẽ bị từ chối."
            )
        self.view.say("Yêu cầu: " + " ".join(command) + " | tốc độ %d" % self.profile.speed)
        self.robot_command(command, motion=selected not in ("7", "8"))

    def camera(self):
        selected = self.view.choose(
            "Camera", {"1": "Xem stream trực tiếp", "2": "Chụp ảnh để thử perception"}
        )
        if selected is None:
            return
        device = self.profile.camera_device
        if not self.runner.dry_run and not Path(device).exists():
            raise RuntimeError("Không thấy camera %s. Chọn Thiết lập → Camera." % device)
        argv = [
            self.python(),
            "-m",
            "m750.operator.camera",
            "stream" if selected == "1" else "photo",
            "--device",
            device,
        ]
        if selected == "1":
            port = self.profile.camera_port
            argv += ["--port", str(port)]
            self.view.say("Mở http://<IP-server>:%d ; Ctrl-C để đóng stream." % port)
        else:
            path = self.root / ".local_data" / "photos" / ("camera-%d.jpg" % time.time_ns())
            if not self.runner.dry_run:
                path.parent.mkdir(parents=True, exist_ok=True)
            argv += ["--out", str(path)]
        self.runner.run(argv)
        if selected == "2" and not self.runner.dry_run:
            if not path.is_file():
                raise RuntimeError("Camera chưa ghi được ảnh. Kiểm tra thiết bị và quyền ghi.")
            self.save(replace(self.profile, last_photo=str(path)))
            self.view.say("Ảnh: %s. Mục Thử perception sẽ gợi ý ảnh này." % path)

    def simulation(self):
        python = self.python()
        pipeline = self.pipeline()
        worker_python = pipeline / ".venv/bin/python"
        if not self.runner.dry_run and not os.access(str(worker_python), os.X_OK):
            raise RuntimeError(
                "Pipeline chưa có environment. Mở mục Thử perception → Chuẩn bị môi trường."
            )
        self.view.say(
            "Mô phỏng MuJoCo, không gửi chuyển động robot thật. Gravity dùng tâm vật thể ground-truth; đây là gate mô phỏng có volume được cung cấp."
        )
        self.view.say(
            "Sẽ restart worker với DA3/confidence 0.05 cho preset này; không sửa mặc định hoặc profile của pipeline."
        )
        if not self.runner.dry_run and not self.view.confirm("Chạy 10 cảnh với preset mô phỏng?"):
            return
        worker_env = {
            "GRASP_DEPTH_BACKEND": "da3",
            "YOLOE_CONF": "0.05",
            "GRASP_WORKER_SOCKET": self.profile.socket,
        }
        result = self.runner.run(["bash", pipeline / "scripts/worker.sh", "restart"], worker_env)
        if result.returncode:
            return
        self.runner.run(
            [
                python,
                self.root / "tools/sim_grasp_validation.py",
                "--mode",
                "e2e",
                "--volume",
                "gravity",
                "--depth-source",
                "worker",
                "--pipeline-repo",
                pipeline,
                "--socket",
                self.profile.socket,
            ],
            {"MUJOCO_GL": "egl"},
        )
        self.view.say(
            "Report/ảnh/video: .local_data/sim_grasp_validation/ . Đọc số thành công thực tế trong report; preset không tự bảo đảm 10/10."
        )

    def perception(self):
        pipeline = self.pipeline()
        entry = pipeline / "start"
        if not entry.is_file():
            raise RuntimeError(
                "Checkout pipeline chưa có menu ./start. Cập nhật PR menu của pipeline trước."
            )
        argv = ["bash", entry, "--socket", self.profile.socket]
        if self.runner.dry_run:
            argv.append("--dry-run")
        if self.profile.last_photo and Path(self.profile.last_photo).is_file():
            argv += ["--image", self.profile.last_photo]
        self.view.say("Chuyển sang menu perception. Thoát menu con để quay lại robot.")
        self.runner.run(argv)

    def advanced(self):
        selected = self.view.choose(
            "Nâng cao",
            {
                "1": "Web ROS: camera / teleop / offset (cần ROS + SDK lab)",
                "2": "VLA mock offline (thử nghiệm; không điều khiển robot)",
                "3": "Web preview cũ (cần simu/assets/myarm_m750.xml)",
            },
        )
        if selected == "1":
            self.view.say(
                "Stack ROS hiện có có thể bật servo khi khởi động. Dùng cấu hình lab/ROS trong Thiết lập → ROS."
            )
            if self.runner.dry_run or self.view.confirm("Khởi động stack ROS cho robot thật?"):
                self.runner.run(["bash", self.root / "run_web.sh"], self.profile.ros_environment())
        elif selected == "2":
            self.runner.run([self.python(), "-m", "m750.pipeline.examples.mock_run"])
        elif selected == "3":
            scene = self.root / "simu/assets/myarm_m750.xml"
            if not self.runner.dry_run and not scene.is_file():
                raise RuntimeError(
                    "Thiếu scene máy: %s. Dùng mục Thử gắp mô phỏng với scene đã có trong repo."
                    % scene
                )
            self.view.say(
                "Preview cũ có chức năng gửi lệnh robot; xem lại chế độ trước khi dùng sync/realtime."
            )
            if self.runner.dry_run or self.view.confirm(
                "Mở preview cũ có chức năng điều khiển robot?"
            ):
                self.runner.run(
                    [self.python(), "-c", "from m750.cli import preview_main; preview_main()"],
                    {"MUJOCO_GL": "egl"},
                )

    def configure(self):
        selected = self.view.choose(
            "Thiết lập một lần, dùng lại",
            {
                "1": "Robot: serial / baud / tốc độ",
                "2": "Camera: thiết bị / port stream",
                "3": "Perception: repo / socket",
                "4": "ROS: lab / Python / setup",
            },
        )
        if selected == "1":
            port = self.view.ask("Serial port", self.profile.port)
            baud = int(self.view.ask("Baud", self.profile.baud))
            speed = int(self.view.ask("Tốc độ 1..100", self.profile.speed))
            self.save(replace(self.profile, port=port, baud=baud, speed=speed))
        elif selected == "2":
            device = self.view.ask("Thiết bị camera V4L2", self.profile.camera_device)
            port = int(self.view.ask("Port stream", self.profile.camera_port))
            self.save(replace(self.profile, camera_device=device, camera_port=port))
        elif selected == "3":
            repo = (
                Path(self.view.ask("Root repo pipeline_grasppose", self.profile.pipeline_repo))
                .expanduser()
                .resolve()
            )
            socket = self.view.ask("Socket worker", str(repo / ".runtime/worker.sock"))
            self.save(replace(self.profile, pipeline_repo=str(repo), socket=socket))
        elif selected == "4":
            lab = self.view.ask(
                "MYARM_LAB (trống để dùng cấu hình run_web.sh)", self.profile.ros_lab
            )
            python = self.view.ask("MYARM_PY (Python của SDK ROS)", self.profile.ros_python)
            setup = self.view.ask("MYARM_ROS_SETUP (setup.bash)", self.profile.ros_setup)
            self.save(replace(self.profile, ros_lab=lab, ros_python=python, ros_setup=setup))

    def setup(self):
        self.view.say(
            "Chuẩn bị environment robot bằng script Conda hiện có, tách biệt environment perception."
        )
        if self.view.confirm("Chạy cài môi trường robot?"):
            self.runner.run(["bash", self.root / "scripts/setup-env.sh"])

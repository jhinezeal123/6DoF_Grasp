"""Default máy chủ dùng chung, import được bằng stdlib Python 3.8 trở lên."""

from pathlib import Path

# Nguồn DUY NHẤT cho thiết bị: symlink cấp người dùng ~/myarm/tty -> /dev/ttyACM0.
# Đổi tay thì sửa symlink, không sửa code; spec.py và operator/config.py import từ đây.
DEFAULT_PORT = str(Path("~/myarm/tty").expanduser())
DEFAULT_BAUDRATE = 1_000_000
DEFAULT_CAMERA = "/dev/v4l/by-id/usb-046d_Logitech_Webcam_C925e_3F4C8F2F-video-index0"

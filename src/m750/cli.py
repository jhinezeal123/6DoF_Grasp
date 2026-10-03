"""cli.py - Entry points tien ich (cai qua pyproject [project.scripts]).

    m750-state     # doc trang thai tay (mo port, doc goc, in)
    m750-camera    # stream MJPEG http://<ip>:8080/ (tat: touch stop_cam hoac Ctrl-C)
    m750-preview   # web xem truoc mo phong http://<ip>:8081/
"""
from __future__ import annotations

import os
import signal
import threading


def state_main() -> int:
    """Doc va in trang thai myArm M750 (6 khop, gripper, pose URDF, fw coords)."""
    from .robot.control import ArmController

    ArmController().print_state()
    return 0


def camera_main() -> int:
    """Stream MJPEG camera. Tat bang file stop_cam (hoac Ctrl-C)."""
    from .camera import MjpegStream

    # Thay cho input(): terminal van dung duoc, tat qua file co
    STOP = os.path.join(os.getcwd(), "stop_cam")
    if os.path.exists(STOP):
        os.remove(STOP)
    MjpegStream().start()
    print("tat bang: touch %s   (hoac pkill -f 'm750-camera')" % STOP)
    stop = threading.Event()

    def _sigint(_sig, _frm):
        stop.set()

    signal.signal(signal.SIGINT, _sigint)
    while not os.path.exists(STOP) and not stop.is_set():
        stop.wait(0.3)
    print("thay tin hieu dung -> tat")
    return 0


def preview_main() -> int:
    """Web xem truoc mo phong; giu tien trinh song, atexit tu don khi thoat."""
    from .preview import PreviewServer

    PreviewServer().start()
    threading.Event().wait()  # giu tien trinh song; atexit tu don khi thoat
    return 0


__all__ = ["state_main", "camera_main", "preview_main"]

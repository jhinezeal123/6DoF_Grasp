"""Render HTML thuần; template và logic robot được đọc ở hai file riêng."""

import json
import re
from pathlib import Path

import numpy as np

_TEMPLATE = Path(__file__).with_name("control.html").read_text(encoding="utf-8")
_PLACEHOLDER = re.compile(r"@@M750_\d+@@")


def render_html(robot, usb_cameras):
    """Tao giao dien Web HTML/CSS/JS hien dai voi slider, camera stream, va buttons."""
    joint_names = robot.JOINT_NAMES
    rad_min = robot.rad_min
    rad_max = robot.rad_max
    deg_min = np.degrees(rad_min).tolist()
    deg_max = np.degrees(rad_max).tolist()

    # Khung camera USB: sinh tu usb_cameras de danh sach thiet bi chi duoc
    # khai bao o MOT cho (program/camera/usb_camera.py).
    usb_windows = "".join(
        '<div class="usb-window">'
        '<img src="/stream/cam_%s.mjpg" alt="%s">'
        '<div class="usb-label">%s</div>'
        "</div>" % (key, key, device)
        for key, device in usb_cameras
    )

    values = {
        "@@M750_0@@": usb_windows,
        "@@M750_1@@": json.dumps(joint_names),
        "@@M750_2@@": json.dumps(deg_min),
        "@@M750_3@@": json.dumps(deg_max),
    }
    html = _PLACEHOLDER.sub(lambda match: str(values[match.group()]), _TEMPLATE)
    return html.encode("utf-8")

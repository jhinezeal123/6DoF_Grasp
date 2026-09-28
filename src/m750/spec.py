"""spec.py - Thong so myArm M750: MOT NGUON SU THAT cho ca repo.

Truoc day cac hang so nay roi rac o 2-3 noi (ductocbatdat.py, ros_bridge,
test_2_vlm) - doi mot so phai tim nhieu cho. Bay gio tat ca import tu day.

Chay tren server ktmt (port /dev/ttyACM1, baud 1000000 - mac dinh 115200 sai).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from importlib.resources import files

# baud mac dinh 115200 -> khong noi duoc voi tay
DEFAULT_PORT = "/dev/ttyACM1"
DEFAULT_BAUDRATE = 1_000_000
# by-id: so video* doi theo thu tu cam
DEFAULT_CAMERA = "/dev/v4l/by-id/usb-046d_Logitech_Webcam_C925e_3F4C8F2F-video-index0"

# gioi han firmware, khac URDF (q2=100, q3=-100)
FW_MIN_DEG = (-165.0, -80.0, -100.0, -160.0, -90.0, -180.0)
# firmware chan ca goi lenh neu vuot, khong bao loi
FW_MAX_DEG = (165.0, 100.0, 80.0, 160.0, 120.0, 180.0)
# kep sat gioi han -> servo gong, dung yen (do that: q2 kep 100 dung, 99 chay)
FW_SAFE_MARGIN_DEG = 1.5

# mm: tool0 -> tam gap (gripper_base_link), lay tu URDF chu khong do firmware.
# Huong ra ngoai cua tool la -z (flange o +118, tool0 o 0, dau ngon o +74.5..+99.5).
# tests/test_gripper_pose.py kiem tra lai hang so nay voi URDF.
GRIP_L_MM = 87.0

JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_flex_joint",
    "forearm_roll_joint",
    "wrist_flex_joint",
    "wrist_roll_joint",
)

# URDF la nguon gioi han chinh - tai lieu lab muc 7.5 (get_joint_max() khong dang tin).
URDF_PATH = files("m750.model").joinpath("myarm_m750_full.urdf")


def model_dir() -> Path:
    """Thu muc model (URDF/MJCF/scene/meshes) - dung cho cau hinh MuJoCo."""
    return Path(str(files("m750.model")))


@dataclass(frozen=True)
class RobotSpec:
    """Cau hinh mot con tay M750: port, camera, gioi han, duong dan URDF.

    Frozen: doi thong so = tao spec moi, khong ai sua bien toan cuc giua chung.
    """

    port: str = DEFAULT_PORT
    baudrate: int = DEFAULT_BAUDRATE
    camera_device: str = DEFAULT_CAMERA
    fw_min_deg: tuple = FW_MIN_DEG
    fw_max_deg: tuple = FW_MAX_DEG
    margin_deg: float = FW_SAFE_MARGIN_DEG
    grip_l_mm: float = GRIP_L_MM
    # URDF nam trong package (m750/model) -> chay duoc sau pip install -e .
    urdf_path: Path = URDF_PATH


__all__ = [
    "DEFAULT_PORT",
    "DEFAULT_BAUDRATE",
    "DEFAULT_CAMERA",
    "FW_MIN_DEG",
    "FW_MAX_DEG",
    "FW_SAFE_MARGIN_DEG",
    "GRIP_L_MM",
    "JOINT_NAMES",
    "URDF_PATH",
    "RobotSpec",
    "model_dir",
]

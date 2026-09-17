"""
ros_bridge.py - Client ROS 2 duy nhat cho myArm M750 + camera C925e.

Moi dieu khien robot/camera di qua stack ROS 2 cua MyArmM750_Controller_Lab.
Lop nay KHONG tu mo serial, KHONG tu mo V4L, KHONG tu tinh FK/IK, KHONG giu
bang offset rieng: tat ca da nam trong driver/kinematics cua stack do.

Device co dinh -> hardcode ngay trong file, khong truyen qua tham so.

YEU CAU: phai source ROS 2 truoc khi dung (xem SDK/run_ros.sh).
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import rclpy
from diagnostic_msgs.msg import DiagnosticArray
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64, String
from std_srvs.srv import Trigger

# --------------------------------------------------------------- device co dinh
ROS_DOMAIN_ID = 10
SERIAL_PORT = "/dev/ttyACM1"
CAMERA_DEVICE = "/dev/v4l/by-id/usb-046d_Logitech_Webcam_C925e_3F4C8F2F-video-index0"

JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_flex_joint",
    "forearm_roll_joint",
    "wrist_flex_joint",
    "wrist_roll_joint",
)

# ------------------------------------------- ten tren ROS graph (services.yaml)
TOPIC_JOINT_STATE = "/myarm/state/joint_state"
TOPIC_JOINT_GOAL = "/myarm/command/joint_goal"
TOPIC_TCP_STATE = "/myarm/state/tcp_pose"
TOPIC_TCP_GOAL = "/myarm/command/tcp_pose"
TOPIC_GRIPPER_COMMAND = "/myarm/gripper/command"
TOPIC_GRIPPER_STATE = "/myarm/gripper/state"
TOPIC_SAFETY_STATE = "/myarm/robot/safety_state"
TOPIC_DIAGNOSTICS = "/myarm/robot/diagnostics"
TOPIC_MOTION_DIAGNOSTICS = "/myarm/motion_execution/diagnostics"
TOPIC_ROBOT_DESCRIPTION = "/robot_description"
TOPIC_CAMERA_IMAGE = "/myarm/cameras/cam01/image_raw"

SERVICE_STOP = "/myarm/robot/stop"
SERVICE_REARM = "/myarm/robot/rearm"
SERVICE_POWER_ON = "/myarm/robot/power_on"
SERVICE_POWER_OFF = "/myarm/robot/power_off"

# myarm_motion_execution chot fault RIENG cua no khi stop, doc lap voi
# safety_state cua driver: "joint goal rejected; call reset after fault".
# Phai goi CA HAI thi moi chay lai duoc.
SERVICE_MOTION_RESET = "/myarm/motion_execution/reset"

# Do mo TONG giua hai dau ngon tay (met). URDF: left_gripper_joint 0..0.04 va
# hai ngon doi xung -> tong 0..0.08. Day la quy uoc cua stack ROS, KHAC quy uoc
# "toa do MOT ngon" 0..0.0345 cua SDK cu.
MAX_OPENING_M = 0.08

# ------------------------------------------- hieu chuan phan cung (NGUON DUY NHAT)
# Quy uoc: raw = model + OFFSETS_DEG  (giong myarm_m750_robot_arm*.yaml cua lab).
# Cac so nay do bang cach doi chieu render MuJoCo voi anh chup robot that.
#
# CANH BAO: config cua lab (myarm_m750_robot_arm_acm0.yaml) tung ghi
# [0, 10, -10, 0, 0, 0] - bo so do KHONG duoc do, ma giai nguoc tu
# "offset = gioi han firmware - gioi han URDF". No lam FK/IK cua myarm_kinematics
# sai.
#
# OFFSETS_DEG LA NGUON DUY NHAT. _OFFSETS / MODEL_MIN_RAD / MODEL_MAX_RAD deu
# duoc SUY RA tu no ben trong _apply_offsets_deg(), khong phai ban sao doc lap:
# sua mot lan la ca tien trinh doi theo. Gia tri hieu chuan luu o offsets.json
# (canh file nay) va doc lai luc khoi dong - xem save_offsets_deg().
DEFAULT_OFFSETS_DEG = (-10.1, 28.58, -7.42, 2.1, -21.19, 2.54)
OFFSETS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "offsets.json")

# Gioi han goc RAW cua firmware MyArm M750 (do 2026-09 bang MyArmMControl).
# KHAC URDF: q2 toi 100 (khong phai 90), q3 lui toi -100 (khong phai -90).
# Firmware CHAN CA GOI lenh neu BAT KY khop nao vuot gioi han, va no khong bao
# loi -> robot dung yen trong khi lenh van "gui thanh cong".
FW_MIN_DEG = (-165.0, -80.0, -100.0, -160.0, -90.0, -180.0)
FW_MAX_DEG = (165.0, 100.0, 80.0, 160.0, 120.0, 180.0)

# Lui vao trong khi kep gioi han. Kep SAT bang gioi han thi lenh duoc nhan nhung
# tay khong nhuc nhich (servo gong lai cham co) - da do tren may that: khop 2
# nghi o 106 do, gioi han 100; kep ve 100 -> dung yen, kep ve 99 -> chay.
FW_SAFE_MARGIN_DEG = 1.5

_FW_MIN = np.array(FW_MIN_DEG, dtype=np.float64)
_FW_MAX = np.array(FW_MAX_DEG, dtype=np.float64)

# Gia tri khoi tao chi de cho bien co mat TRUOC khi _apply_offsets_deg chay o
# cuoi khoi nay. Khong dung truc tiep - moi thu doc qua _apply_offsets_deg().
OFFSETS_DEG = DEFAULT_OFFSETS_DEG
_OFFSETS = np.array(DEFAULT_OFFSETS_DEG, dtype=np.float64)
MODEL_MIN_RAD = np.zeros(6)
MODEL_MAX_RAD = np.zeros(6)


def _apply_offsets_deg(values) -> None:
    """Ap dung offset moi va dung lai MOI dai luong dan xuat tu no.

    Day la cho duy nhat duoc phep gan OFFSETS_DEG / _OFFSETS / MODEL_MIN_RAD /
    MODEL_MAX_RAD. Khong noi nao khac duoc tu tinh lai offset: do la cach hai ban
    sao lech nhau roi khong ai biet ban nao dung.
    """
    global OFFSETS_DEG, _OFFSETS, MODEL_MIN_RAD, MODEL_MAX_RAD
    OFFSETS_DEG = tuple(float(v) for v in values)
    _OFFSETS = np.array(OFFSETS_DEG, dtype=np.float64)
    # Gioi han firmware doi sang khong gian model de con so sanh duoc voi
    # /myarm/state/joint_state va /robot_description.
    MODEL_MIN_RAD = np.radians(_FW_MIN + FW_SAFE_MARGIN_DEG - _OFFSETS)
    MODEL_MAX_RAD = np.radians(_FW_MAX - FW_SAFE_MARGIN_DEG - _OFFSETS)


MAX_ABS_OFFSET_DEG = 180.0


def save_offsets_deg(values) -> Tuple[float, ...]:
    """Ghi offset hieu chuan ra offsets.json roi ap dung ngay cho tien trinh nay.

    Chan offset vo ly o day vi day la bien tin cay: con so den tu thao tac tay cua
    nguoi dung tren UI, duoc ghi xuong dia, va tu do ve lai con robot ao
    (FakeRobot.sync -> model_rad) cung nhu mien keo cua slider.

    LUU Y: KHONG the kiem tra bang "gioi han co dao nghich khong" - hieu
    MODEL_MAX - MODEL_MIN = FW_MAX - FW_MIN - 2*FW_SAFE_MARGIN, khong he phu thuoc
    offset, nen phep kiem do khong bao gio ngoi no (da thu: offset 1e6 do van
    lot). Chi con cach chan theo do lon vat ly.
    """
    values = tuple(float(v) for v in values)
    if len(values) != 6 or not all(np.isfinite(values)):
        raise ValueError("offset phai gom 6 so huu han")
    if max(abs(v) for v in values) > MAX_ABS_OFFSET_DEG:
        raise ValueError(
            "offset vuot qua %.0f do -> gan nhu chac chan la hieu chuan sai"
            % MAX_ABS_OFFSET_DEG)
    _apply_offsets_deg(values)
    with open(OFFSETS_PATH, "w", encoding="utf-8") as fh:
        json.dump({"offsets_deg": list(OFFSETS_DEG)}, fh, indent=2)
    return OFFSETS_DEG


def _saved_offsets_deg():
    try:
        with open(OFFSETS_PATH, encoding="utf-8") as fh:
            return json.load(fh)["offsets_deg"]
    except Exception:
        # Chua hieu chuan lan nao, hoac file hong -> dung so mac dinh.
        return DEFAULT_OFFSETS_DEG


def raw_deg(q_model) -> np.ndarray:
    """Doi goc model (radian) sang goc RAW (do) ma firmware thuc su nhin thay.

    Doi chieu: sim -> real. Dung o hai cho:
      - doc/doi chieu khi can biet firmware dang thay so nao;
      - apply_fake_pose_to_real() tren UI: pose MuJoCo dang ve la "sim", muon tay
        that ve dung cho NHIN THAY thi phai cong offset tra lai truoc khi gui lenh.

    Duong lenh THUONG (keo slider) KHONG dung ham nay: stack lab nhan model space
    tren /myarm/command/joint_goal va tu doi sang hardware truoc khi ghi xuong servo.
    """
    return np.degrees(np.asarray(q_model, dtype=np.float64)) + _OFFSETS


def model_rad(q_raw) -> np.ndarray:
    """Doi goc topic (radian) sang goc model DA HIEU CHUAN. Nghich dao raw_deg.

    Dung o dung MOT cho: FakeRobot.sync() khi ve con robot ao. Con robot ao phai
    hien thi o goc da hieu chuan - neu ve thang gia tri topic thi nguoi dung
    nhin thay tay ao lech so voi thuc te dung bang OFFSETS_DEG, va do chinh la
    thu che do OFFSET tren UI dung de sua.
    """
    return np.asarray(q_raw, dtype=np.float64) - np.radians(_OFFSETS)


_apply_offsets_deg(_saved_offsets_deg())


# QoS cua /myarm/robot/safety_state la TRANSIENT_LOCAL: phai khop durability,
# neu khong se khong nhan duoc gi (driver chi publish khi trang thai doi).
_SAFETY_QOS = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
_LATCHED_QOS = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)


def _urdf_joint_limits(xml_text: str) -> Dict[str, Tuple[float, float]]:
    """Doc gioi han khop tu URDF. Mot nguon su that, khong hardcode lai."""
    import xml.etree.ElementTree as ET

    limits: Dict[str, Tuple[float, float]] = {}
    for joint in ET.fromstring(xml_text).iter("joint"):
        name = joint.get("name")
        limit = joint.find("limit")
        if name is None or limit is None:
            continue
        lower, upper = limit.get("lower"), limit.get("upper")
        if lower is not None and upper is not None:
            limits[name] = (float(lower), float(upper))
    return limits


class RosBridge:
    """
    Mot node ROS 2 duy nhat dung chung cho Robot / FakeRobot / Camera.

    Chi doc trang thai moi nhat va gui lenh. Khong tu mo phong, khong tu noi suy,
    khong tu tinh dong hoc: moi thu do deu la viec cua stack ROS 2.
    """

    def __init__(self, node_name: str = "myarm_sdk_bridge") -> None:
        if not rclpy.ok():
            os.environ.setdefault("ROS_DOMAIN_ID", str(ROS_DOMAIN_ID))
            rclpy.init(args=None)

        self.node = Node(node_name)
        self._lock = threading.RLock()

        # --- cache trang thai moi nhat, do callback ghi, moi thread doc ---
        self._joint_rad: Optional[np.ndarray] = None
        self._joint_t = 0.0
        self._tcp_pos: Optional[np.ndarray] = None
        self._tcp_quat: Optional[np.ndarray] = None
        self._opening_m: Optional[float] = None
        self._safety_state = "unknown"
        self._safety_epoch = 0
        self._temperatures: Optional[Tuple[float, ...]] = None
        self._motion_state = ""
        self._motion_detail = ""
        self._motion_seq = 0
        self._limits: Optional[List[Tuple[float, float]]] = None

        self.node.create_subscription(
            JointState, TOPIC_JOINT_STATE, self._on_joint_state, 10
        )
        self.node.create_subscription(PoseStamped, TOPIC_TCP_STATE, self._on_tcp, 10)
        self.node.create_subscription(
            JointState, TOPIC_GRIPPER_STATE, self._on_gripper, 10
        )
        self.node.create_subscription(
            String, TOPIC_SAFETY_STATE, self._on_safety, _SAFETY_QOS
        )
        self.node.create_subscription(
            DiagnosticArray, TOPIC_DIAGNOSTICS, self._on_diagnostics, 10
        )
        self.node.create_subscription(
            DiagnosticArray,
            TOPIC_MOTION_DIAGNOSTICS,
            self._on_motion_diagnostics,
            10,
        )
        self.node.create_subscription(
            String, TOPIC_ROBOT_DESCRIPTION, self._on_description, _LATCHED_QOS
        )

        self._pub_joint = self.node.create_publisher(JointState, TOPIC_JOINT_GOAL, 10)
        self._pub_tcp = self.node.create_publisher(PoseStamped, TOPIC_TCP_GOAL, 10)
        self._pub_gripper = self.node.create_publisher(
            Float64, TOPIC_GRIPPER_COMMAND, 10
        )
        self._clients = {
            name: self.node.create_client(Trigger, name)
            for name in (
                SERVICE_STOP,
                SERVICE_REARM,
                SERVICE_POWER_ON,
                SERVICE_POWER_OFF,
                SERVICE_MOTION_RESET,
            )
        }

        self._thread = threading.Thread(
            target=rclpy.spin, args=(self.node,), name="RosBridgeSpin", daemon=True
        )
        self._thread.start()

    # ------------------------------------------------------------------ callback
    def _on_joint_state(self, message: JointState) -> None:
        by_name = dict(zip(message.name, message.position))
        try:
            values = np.array([by_name[n] for n in JOINT_NAMES], dtype=np.float64)
        except KeyError:
            return  # feedback chua du 6 khop -> bo qua, khong doan
        with self._lock:
            self._joint_rad = values
            self._joint_t = time.monotonic()

    def _on_tcp(self, message: PoseStamped) -> None:
        p, q = message.pose.position, message.pose.orientation
        with self._lock:
            self._tcp_pos = np.array([p.x, p.y, p.z], dtype=np.float64)
            self._tcp_quat = np.array([q.x, q.y, q.z, q.w], dtype=np.float64)

    def _on_gripper(self, message: JointState) -> None:
        if not message.position:
            return
        with self._lock:
            # /myarm/gripper/state mang toa do MOT ngon -> tong = 2x.
            self._opening_m = float(message.position[0]) * 2.0

    def _on_safety(self, message: String) -> None:
        # Driver phat chuoi dang "state=armed;epoch=1;reason=operator_rearm".
        fields = dict(
            token.split("=", 1)
            for token in message.data.replace(";", " ").split()
            if "=" in token
        )
        with self._lock:
            self._safety_state = fields.get("state", "unknown")
            try:
                self._safety_epoch = int(fields.get("epoch", 0))
            except ValueError:
                pass

    def _on_diagnostics(self, message: DiagnosticArray) -> None:
        # Driver publish nhiet do servo duoi dang temperature_j1..j6 trong
        # /myarm/robot/diagnostics. Chi lay khi DU 6 gia tri, neu thieu thi giu
        # gia tri cu (driver cu chua co khoa nay -> tra None, khong phai loi).
        found: Dict[int, float] = {}
        for status in message.status:
            for entry in status.values:
                key = entry.key
                if not key.startswith("temperature_j"):
                    continue
                try:
                    index = int(key[len("temperature_j"):])
                    found[index] = float(entry.value)
                except ValueError:
                    continue
        if len(found) != 6:
            return
        with self._lock:
            self._temperatures = tuple(found[i] for i in range(1, 7))

    def _on_motion_diagnostics(self, message: DiagnosticArray) -> None:
        # myarm_motion_execution chi nhan MOT lenh tai mot thoi diem; goi moi
        # trong luc dang chay bi tu choi va no bao qua day:
        #   state=succeeded|running|rejected, detail="joint goal rejected ..."
        # Day la tin hieu DUY NHAT cho biet lenh co duoc nhan hay khong - publish
        # luon thanh cong nen khong the biet qua gia tri tra ve.
        for status in message.status:
            fields = {entry.key: entry.value for entry in status.values}
            with self._lock:
                self._motion_state = fields.get("state", "")
                self._motion_detail = fields.get("detail", "")
                self._motion_seq += 1

    def _on_description(self, message: String) -> None:
        try:
            limits = _urdf_joint_limits(message.data)
        except Exception:
            return
        if all(name in limits for name in JOINT_NAMES):
            with self._lock:
                self._limits = [limits[name] for name in JOINT_NAMES]

    # -------------------------------------------------------------------- trang thai
    def wait_for_state(self, timeout_s: float = 5.0) -> bool:
        """Cho feedback do dau tien. False neu het timeout (stack chua chay?)."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                if self._joint_rad is not None:
                    return True
            time.sleep(0.02)
        return False

    def wait_for_tcp(self, timeout_s: float = 5.0) -> bool:
        """Cho myarm_kinematics cong bo TCP pose dau tien (no chi phat sau khi co FK)."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                if self._tcp_pos is not None:
                    return True
            time.sleep(0.02)
        return False

    @property
    def joint_rad(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._joint_rad is None else self._joint_rad.copy()

    @property
    def joint_age_s(self) -> float:
        with self._lock:
            if self._joint_rad is None:
                return float("inf")
            return time.monotonic() - self._joint_t

    @property
    def tcp_pos(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._tcp_pos is None else self._tcp_pos.copy()

    @property
    def tcp_quat(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._tcp_quat is None else self._tcp_quat.copy()

    @property
    def opening_m(self) -> Optional[float]:
        with self._lock:
            return self._opening_m

    @property
    def safety_state(self) -> str:
        with self._lock:
            return self._safety_state

    @property
    def is_armed(self) -> bool:
        return self.safety_state == "armed"

    @property
    def temperatures(self) -> Optional[Tuple[float, ...]]:
        """Nhiet do 6 servo (do C), do driver doc tu get_servo_temps().

        None khi driver chua cong bo (driver cu, hoac backend fake_robot_arm
        khong co phan cung de doc). Khi do UI hien '--' chu khong phai 0.
        """
        with self._lock:
            return self._temperatures

    @property
    def motion_state(self) -> Tuple[str, str, int]:
        """(state, detail, seq) cua myarm_motion_execution.

        state: "succeeded" | "running" | "rejected" | "" (chua co tin hieu).
        seq tang moi lan co diagnostics moi, de ben goi biet da co cap nhat.
        """
        with self._lock:
            return self._motion_state, self._motion_detail, self._motion_seq

    @property
    def joint_limits(self) -> List[Tuple[float, float]]:
        """
        Giao cua gioi han URDF va duong bao firmware.

        Driver chi validate theo URDF, nen mot dich nam trong URDF van co the
        vuot gioi han firmware -> firmware huy NGUYEN goi lenh va robot dung yen
        ma khong bao loi. Giao nay moi la mien thuc su gui duoc.
        """
        with self._lock:
            urdf = self._limits
        if urdf is None:
            urdf = [(-np.pi, np.pi)] * 6
        return [
            (max(lo, fw_lo), min(hi, fw_hi))
            for (lo, hi), fw_lo, fw_hi in zip(urdf, MODEL_MIN_RAD, MODEL_MAX_RAD)
        ]

    # -------------------------------------------------------------------- lenh
    def send_joint_goal(self, q_rad) -> bool:
        """Gui dich 6 khop (radian, thu tu canonical) cho myarm_motion_execution."""
        values = np.asarray(q_rad, dtype=np.float64).flatten()
        if values.size != 6 or not np.all(np.isfinite(values)):
            return False
        message = JointState()
        message.header.stamp = self.node.get_clock().now().to_msg()
        message.name = list(JOINT_NAMES)
        message.position = [float(v) for v in values]
        self._pub_joint.publish(message)
        return True

    def send_tcp_pose(self, position, quaternion) -> bool:
        """Gui dich TCP (base_link) cho myarm_kinematics; ket qua di qua IK + executor."""
        message = PoseStamped()
        message.header.stamp = self.node.get_clock().now().to_msg()
        message.header.frame_id = "base_link"
        message.pose.position.x = float(position[0])
        message.pose.position.y = float(position[1])
        message.pose.position.z = float(position[2])
        message.pose.orientation.x = float(quaternion[0])
        message.pose.orientation.y = float(quaternion[1])
        message.pose.orientation.z = float(quaternion[2])
        message.pose.orientation.w = float(quaternion[3])
        self._pub_tcp.publish(message)
        return True

    def set_opening(self, opening_m: float) -> bool:
        """Gui do mo TONG giua hai dau ngon (met, 0..0.08)."""
        value = float(np.clip(opening_m, 0.0, MAX_OPENING_M))
        message = Float64()
        message.data = value
        self._pub_gripper.publish(message)
        return True

    def call_trigger(self, service: str, timeout_s: float = 3.0):
        """Goi service std_srvs/Trigger. Tra (thanh_cong, thong_diep)."""
        client = self._clients[service]
        if not client.wait_for_service(timeout_sec=timeout_s):
            return False, "service {} khong co".format(service)
        future = client.call_async(Trigger.Request())
        done = threading.Event()
        future.add_done_callback(lambda _: done.set())
        if not done.wait(timeout_s):
            return False, "het thoi gian cho {}".format(service)
        try:
            response = future.result()
        except Exception as exc:  # noqa: BLE001 - bao loi that ra ngoai
            return False, str(exc)
        return bool(response.success), str(response.message)

    def close(self) -> None:
        try:
            self.node.destroy_node()
        except Exception:
            pass


_bridge: Optional[RosBridge] = None
_bridge_lock = threading.Lock()


def get_bridge() -> RosBridge:
    """Bridge dung chung cho ca tien trinh: rclpy.init chi duoc goi mot lan."""
    global _bridge
    with _bridge_lock:
        if _bridge is None:
            _bridge = RosBridge()
        return _bridge


__all__ = [
    "CAMERA_DEVICE",
    "DEFAULT_OFFSETS_DEG",
    "FW_MAX_DEG",
    "FW_MIN_DEG",
    "FW_SAFE_MARGIN_DEG",
    "JOINT_NAMES",
    "MAX_OPENING_M",
    "MODEL_MAX_RAD",
    "MODEL_MIN_RAD",
    "OFFSETS_DEG",
    "OFFSETS_PATH",
    "ROS_DOMAIN_ID",
    "RosBridge",
    "SERIAL_PORT",
    "TOPIC_CAMERA_IMAGE",
    "get_bridge",
    "model_rad",
    "raw_deg",
    "save_offsets_deg",
]

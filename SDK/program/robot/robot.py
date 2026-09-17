"""
robot/robot.py - Lop Robot dai dien cho myArm M750 THAT.

Lop nay KHONG dung pymycobot: moi lenh va moi feedback deu di qua stack ROS 2
(myarm_robot_driver + myarm_kinematics + myarm_motion_execution).

Nhung pymycobot VAN DUOC PHEP va dang duoc dung o cac tang khac - cau "khong con
pymycobot" o ban truoc la sai:
  - driver noi voi servo qua serial bang plugin_adapter/robot_arm
    (myarm_m750_robot_arm.yaml cua lab);
  - run_web.sh bat nguon servo bang pymycobot TRUOC khi launch driver, vi driver
    khong khoi dong duoc khi servo mat dien;
  - module moi duoc phep goi pymycobot truc tiep khi can.

  - Doc trang thai  : /myarm/state/joint_state, /myarm/state/tcp_pose
  - Gui dich khong  : /myarm/command/joint_goal, /myarm/command/tcp_pose
  - Tay kep         : /myarm/gripper/command
  - Vong doi an toan: /myarm/robot/{power_on,power_off,rearm,stop}

Serial va camera la co dinh: xem ros_bridge.py, khong truyen qua tham so.

QUY UOC TAY KEP: `gripper` / MAX_GRIPPER_M nay la do mo TONG giua hai dau ngon
tay (met, 0..0.08) theo URDF cua stack ROS. SDK cu dung toa do MOT ngon
(0..0.0345); gia tri cu gap doi moi bang gia tri moi.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Union

import numpy as np

from program.ros_bridge import (
    JOINT_NAMES,
    MAX_OPENING_M,
    SERVICE_POWER_OFF,
    SERVICE_POWER_ON,
    SERVICE_REARM,
    SERVICE_STOP,
    SERVICE_MOTION_RESET,
    get_bridge,
    raw_deg,
)

GRIPPER_KEYS = ("gripper", "grip", "pos_left_gripper")


class Robot:
    """
    Mat na (facade) cho myArm M750 that.

    Moi lenh deu la mot dich 6 khop gui cho myarm_motion_execution; viec noi suy,
    kiem tra gioi han va an toan la cua stack ROS. Vi vay khong con trang thai
    `_target_rad` song song de lech pha voi encoder.
    """

    JOINT_NAMES = list(JOINT_NAMES)
    MAX_GRIPPER_M = MAX_OPENING_M
    # Dung sai "toi noi" (radian). Do tren myArm M750 THAT: khop duoc lenh di
    # chuyen dung lai van con lech ~1.1 deg (0.019 rad) - sai so tich luy cua
    # servo duoi trong luc. Gia tri cu 0.02 rad chi hon dung sai do 0.004 rad,
    # nen ket qua phu thuoc vao may man. 0.05 rad (2.9 deg) = ~2.5 lan sai so do.
    SETTLE_TOL_RAD = 0.05
    # Driver phat feedback khop o 5 Hz. Ngay sau khi executor bao xong, ban tin
    # moi nhat van co the la cua nhip TRUOC, nen phai cho qua mot nhip roi moi
    # doc qpos lam goc dung dich.
    FEEDBACK_SETTLE_S = 0.35
    # Feedback khop phat o 5 Hz. Qua ngan nay khong co ban tin moi thi coi nhu
    # MAT feedback (rut cap / mat nguon / driver dung) - khong duoc doan pose.
    FEEDBACK_STALE_S = 1.0

    def __init__(self) -> None:
        self._b = get_bridge()
        # Seq cua ban diagnostics luc gui lenh gan nhat. Dung de biet trang thai
        # doc duoc la MOI hay van la cua lenh truoc (xem _wait_until_ready).
        self._send_seq = -1
        # Da tung gui lenh nao chua. Truoc lan gui dau tien thi _cmd (toan 0)
        # khong phai la dich that, nen khong duoc dung no lam moc hoi tu.
        self._sent = False
        # Chi so cac khop ma lenh vua roi THUC SU yeu cau di chuyen. Khop giu
        # nguyen vi tri khong the "toi noi" theo nghia nao (xem wait_until_at).
        self._moving = None

    # ------------------------------------------------------------------ gioi han
    @property
    def rad_min(self) -> np.ndarray:
        return np.array([lo for lo, _ in self._b.joint_limits], dtype=np.float64)

    @property
    def rad_max(self) -> np.ndarray:
        return np.array([hi for _, hi in self._b.joint_limits], dtype=np.float64)

    # ------------------------------------------------------------------ trang thai
    @property
    def is_real_connected(self) -> bool:
        """Driver co dang phat feedback moi khong (tre qua 0.5 s = mat ket noi)."""
        return self._b.joint_age_s < 0.5

    @property
    def safety_state(self) -> str:
        """disarmed | armed | stopping | fault_latched (xem myarm_robot_driver)."""
        return self._b.safety_state

    @property
    def is_armed(self) -> bool:
        """Chi khi armed thi executor moi nhan duoc quy dao."""
        return self._b.is_armed

    def wait_until_online(self, timeout_s: float = 5.0) -> bool:
        """
        Cho stack ROS san sang: feedback khop va TCP pose dau tien.

        TCP pose do myarm_kinematics phat, nen no tre hon feedback khop vai nhip.
        """
        if not self._b.wait_for_state(timeout_s):
            return False
        self._b.wait_for_tcp(min(timeout_s, 3.0))
        return True

    @property
    def qpos(self) -> np.ndarray:
        """Goc khop DO DUOC (radian) tu /myarm/state/joint_state.

        Tra nan khi chua co feedback hoac feedback qua cu - KHONG tra 0.

        LOI THAT (phat hien khi robot mat nguon giua bai test FollowTrajectory):
        truoc day tra zeros(6) khi thieu feedback, tuc la bao "tay dang o pose
        zero". Moi lenh tuong doi (set_joint/change/FollowTrajectory) deu lay
        qpos lam goc, nen mot lenh cho j2 bien thanh "dua CA 6 khop ve 0" -
        dung luc tay da tut xuong vi mat mo-men. Tra nan thi _go() TU CHOI
        (xem _go), con tcp_pos/temperatures trong file nay cung da theo quy uoc
        nan tu truoc.
        """
        q = self._b.joint_rad
        if q is None or self._b.joint_age_s > self.FEEDBACK_STALE_S:
            return np.full(6, np.nan)
        return q

    # Giu ten cu: truoc day la cache serial, nay feedback da duoc driver cache o 5 Hz.
    qpos_cached = qpos

    @property
    def ctrl(self) -> np.ndarray:
        """Dich 6 khop da gui gan nhat (radian). Khong phai gia tri do duoc."""
        return self._cmd.copy()

    @property
    def tcp_pos(self) -> np.ndarray:
        """Toa do TCP [X, Y, Z] (met) do myarm_kinematics tinh FK. nan neu chua co."""
        p = self._b.tcp_pos
        return np.full(3, np.nan) if p is None else p

    @property
    def tcp_quat(self) -> np.ndarray:
        """Huong TCP theo quaternion ROS [qx, qy, qz, qw]."""
        q = self._b.tcp_quat
        return np.full(4, np.nan) if q is None else q

    @property
    def gripper(self) -> float:
        """Do mo TONG giua hai dau ngon tay (met, 0..0.08)."""
        value = self._b.opening_m
        return float(self._cmd_grip if value is None else value)

    @property
    def gripper_pct(self) -> float:
        return float(np.clip(self.gripper / self.MAX_GRIPPER_M * 100.0, 0.0, 100.0))

    def temperatures(self) -> np.ndarray:
        """Nhiet do 6 servo (do C), do myarm_robot_driver doc tu get_servo_temps().

        Tra nan cho tung khop khi driver chua cong bo nhiet do (driver cu, hoac
        backend fake_robot_arm khong co phan cung). Servo qua nhiet thi tu ngat
        va TU CHOI lenh, nen day la cach biet vi sao robot khong nhuc nhich.

        Dung nan chu KHONG phai 0: 0 do C la gia tri that va se gay hieu nham.
        """
        values = self._b.temperatures
        if values is None:
            return np.full(6, np.nan)
        return np.asarray(values, dtype=np.float64)

    @property
    def distribution(self) -> Dict[str, Any]:
        return {
            "qpos": self.qpos,
            "ctrl": self.ctrl,
            "gripper": self.gripper,
            "gripper_pct": self.gripper_pct,
            "tcp_pos": self.tcp_pos,
            "tcp_quat": self.tcp_quat,
            "temperatures": self.temperatures(),
            "joint_names": list(self.JOINT_NAMES),
            "rad_max": self.rad_max,
            "rad_min": self.rad_min,
            "safety_state": self.safety_state,
            "is_simulation": False,
            "is_real_connected": self.is_real_connected,
        }

    distribute = distribution

    # ------------------------------------------------------------------ lenh
    def _resolve_joint_index(self, p: Union[int, str]) -> Optional[int]:
        if isinstance(p, int):
            return p if 0 <= p < 6 else None
        if isinstance(p, str):
            key = p.lower()
            if key.startswith("j") and key[1:].isdigit():
                idx = int(key[1:]) - 1
                return idx if 0 <= idx < 6 else None
            if key in self.JOINT_NAMES:
                return self.JOINT_NAMES.index(key)
        return None

    def _go(self, q_rad) -> bool:
        """
        Gui mot dich 6 khop va ghi nho gia tri da lenh.

        Kep vao mien gui duoc (URDF giao voi duong bao firmware) va IN CANH BAO
        khi phai kep, vi firmware huy nguyen goi lenh neu mot khop vuot gioi han
        ma khong he bao loi - im lang dung yen la kieu that bai te nhat.

        myarm_motion_execution CHI nhan mot lenh tai mot thoi diem: goi moi trong
        luc dang chay bi TU CHOI ("joint goal rejected while an execution is
        active"). Publish luon thanh cong nen khong the biet qua gia tri tra ve;
        phai doi tin hieu that tu /myarm/motion_execution/diagnostics. Neu khong
        cho, keo slider lien tuc se lam hau het lenh bi bo va tay di sai cho.
        """
        # Chot an toan CUOI CUNG truoc khi publish: dich phai huu han.
        #
        # Dich nan nghia la no duoc tinh tu qpos nan, tuc la tuong quan voi mot
        # pose KHONG doc duoc (xem qpos). Gui di la gui mot cu giat mu. Chan o
        # day vi moi lenh (set_joint/change/set_pose/FollowTrajectory) deu di
        # qua _go, con dich TUYET DOI tinh san (vd tu model VLA) van huu han nen
        # khong bi chan oan.
        requested = np.asarray(q_rad, dtype=np.float64).flatten()[:6]
        if requested.size != 6 or not np.all(np.isfinite(requested)):
            print("[Robot] TU CHOI lenh: dich khong huu han (khong co feedback "
                  "khop moi - mat nguon / rut cap / driver dung?).")
            return False

        if not self._wait_until_ready():
            print("[Robot] Executor chua san sang, bo qua lenh moi.")
            return False

        values = np.clip(requested, self.rad_min, self.rad_max)

        clamped = np.flatnonzero(np.abs(values - requested) > 1e-9)
        if clamped.size:
            raw = raw_deg(requested)
            print("[Robot] Kep vao gioi han firmware: " + ", ".join(
                "j{} {:.1f}->{:.1f} deg (raw {:.1f})".format(
                    i + 1, np.degrees(requested[i]), np.degrees(values[i]), raw[i])
                for i in clamped))

        # Ghi lai so thu tu diagnostics TRUOC khi gui: trang thai doc duoc sau do
        # chi dang tin neu seq da tang, neu khong thi dang doc trang thai CU.
        self._send_seq = self._b.motion_state[2]
        # Pose do duoc NGAY TRUOC khi gui: moc de biet khop nao thuc su phai di.
        measured = self.qpos
        if not np.all(np.isfinite(measured)):
            self._moving = np.arange(6)
        else:
            self._moving = np.flatnonzero(np.abs(values - measured) > self.SETTLE_TOL_RAD)
        if not self._b.send_joint_goal(values):
            return False
        self._cmd = values
        self._sent = True
        return True

    def _wait_until_ready(self, timeout_s: float = 15.0) -> bool:
        """
        Cho executor ranh VA feedback duoi kip lenh truoc, roi moi cho gui lenh moi.

        Hai dieu kien, ca hai deu bat buoc:

        1. myarm_motion_execution chi nhan MOT lenh tai mot thoi diem; goi moi
           trong luc dang chay bi tu choi ("joint goal rejected while an execution
           is active"). Publish luon thanh cong nen KHONG the biet qua gia tri tra
           ve. Trang thai that doc tu /myarm/motion_execution/diagnostics:
           "executing" (dang ban) / "succeeded" (ranh). Phai doi seq TANG, vi ban
           diagnostics ngay sau khi gui van mang trang thai CU -> tuong ranh roi
           gui de, lenh bi tu choi im lang.

        2. Feedback khop phat o 5 Hz nen sau khi executor bao xong, qpos van co the
           la gia tri CU. change()/set_joint() lay qpos lam goc cho cac khop khac,
           nen goc cu se keo cac khop do VE VI TRI CU. Da do: gui 6 lenh don dap
           thi moi lenh xoa lenh truoc, chi 3 khop di chuyen.

        CANH BAO: truong "detail" cua diagnostics dinh lai gia tri cu, KHONG dung
        no lam tin hieu.

        KHONG doi qpos hoi tu ve _cmd: tay dung sai khoang 1.5 do so voi lenh, lon
        hon SETTLE_TOL_RAD (1.15 do), nen phep cho do se khong bao gio dat va MOI
        lenh sau deu bi tu choi. Chi can cho qua mot nhip feedback.
        """
        # Driver khoi dong o trang thai "disarmed" (hardcode cho adapter that) va chi
        # ap setpoint khi da armed -> lenh DAU TIEN sau moi lan restart stack bi nuot
        # im lang. Tu arm, thay vi bat nguoi van hanh bam "Khoi phuc" moi lan.
        #
        # Chi arm khi dang "disarmed". "fault_latched" la hau qua cua DUNG KHAN, co y
        # de nguoi van hanh chu dong khoi phuc -> khong tu vuot qua.
        # "unknown" = chua nhan duoc topic safety; luc do cung phai arm, neu khong
        # lenh dau tien se bi nuot ma khong co tin hieu gi.
        if self._b.safety_state in ("disarmed", "unknown"):
            self.rearm()

        deadline = time.monotonic() + timeout_s
        reset_done = False
        while time.monotonic() < deadline:
            state, _detail, seq = self._b.motion_state

            # "holding"/"fault" la ngo cut cua executor: no TU CHOI moi lenh sau do
            # cho toi khi co reset. Tu go luon. Khong tu go thi nguoi dung phai bam
            # "Khoi phuc" sau MOI lan tay khong bam noi quy dao.
            #
            # An toan: DUNG KHAN chot o tang DRIVER (fault_latched), khong phai o day.
            # Reset executor khong mo duoc khoa do, nen dung khan van con hieu luc.
            if state in ("holding", "fault") and not reset_done:
                self._b.call_trigger(SERVICE_MOTION_RESET)
                reset_done = True
                time.sleep(0.2)
                continue

            # Chi IDLE/SUCCEEDED/CANCELED moi nhan lenh moi. Truoc day dieu kien chi
            # loai "executing", nen "holding"/"fault" lot qua -> publish thanh cong
            # nhung executor tu choi, va lenh bien mat KHONG mot tin hieu nao.
            if seq > self._send_seq and state not in ("executing", "holding", "fault"):
                if self._sent:
                    time.sleep(self.FEEDBACK_SETTLE_S)
                return True
            time.sleep(0.02)
        return False

    def change(self, p: Union[int, str], delta: float) -> bool:
        """
        Dich mot khop hoac gripper them mot luong delta.

        Luon tinh tu gia tri DO DUOC, nen khong con lech pha khi tay bi keo tay
        hoac bi script khac dieu khien (loi cu cua `_target_rad`).

        PHAI cho executor ranh TRUOC khi doc qpos: qpos la goc cho CA 6 khop, nen
        doc no khi con dang chay se dung goc cu va keo cac khop khac ve vi tri cu.
        Da do: gui 6 lenh don dap thi moi lenh xoa lenh truoc, chi 3 khop di chuyen.
        """
        if isinstance(p, str) and p.lower() in GRIPPER_KEYS:
            return self.set_opening(self.gripper + float(delta))

        idx = self._resolve_joint_index(p)
        if idx is None:
            return False
        if not self._wait_until_ready():
            print("[Robot] Executor chua san sang, bo qua lenh moi.")
            return False
        target = self.qpos.copy()
        target[idx] = float(np.clip(self.qpos[idx] + delta,
                                    self.rad_min[idx], self.rad_max[idx]))
        return self._go(target)

    def set_joint(self, p: Union[int, str], target_val: float) -> bool:
        """Dat gia tri tuyet doi cho mot khop (radian) hoac gripper (met, do mo tong)."""
        if isinstance(p, str) and p.lower() in GRIPPER_KEYS:
            return self.set_opening(float(target_val))

        idx = self._resolve_joint_index(p)
        if idx is None:
            return False
        if not self._wait_until_ready():
            print("[Robot] Executor chua san sang, bo qua lenh moi.")
            return False
        target = self.qpos.copy()
        target[idx] = float(np.clip(target_val, self.rad_min[idx], self.rad_max[idx]))
        return self._go(target)

    def set_joint_deg(self, p: Union[int, str], target_deg: float) -> bool:
        return self.set_joint(p, np.radians(target_deg))

    def set_pose(self, q_rad, grip: Optional[float] = None) -> bool:
        """Dat CA 6 khop trong MOT dich duy nhat (khong phai 6 lenh noi tiep)."""
        ok = self._go(q_rad)
        if grip is not None:
            ok = self.set_opening(grip) and ok
        return ok

    def set_opening(self, opening_m: float) -> bool:
        """Dat do mo TONG giua hai dau ngon tay (met, 0..0.08)."""
        value = float(np.clip(opening_m, 0.0, self.MAX_GRIPPER_M))
        if not self._b.set_opening(value):
            return False
        self._cmd_grip = value
        return True

    def set_gripper_pct(self, pct: float) -> bool:
        """Dat do mo gripper theo phan tram 0..100 (100 = mo het 0.08 m)."""
        return self.set_opening(float(np.clip(pct, 0.0, 100.0)) / 100.0 * self.MAX_GRIPPER_M)

    def set_tcp_pose(self, position, quaternion) -> bool:
        """Gui dich TCP o base_link; myarm_kinematics giai IK roi dua sang executor."""
        return self._b.send_tcp_pose(position, quaternion)

    def stop(self) -> bool:
        """DUNG KHAN CAP qua /myarm/robot/stop (latch safety epoch moi)."""
        return self._b.call_trigger(SERVICE_STOP)[0]

    def power_on(self) -> bool:
        return self._b.call_trigger(SERVICE_POWER_ON)[0]

    def power_off(self) -> bool:
        return self._b.call_trigger(SERVICE_POWER_OFF)[0]

    def rearm(self) -> bool:
        """
        Khoi phuc sau DUNG KHAN.

        Phai goi CA HAI service, vi sau stop co HAI fault doc lap:
          - driver: /myarm/robot/rearm        -> safety_state fault_latched
          - executor: /myarm/motion_execution/reset
            -> state "fault", detail "joint goal rejected; call reset after fault"
        Chi goi mot cai thi robot van im lang du moi thu khac trong nhu binh thuong.
        """
        driver_ok = self._b.call_trigger(SERVICE_REARM)[0]
        reset_ok = self._b.call_trigger(SERVICE_MOTION_RESET)[0]
        return bool(driver_ok and reset_ok)

    # ------------------------------------------------------------------ quy dao
    def wait_until_at(self, q_goal, tol_rad: Optional[float] = None,
                      timeout_s: float = 30.0) -> bool:
        """Cho cac khop toi dich. True neu toi noi, False neu het thoi gian.

        CHI xet cac khop ma lenh vua roi THUC SU yeu cau di chuyen (self._moving).

        LOI THAT (do tren robot that): truoc day doi CA 6 khop nam trong dung
        sai. Khop chi bi GIU nguyen vi tri thi khong the "toi noi" theo nghia
        nao - servo myArm giu khong noi, do duoc j3 troi 1.32 deg (0.023 rad)
        chi vi trong luc keo j2. The la wait_until_at truot o mot khop KHONG he
        duoc lenh di chuyen, FollowTrajectory huy ca quy dao ngay diem dau va
        tra False sau dung 30s, du tay da di duoc phan lon duong.
        """
        target = np.asarray(q_goal, dtype=np.float64).flatten()[:6]
        # Khong co feedback thi khong the xac nhan toi noi. Thoat ngay thay vi
        # quay du timeout_s roi tra False sau 30s im lang.
        if not np.all(np.isfinite(self.qpos)):
            print("[Robot] Khong co feedback khop moi - khong xac nhan duoc toi noi.")
            return False
        tol = self.SETTLE_TOL_RAD if tol_rad is None else float(tol_rad)
        # So voi dich SAU khi kep gioi han: neu _go da phai kep thi dich goc la
        # dich khong toi duoc, so vao do thi khong bao gio toi noi.
        target = np.clip(target, self.rad_min, self.rad_max)
        idx = np.arange(6) if self._moving is None else self._moving
        if idx.size == 0:
            return True   # lenh khong yeu cau khop nao di -> coi nhu da xong
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if float(np.max(np.abs(self.qpos[idx] - target[idx]))) <= tol:
                return True
            time.sleep(0.05)
        return False

    def _waypoints(self, trajectory) -> List[np.ndarray]:
        """Rut danh sach diem dich tu cac dang du lieu ma SDK cu da nhan."""
        if isinstance(trajectory, dict) and "points" in trajectory:
            return [np.asarray(p["positions"], dtype=np.float64) for p in trajectory["points"]]
        if isinstance(trajectory, (list, tuple)) and trajectory and isinstance(trajectory[0], dict):
            return [np.asarray(p["positions"], dtype=np.float64) for p in trajectory]
        array = np.asarray(trajectory, dtype=np.float64)
        if array.ndim == 2:
            # Chunk hanh dong cua model VLA: shape (T, 6) hoac (T, 7).
            return [row for row in array]
        return [array.flatten()]

    def FollowTrajectory(self, trajectory: Union[np.ndarray, Dict[str, Any], List[Any]],
                         max_step_rad: float = 0.08, dt_sim: float = 0.02,
                         callback: Optional[Any] = None, realtime: bool = True,
                         wait_arrival: bool = True) -> bool:
        """
        Gui tung diem dich cho myarm_motion_execution.

        Viec noi suy va kiem tra gioi han la cua minimum-jerk planner trong stack
        ROS, khong con tu chia buoc o day. Cac tham so max_step_rad/dt_sim duoc
        giu lai chi de khong lam vo loi goi cu.
        """
        ok = True
        grip = None
        for point in self._waypoints(trajectory):
            goal = point[:6]
            if not self._go(goal):
                ok = False
            grip = float(point[6]) if point.size > 6 else None
            if callback:
                callback(goal)
            if wait_arrival and not self.wait_until_at(goal):
                return False
        if grip is not None:
            self.set_opening(grip)
        return ok

    FollowTrajector = FollowTrajectory

    # ------------------------------------------------------------------ noi bo
    _cmd = np.zeros(6, dtype=np.float64)
    _cmd_grip = 0.0


__all__ = ["Robot"]

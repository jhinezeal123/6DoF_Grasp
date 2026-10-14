"""arm.py - MyArmM750: so huu port serial pymycobot.

Thay global `arm` + _open() cu: mot instance = mot ket noi, mo/dong qua
context manager. Van giu nguyen:
  - kiem tra port bi giu (pymycobot KHONG bat exclusive: mo chong len van
    'thanh cong' nhung 2 ben cung ghi -> lenh hong)
  - _move: retry khi firmware nuot lenh, bu khi servo dung non
  - _wait_stop: cho tay dung HAN that (is_moving() tra 0 CA KHI tay dang quay)
"""

from __future__ import annotations

import os
import stat
import time
from glob import glob

from pymycobot import MyArmMControl

from ..spec import RobotSpec


def _held(port: str) -> list:
    """Danh sach tien trinh dang giu port (doc /proc/*/fd).

    So sanh theo st_rdev (danh tinh thiet bi) chu khong chi theo duong dan:
    cung mot tay co the duoc mo qua /dev/ttyACM0, qua symlink ~/myarm/tty
    hoac qua /dev/serial/by-id/... — so chuoi duong dan se bo sot ke giu port,
    roi hai ben cung ghi lam lenh hong ma khong bao loi.
    """
    out, seen = [], set()
    real = os.path.realpath(port)
    try:
        device = os.stat(port)
    except OSError:
        device = None
    for fd in glob("/proc/[0-9]*/fd/*"):
        try:
            link = os.readlink(fd)
            if link == real:
                same = True
            elif device is not None:
                st = os.stat(fd)
                same = stat.S_ISCHR(st.st_mode) and st.st_rdev == device.st_rdev
            else:
                same = False
            if not same:
                continue
            pid = fd.split("/")[2]
            if pid in seen:
                continue
            seen.add(pid)
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmd = f.read().replace(b"\0", b" ").decode(errors="replace")
            out.append(f"pid {pid}  {cmd[:70]}")
        except OSError:  # tien trinh khac user hoac da thoat
            pass
    return out


# Cho phep lech nho tai bien gioi han khi doc (firmware tu chan lenh, khong chan viec doc).
_READ_TOL_DEG = 2.0


def _joints6(q) -> list | None:
    """Lay 6 goc khop tu response cua firmware, None neu doc loi.

    Firmware M750 dang chay tra 7 gia tri: 6 khop + truc end. pymycobot ghi ro
    "to support end control, ID 7 is allowed"; do that tren tay that (ba lan doc
    lien tiep deu 7 gia tri). Contract cua repo la 6 khop -> cat truc end.
    Day la noi DUY NHAT doi response vendor thanh 6 khop; caller khong tu cat.
    Doc serial thinh thoang tra ve int (-1) -> phai kiem tra truoc khi dung.
    """
    if not isinstance(q, (list, tuple)) or len(q) not in (6, 7) or -1 in q:
        return None
    return [float(value) for value in q[:6]]


def _out_of_range(q, spec) -> bool:
    """True neu co goc nam ngoai gioi han firmware.

    Do that: khi servo chua bat dien, get_angles() tra gia tri cau hinh
    (-200.0, 210.65, ...) ngoai FW_MIN/FW_MAX. Gia tri do khong phai tu the
    that -> khong duoc dem di tinh FK hay gui chuyen dong.
    """
    return any(
        not (low - _READ_TOL_DEG <= value <= high + _READ_TOL_DEG)
        for value, (low, high) in zip(q, zip(spec.fw_min_deg, spec.fw_max_deg))
    )


def _valid6(q, spec) -> list | None:
    """6 goc khop dung so luong VA trong gioi han firmware; None neu khong."""
    values = _joints6(q)
    if values is None or _out_of_range(values, spec):
        return None
    return values


def _ok6(q) -> bool:
    """True neu doc ra du 6 goc khop. Giu ten cu cho caller cu."""
    return _joints6(q) is not None


class MyArmM750:
    """Dieu khien truc tiep myArm M750 qua pymycobot (khong qua ROS).

    Dung: ``with MyArmM750() as arm: ...`` hoac arm.open()/arm.close() thu cong.
    Mo chong: neu port dang bi giu, open() RA LOI ngay (vi mo chong len khong
    bao loi nhung hai ben cung ghi -> lenh hong).
    """

    def __init__(self, spec: RobotSpec | None = None) -> None:
        self.spec = spec or RobotSpec()
        self._arm = None

    # ------------------------------------------------------------- vong doi
    def __enter__(self) -> "MyArmM750":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        # Khong tu dong dong serial: robot giu trang thai, chi nha khi nguoi dung
        # dong y. Dong o day khac dong TCP - khong co "transaction" de rollback.
        return False

    @property
    def is_open(self) -> bool:
        return self._arm is not None

    def open(self) -> MyArmMControl:
        """Mo port (idempotent). RA RuntimeError neu port dang bi tien trinh khac giu."""
        if self._arm is not None:
            return self._arm  # da mo roi: bo qua chot chan, neu khong chinh minh
            # bi coi la ke chiem port
        if held := _held(self.spec.port):
            raise RuntimeError(
                f"{self.spec.port} đang bị giữ: {', '.join(held)}. Tắt tiến trình/ROS trước."
            )
        self._arm = MyArmMControl(self.spec.port, baudrate=self.spec.baudrate)
        return self._arm

    def close(self) -> None:
        """Dong that port serial roi moi bo tham chieu.

        pymycobot 4.0.5 khong co API close o MyArmMControl, nhung doi tuong
        serial ben trong (``_serial_port``) thi co. Chi bo tham chieu la khong
        du: fd van mo, nen lan mo lai trong cung tien trinh se bi chinh chot
        chan cua driver bao "dang bi giu: pid <chinh minh>".
        """
        device, self._arm = self._arm, None
        serial_port = getattr(device, "_serial_port", None)
        if serial_port is not None and getattr(serial_port, "is_open", False):
            try:
                serial_port.close()
            except Exception:
                pass  # dong that bai thi van xu su nhu cu: chi bo tham chieu


    def _open(self) -> MyArmMControl:
        """Compat voi code cu goi _open()."""
        return self.open()

    # ------------------------------------------------------------- trang thai
    @property
    def q_deg(self) -> tuple | None:
        """Doc 6 goc khop hien tai (do). Tra None neu loi doc."""
        raw = self.open().get_angles()
        q = _valid6(raw, self.spec)
        if q is not None:
            return tuple(q)
        # Do that tren tay nay (40/40 lan doc sach khi board tra loi, ke ca luc
        # mat dien; chi im lang khi co van de o board/duong truyen):
        #   is_powered_on() = 0  -> board VAN tra loi, chi la servo chua bat dien
        #   is_powered_on() = -1 -> board khong tra loi CA truy van nguon
        # Tach hai ca nay vi khuyen "bat dien" khi board im lang la chi sai huong.
        powered = self.open().is_powered_on()
        if powered == 1:
            print("doc goc khop loi:", raw, "-> khong chay gi")
        elif powered == 0:
            print(
                "doc goc khop loi:",
                raw,
                "| is_powered_on = 0",
                "-> servo chua bat dien, chay ./robot power-on roi doc lai",
            )
        else:
            print(
                "doc goc khop loi:",
                raw,
                "| is_powered_on =",
                powered,
                "-> board khong tra loi (khong doc duoc ca trang thai nguon):",
                "kiem tra nguon board, cap USB, hoac tien trinh khac dang giu cong roi thu lai",
            )
        return None

    def power_on(self, wait_s: float = 2.5) -> bool:
        """Bat dien servo, tra True/False that.

        power_on() tra -1 KE CA khi thanh cong -> phai doc lai is_powered_on().
        """
        self.open().power_on()
        time.sleep(wait_s)
        return self._arm.is_powered_on() == 1

    def release_all_servos(self, data=None) -> None:
        """CAT MO-MEN -> tay ROT theo trong luc. Do tay/ke chan truoc. Co mo-men lai: power_on()."""
        self.open().release_all_servos(data)

    def temperatures(self) -> list | None:
        """Nhiet do 6 servo (do C, 0..255) - xem truoc khi chay lien tuc. Tra None neu loi/doc khong ra."""
        t = self.open().get_servo_temps()
        return t if isinstance(t, list) and len(t) == 6 else None

    def port_free(self) -> bool:
        """True neu khong ai giu port - kiem tra TRUOC khi mo, vi mo chong len khong bao loi nhung lenh se hong."""
        return not _held(self.spec.port)

    # ------------------------------------------------------------- di chuyen
    def wait_settled(
        self, timeout_s: float = 30, quiet_s: float = 0.5, start_s: float = 2.0
    ) -> bool:
        """Cho tay dung HAN that: True neu dung yen, False neu qua timeout hoac doc goc loi.

        KHONG dung is_moving(): no tra 0 CA KHI tay dang quay -> lenh ke tiep de
        ra lenh truoc -> servo qua tai. Da gap that: gripper over-current,
        firmware chot loi va chan MOI chuyen dong cho toi khi cycle nguon
        (power_off + power_on moi xoa duoc, stop() khong xoa).
        Phai cho qua start_s: lan doc dau (tay chua kip khoi dong) trong nhu
        da dung yen.
        """
        a, t0, last, quiet, started = self.open(), time.time(), None, 0.0, False
        while time.time() - t0 < timeout_s:
            q = _valid6(a.get_angles(), self.spec)
            if q is None:
                return False
            if last is not None:
                if max(abs(x - y) for x, y in zip(q, last)) > 0.05:
                    started, quiet = True, 0.0
                elif started:
                    quiet += 0.1
                    if quiet >= quiet_s:
                        return True
                elif time.time() - t0 > start_s:
                    return True  # khong nhuc nhich: da o dich, hoac bi chan
            last = q
            time.sleep(0.1)
        return False

    def write_joints(
        self, target, speed: int = 20, timeout_s: float = 30, tol: float = 1.0, tries: int = 4
    ) -> bool:
        """Ghi 6 goc khop roi DOC LAI kiem tra. Tra True neu moi khop |q - target| <= tol.

        Phai kiem tra lai vi do that tren tay nay (khong phai loi doan):
          (1) firmware THINH THOANG NUOT LENH - write_angles tra ack=1 nhung khop
              dung yen, ~1/10 lenh;
          (2) servo DUNG NON ~1 do so voi lenh -> o tam voi 470mm thanh ~20mm
              sai so vi tri.
        Lenh bi nuot -> gui lai y nguyen. Dung non -> bu dan, nhung chan bu
        trong +/-3 do (dang ti vao vat thi dung day mai).
        """
        a, target = self.open(), [float(v) for v in target]
        cmd = list(target)
        for _ in range(tries):
            q0 = _valid6(a.get_angles(), self.spec)  # doc truoc: biet lenh co bi nuot khong
            a.write_angles(cmd, speed)
            self.wait_settled(timeout_s)
            q = _valid6(a.get_angles(), self.spec)
            if q is None:
                continue
            err = [t - v for t, v in zip(target, q)]
            if max(abs(e) for e in err) <= tol:
                return True
            if q0 is not None and max(abs(v - w) for v, w in zip(q, q0)) < 0.05:
                continue  # khong nhuc nhich: lenh bi nuot, gui lai y nguyen
            cmd = [t + max(-3.0, min(3.0, c + e - t)) for t, c, e in zip(target, cmd, err)]
        q = _valid6(a.get_angles(), self.spec)
        return q is not None and max(abs(t - v) for t, v in zip(target, q)) <= tol

    def set_joints(self, angles, speed: int = 30, wait: bool = True, timeout_s: float = 30) -> bool:
        """Dat 6 khop [q1..q6] do, DI CHUYEN THAT. write_angles tu chan ngoai FW limits.

        wait=True: cho dung han roi DOC LAI, gui lai neu bi nuot lenh, bu neu dung non.
        """
        if not wait:
            self.open().write_angles(list(angles), speed)
            return True
        return self.write_joints(angles, speed, timeout_s)

    def set_joint(
        self,
        joint_id: int,
        degree: float,
        speed: int = 30,
        wait: bool = True,
        timeout_s: float = 30,
    ) -> bool:
        """Doi 1 khop (1..6) sang goc moi, 5 khop kia GIU NGUYEN. joint_id ngoai 1..6 -> loi ngay."""
        if not 1 <= int(joint_id) <= 6:
            raise ValueError("joint_id phai trong 1..6, nhan duoc %r" % (joint_id,))
        q = self.q_deg
        if q is None:
            return False
        q = list(q)
        q[int(joint_id) - 1] = float(degree)
        if not wait:
            self.open().write_angles(q, speed)
            return True
        return self.write_joints(q, speed, timeout_s)


__all__ = ["MyArmM750", "_joints6", "_valid6", "_ok6"]

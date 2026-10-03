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
import time
from glob import glob

from pymycobot import MyArmMControl

from ..spec import RobotSpec


def _held(port: str) -> list:
    """Danh sach tien trinh dang giu port (doc /proc/*/fd)."""
    out, port = [], os.path.realpath(port)
    for fd in glob("/proc/[0-9]*/fd/*"):
        try:
            if os.readlink(fd) == port:
                pid = fd.split("/")[2]
                with open(f"/proc/{pid}/cmdline", "rb") as f:
                    cmd = f.read().replace(b"\0", b" ").decode(errors="replace")
                out.append(f"pid {pid}  {cmd[:70]}")
        except OSError:  # tien trinh khac user hoac da thoat
            pass
    return out


def _ok6(q) -> bool:
    """True neu doc ra du 6 goc khop. Doc serial thinh thoang tra ve int -> phai kiem tra truoc khi dung."""
    return isinstance(q, list) and len(q) == 6 and -1 not in q


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
        """Nha port serial (co the la MyArmMControl khong co API dong - chi bo tham chieu)."""
        self._arm = None

    def _open(self) -> MyArmMControl:
        """Compat voi code cu goi _open()."""
        return self.open()

    # ------------------------------------------------------------- trang thai
    @property
    def q_deg(self) -> tuple | None:
        """Doc 6 goc khop hien tai (do). Tra None neu loi doc."""
        q = self.open().get_angles()
        if not _ok6(q):
            print("doc goc khop loi:", q, "-> khong chay gi")
            return None
        return tuple(float(v) for v in q)

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
            q = a.get_angles()
            if not _ok6(q):
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
            q0 = a.get_angles()  # doc truoc khi ghi: de biet lenh co bi nuot khong
            a.write_angles(cmd, speed)
            self.wait_settled(timeout_s)
            q = a.get_angles()
            if not _ok6(q):
                continue
            err = [t - v for t, v in zip(target, q)]
            if max(abs(e) for e in err) <= tol:
                return True
            if _ok6(q0) and max(abs(v - w) for v, w in zip(q, q0)) < 0.05:
                continue  # khong nhuc nhich: lenh bi nuot, gui lai y nguyen
            cmd = [t + max(-3.0, min(3.0, c + e - t)) for t, c, e in zip(target, cmd, err)]
        q = a.get_angles()
        return _ok6(q) and max(abs(t - v) for t, v in zip(target, q)) <= tol

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


__all__ = ["MyArmM750", "_ok6"]

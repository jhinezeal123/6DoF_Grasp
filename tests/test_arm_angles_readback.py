"""Doc goc khop tu tay that: hai dac tinh phan cung da do duoc tren M750.

- get_angles() tra 7 gia tri: 6 khop + truc end (pymycobot cho phep ID 7 cho
  end control). Repo chi dung 6 khop dau.
- Khi servo chua bat dien, get_angles() tra gia tri cau hinh NGOAI gioi han
  firmware (do duoc: -200.0, 210.65, ...). Do khong phai tu the that -> phai
  coi la doc loi, khong duoc dem di tinh FK hay gui chuyen dong.
"""

from types import SimpleNamespace

from m750.robot.arm import MyArmM750, _joints6, _valid6
from m750.spec import RobotSpec

REPLY_7 = [-4.39, 72.87, -29.72, 5.27, -47.98, -4.74, -62.4]
REPLY_MAT_DIEN = [-200.0, 210.65, -210.65, -200.0, 200.0, -200.0, 200.0]


def test_bay_gia_tri_cat_con_sau_khop():
    assert _joints6(REPLY_7) == REPLY_7[:6]
    assert _joints6(REPLY_7[:6]) == REPLY_7[:6]


def test_gia_tri_ngoai_gioi_han_firmware_bi_loai():
    spec = RobotSpec()
    assert _valid6(REPLY_MAT_DIEN, spec) is None
    assert _valid6(REPLY_7, spec) == REPLY_7[:6]


def test_doc_loi_that_thi_tra_ve_none():
    assert _joints6(-1) is None
    assert _joints6([1.0] * 5) is None
    assert _joints6([1.0] * 8) is None


def _arm(reply, powered=1):
    device = SimpleNamespace(get_angles=lambda: reply, is_powered_on=lambda: powered)
    arm = MyArmM750(RobotSpec())
    arm.open = lambda: device  # khong mo serial that trong test
    return arm


def test_q_deg_tra_sau_goc_tu_bay_gia_tri():
    assert _arm(REPLY_7).q_deg == tuple(REPLY_7[:6])


def test_q_deg_tu_choi_khi_servo_chua_bat_dien(capsys):
    assert _arm(REPLY_MAT_DIEN, powered=0).q_deg is None
    assert "servo chua bat dien" in capsys.readouterr().out


def test_q_deg_bao_thieu_dien_khi_doc_tra_minus_one(capsys):
    assert _arm(-1, powered=0).q_deg is None
    assert "servo chua bat dien" in capsys.readouterr().out

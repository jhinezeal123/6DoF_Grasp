"""Prompt chinh xac cua nhom tac gia PhysBrain.

Sao chep nguyen van tu `app.py` cua Space `DeepCybo/physbrain1-5-8b-demo`. Sua
mot chu la hong im lang: model van tra loi, chi la tra loi sai dinh dang hoac
sai toa do. Xem `P1_KET_QUA.md` §1.
"""

from __future__ import annotations

# Suffix ma EmbodiedEvalKit dung cho cac model backbone Qwen3-VL.
# Toa do tra ve chuan hoa 0-1000, KHONG phai pixel.
POINT_SUFFIX = 'The answer should be presented in JSON format as follows: [{"point_2d": [x, y]}].'

# Hoi ngon kep: day la cau hoi cua rieng test nay, khong co trong harness cua tac gia.
GRIPPER_QUESTION = "Point to the red gripper fingers at the tip of the robot arm."


def point_question(instruction: str) -> str:
    """Cau hoi chi diem cho mot lenh dang 'pick up the blue block'."""
    return instruction.strip().rstrip(".") + ".\n" + POINT_SUFFIX


def pixels(point: tuple[int, int], frame_shape: tuple[int, ...]) -> tuple[float, float]:
    """Doi toa do 0-1000 cua model sang pixel cua khung anh cu the."""
    height, width = frame_shape[:2]
    return (point[0] / 1000.0 * width, point[1] / 1000.0 * height)


__all__ = ["POINT_SUFFIX", "GRIPPER_QUESTION", "point_question", "pixels"]

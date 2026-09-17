"""
model/base.py - Lop Model co so trong framework.

MOT diem vao duy nhat: inference(input). `input` la MOT doi tuong bat ky do ban
tu dinh nghia - SDK khong ap dat gi ve kieu, so luong hay thu tu tham so.

Vi sao mot doi tuong thay vi nhieu tham so roi:
  - Ban tu quyet dinh model can gi (anh? prompt? state? ca ba?) va gom chung
    vao dict / dataclass / NamedTuple / doi tuong rieng cua ban.
  - Doi cau truc input sau nay KHONG phai sua chu ky ham, nen khong pha vo
    hop dong voi pipeline dang goi no.
  - SDK khong phai biet model cua ban can gi. Do la ranh gioi dung.

Vi du:

    class MyVLA(Model):
        def inference(self, input):
            img = input["img"]                  # dict
            prompt = input["prompt"]
            q = input["state"]["qpos"]
            return my_model(img, prompt, q)     # (T, 7) hoac gi cung duoc

    MyVLA().inference({"img": frame, "prompt": "pick the cube", "state": obs})
"""
from abc import ABC, abstractmethod
from typing import Any


class Model(ABC):
    """Lop truu tuong co so cho cac mo hinh AI/ML trong framework."""

    def __init__(self, name: str = "BaseModel", **kwargs):
        self.name = name
        self.kwargs = kwargs

    @abstractmethod
    def inference(self, input: Any) -> Any:
        """
        Suy luan tu MOT doi tuong dau vao, tra ve ket qua cua ban.

        Args:
            input: mot doi tuong duy nhat do nguoi goi quyet dinh (thuong la
                dict gom anh + prompt + trang thai robot, nhung khong bat buoc).

        Returns:
            Any: ket qua model cua ban (vd action chunk shape (T, 7)).
        """
        raise NotImplementedError("Phuong thuc inference() phai duoc trien khai trong lop con.")

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}')>"

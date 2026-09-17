"""
model/ - Package chua lop co so Model trong framework.

KHONG co san model con nao o day. Muon dung model gi (VLA, VLM, policy...),
tao mot package rieng canh day va ke thua tu Model, vi du:

    # program/model/my_vla/my_vla.py
    from program.model.base import Model

    class MyVLA(Model):
        def inference(self, input):
            # input la MOT doi tuong ban tu dinh nghia (dict/dataclass/...).
            return my_model(input["img"], input["prompt"], input["state"]["qpos"])
"""
from .base import Model

__all__ = ["Model"]

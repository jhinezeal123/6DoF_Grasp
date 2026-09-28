"""Client goi PhysBrain1.5-8B qua llama-server (API OpenAI).

Chi lam mot viec: dua anh + cau hoi, tra ve mot diem 2D. Khong biet gi ve robot,
khong biet gi ve servo - de thay bang model khac ma khong dung den pipeline.
"""

from __future__ import annotations

import base64
import re
import time

import cv2
import numpy as np
import requests

from .prompts import POINT_SUFFIX, pixels

# Model KHONG phai luc nao cung theo dung khuon JSON cua tac gia. Do that tren
# may nay bat duoc ca hai kieu tra loi cho cung mot cau hoi:
#     [{"point_2d": [545, 575]}]      <- dung khuon (da so)
#     [(520, 570)]                    <- kieu tuple Python
# Chi khop mot kieu thi cu vai lan goi la vo mot lan, rat kho lan ra nguyen nhan.
_POINT_RES = (
    re.compile(r'"point_2d"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]'),
    re.compile(r"[([]\s*(\d+)\s*,\s*(\d+)\s*[)\]]"),
)


def _parse_point(text: str) -> tuple[int, int] | None:
    for pattern in _POINT_RES:
        match = pattern.search(text)
        if match:
            return int(match.group(1)), int(match.group(2))
    return None


class PhysBrainError(RuntimeError):
    """Khong goi duoc model, hoac model khong tra ve diem nao."""


class PhysBrainClient:
    """Goi llama-server dang chay san.

    Anh truyen vao la RGB (dung quy uoc cua m750.ros.camera.Camera). cv2.imencode
    cho BGR, nen phai doi truoc khi ma hoa - doi nham thi do thanh xanh va model
    se chi vao mot vat khac.
    """

    def __init__(self, url: str = "http://127.0.0.1:8081",
                 timeout_s: float = 180.0, jpeg_quality: int = 92) -> None:
        self.url = url.rstrip("/")
        self.timeout_s = float(timeout_s)
        self.jpeg_quality = int(jpeg_quality)
        self.last_latency_s: float | None = None
        self.last_text: str = ""

    def _encode(self, frame_rgb: np.ndarray) -> str:
        bgr = np.ascontiguousarray(frame_rgb[:, :, ::-1])
        ok, buffer = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY),
                                                self.jpeg_quality])
        if not ok:
            raise PhysBrainError("khong ma hoa duoc khung anh")
        return "data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode("ascii")

    def ready(self) -> bool:
        """Server co song khong. Dung truoc khi chay that de bao loi som."""
        try:
            return requests.get(self.url + "/health", timeout=5.0).json().get("status") == "ok"
        except Exception:  # noqa: BLE001 - chi de bao cao, khong phai logic
            return False

    def point(self, frame_rgb: np.ndarray,
              question: str) -> tuple[float, float] | None:
        """Hoi mot diem, tra ve toa do PIXEL, hoac None neu model khong chi duoc.

        Phan biet ro hai loai that bai:
          - khong goi duoc server      -> nem PhysBrainError (ha tang hong)
          - goi duoc nhung khong parse -> tra None (model khong thay vat)
        Gop hai loai lam mot se khien ben goi thu lai vo ich khi server da chet.
        """
        payload = {
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": self._encode(frame_rgb)}},
                    {"type": "text", "text": question},
                ],
            }],
            "temperature": 0.0,       # greedy, dung nhu cau hinh danh gia cua tac gia
            "max_tokens": 64,
        }
        started = time.monotonic()
        try:
            response = requests.post(self.url + "/v1/chat/completions",
                                     json=payload, timeout=self.timeout_s)
            response.raise_for_status()
            text = response.json()["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001 - bao loi that ra ngoai
            raise PhysBrainError("khong goi duoc model: %s" % exc) from exc
        self.last_latency_s = time.monotonic() - started
        self.last_text = text

        match = _parse_point(text)
        if match is None:
            return None
        return pixels(match, frame_rgb.shape)


__all__ = ["PhysBrainClient", "PhysBrainError", "POINT_SUFFIX"]

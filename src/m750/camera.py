"""camera.py - Camera V4L2: chup 1 khung hoac stream MJPEG.

MjpegStream co stop() la METHOD that (join thread grab, shutdown, close) -
thay cho `srv.stop = lambda: ...` hack. Giu nguyen comment core-dump OpenCV.
"""
from __future__ import annotations

import atexit
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

from .spec import DEFAULT_CAMERA


def capture(path: str = "cap_img.jpg", device: str = DEFAULT_CAMERA,
            width: int = 1280, height: int = 720, tries: int = 5):
    """Chup 1 khung MOI NHAT, ghi DE len file (mac dinh cap_img.jpg). Tra duong dan hoac None.

    Mo camera roi dong ngay: de tuoi khung phai bo vai khung dau (V4L2 giu buffer cu),
    nen doc bo tries khung va lay khung cuoi. Khong dung chung voi MjpegStream.
    """
    c = cv2.VideoCapture(device, cv2.CAP_V4L2)
    c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    c.set(cv2.CAP_PROP_FRAME_WIDTH, width); c.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    try:
        for _ in range(tries):
            ok, im = c.read()
        if not ok:
            print("khong doc duoc khung nao tu", device)
            return None
        cv2.imwrite(path, im)
        print("ghi", os.path.abspath(path), im.shape[1], "x", im.shape[0])
        return path
    finally:
        c.release()  # phai tra camera, khong thi lan goi sau khong mo duoc


class MjpegStream:
    """Stream MJPEG tai http://<ip>:<port>/. goi stop() truoc khi thoat, khong thi core dump.

    stop() JOIN thread grab: no con song luc Python thoat, OpenCV dang chan
    trong cap.read() (C++) -> "terminate called without an active exception"
    -> core dump. Da gap that.
    """

    def __init__(self, port: int = 8080, device: str = DEFAULT_CAMERA,
                 fps: int = 25, quality: int = 75,
                 width: int = 1280, height: int = 720) -> None:
        self.port, self.device = port, device
        self.fps, self.quality = fps, quality
        self.width, self.height = width, height
        self._jpg, self._lock, self._run = [b""], threading.Lock(), threading.Event()
        self._srv = None
        self._t = None

    def _grab(self):
        # thread rieng: cap.read() CHAN khi camera rut ra, goi chung thread render la treo ca stream
        c = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # YUYV ton CPU gap 3
        c.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        c.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        c.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # hang doi 1 khung: khong doc lai anh cu
        while not self._run.is_set():
            ok, im = c.read()
            if not ok:
                time.sleep(0.5); continue  # rut camera: cho, khong quay CPU
            with self._lock:
                self._jpg[0] = cv2.imencode(
                    ".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, self.quality])[1].tobytes()
        c.release()

    def _handler(self):
        stream = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass  # tat log tung request, khong thi stdout ngap

            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type",
                                 "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                last = None
                while True:
                    with stream._lock:
                        j = stream._jpg[0]
                    if j and j is not last:  # khung moi moi gui, khung cu bo qua
                        last = j
                        try:
                            self.wfile.write(
                                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + j + b"\r\n")
                        except (BrokenPipeError, ConnectionResetError):
                            return  # nguoi xem dong tab
                    time.sleep(1.0 / stream.fps)

        return H

    def start(self) -> "MjpegStream":
        """Bat dau stream (idempotent). atexit tu don neu quen stop()."""
        if self._srv is not None:
            return self
        self._srv = ThreadingHTTPServer(("0.0.0.0", self.port), self._handler())
        self._srv.daemon_threads = True
        self._t = threading.Thread(target=self._grab, daemon=True)
        self._t.start()
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()
        # atexit: script ket thuc tu nhien (khong kip goi stop()) VAN duoc don
        # -> het core dump.
        atexit.register(self.stop)
        print("camera: http://0.0.0.0:%d/" % self.port)
        return self

    def stop(self) -> None:
        """Dung stream: giai phong camera va port HTTP. Idempotent."""
        if self._srv is None:
            return
        self._run.set()
        if self._t is not None:
            self._t.join(3)
        self._srv.shutdown()
        self._srv.server_close()
        self._srv, self._t = None, None


def camera_stream(port: int = 8080, device: str = DEFAULT_CAMERA, fps: int = 25,
                  quality: int = 75, width: int = 1280, height: int = 720) -> MjpegStream:
    """Compat voi ham camera_stream() cu: tra MjpegStream DA start."""
    return MjpegStream(port, device, fps, quality, width, height).start()


__all__ = ["MjpegStream", "capture", "camera_stream"]

"""Ánh xạ URL stream sang cache JPEG; giữ một nguồn khai báo camera."""

from m750.ros.usb_camera import USB_CAMERAS

STREAM_KEYS = {
    "/stream/cam_sim.mjpg": "sim",
    "/stream/cam_third.mjpg": "third",
    "/stream/cam_real.mjpg": "real",
}
STREAM_KEYS.update({"/stream/cam_%s.mjpg" % key: key for key, _ in USB_CAMERAS})

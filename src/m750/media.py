"""
m750/media.py - Nén ảnh JPEG hiệu năng cao phục vụ stream mạng.
"""
import cv2
import numpy as np


class ImageCompressor:
    """Bộ nén khung hình sang định dạng JPEG tốc độ cao."""

    @staticmethod
    def encode_jpeg(img: np.ndarray, quality: int = 75) -> bytes:
        """
        Nén ảnh numpy (H, W, C) sang định dạng JPEG bytes.
        
        Args:
            img: Ảnh đầu vào RGB hoặc BGR (H, W, 3) uint8.
            quality: Chất lượng nén 1-100.
            
        Returns:
            bytes dữ liệu ảnh JPEG đã nén.
        """
        if img is None or img.size == 0:
            return b""
        
        # Nếu ảnh là RGB (MuJoCo render mặc định RGB), chuyển sang BGR trước khi imencode
        if len(img.shape) == 3 and img.shape[2] == 3:
            # Kiểm tra xem có cần chuyển RGB sang BGR không
            bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            bgr_img = img

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(np.clip(quality, 10, 100))]
        success, encoded = cv2.imencode('.jpg', bgr_img, encode_param)
        if not success:
            return b""
        return encoded.tobytes()

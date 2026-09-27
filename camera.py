"""MJPEG camera tren myArm M750 - xem tai http://<ip>:8080/ .

Tat bang 1 trong 2 cach (khong can dung terminal dang chay):
    touch stop_cam      # xoa file nay la tat
    pkill -f 'came[r]a\.py'
Khong chiem /dev/ttyACM1: stream va dieu khien robot chay song song duoc.
"""
from ductocbatdat import camera_stream
import os
import threading
import time

STOP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stop_cam")
if os.path.exists(STOP):
    os.remove(STOP)
camera_stream()
print("tat bang: touch %s   (hoac pkill -f 'came[r]a\\.py')" % STOP)
while not os.path.exists(STOP):  # vong lap thay cho input(): terminal van dung duoc
    time.sleep(0.3)
print("thay file co -> tat")

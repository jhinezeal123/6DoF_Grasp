"""Mo web xem truoc mo phong myArm: http://<ip>:8081/ .

Sliders x/y/z/rx/ry/rz -> mo phong cap nhat ngay khi keo.
realtime: doc goc khop THAT tu driver (bam lai -> ve simulate). sync: chay THAT tren tay.
Chuot: keo trai quay, lan zoom, keo phai tinh tien. Tat: pkill -f 'previe[w]'
"""
import sys
sys.path.insert(0, "/workspace/6DoF_Grasp/htc")
from ductocbatdat import preview
import threading

preview()
threading.Event().wait()  # giu tien trinh song; atexit tu don khi thoat

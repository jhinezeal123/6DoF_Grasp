#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark_camera.py (v2) -- do kha nang camera tren Jetson AGX Xavier.

Cac duong doc duoc so sanh:
  v4l2   : cv2.VideoCapture(dev, CAP_V4L2)                    (OpenCV V4L2, decode MJPG bang CPU)
  gst-hw : GStreamer v4l2src ! jpegparse ! nvjpegdec           (decode MJPG bang HW block NVJPG)
  gst-sw : GStreamer v4l2src ! jpegparse ! jpegdec             (decode MJPG bang CPU)

YUYV khong nen nen khong co buoc decode -> chi chay: v4l2, gst-sw.

Tuy chon --lock-exposure bat che do phoi sang thu cong, tranh viec auto-exposure
tu keo dai thoi gian phoi sang lam tut fps (dac biet khi thieu sang).

Vi du:
  python3 benchmark_camera.py --list
  python3 benchmark_camera.py --duration 5 --lock-exposure --exposure 156
  python3 benchmark_camera.py --duration 3 --only-format MJPG
"""

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time

import cv2

CLK_TCK = os.sysconf(os.sysconf_names['SC_CLK_TCK'])

MATRIX = [
    ('YUYV', 640, 480, 30),
    ('YUYV', 800, 600, 24),
    ('YUYV', 1280, 720, 10),
    ('YUYV', 1920, 1080, 5),
    ('MJPG', 640, 480, 30),
    ('MJPG', 1280, 720, 30),
    ('MJPG', 1280, 720, 60),
    ('MJPG', 1920, 1080, 30),
]

COLUMNS = [
    'backend', 'format', 'req_w', 'req_h', 'req_fps', 'exposure',
    'neg_w', 'neg_h', 'neg_fps', 'neg_fourcc',
    'ok', 'open_s', 'warmup_s', 'frames', 'duration_s', 'fps_actual',
    'interval_mean_ms', 'interval_std_ms', 'interval_p50_ms',
    'interval_p95_ms', 'interval_p99_ms', 'interval_max_ms',
    'cpu_percent', 'rss_mb', 'frame_shape', 'note',
]


def backends_for(fmt):
    if fmt == 'YUYV':
        return ['v4l2', 'gst-sw']
    return ['v4l2', 'gst-hw', 'gst-sw']


# --- do luong tai nguyen -----------------------------------------------------

def cpu_ticks():
    with open('/proc/self/stat', 'r') as fh:
        data = fh.read()
    tail = data[data.rfind(')') + 1:].split()
    return int(tail[11]) + int(tail[12])


def rss_mb():
    try:
        with open('/proc/self/status', 'r') as fh:
            for line in fh:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return float('nan')


def percentile(values, q):
    if not values:
        return float('nan')
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


# --- V4L2 controls -----------------------------------------------------------

def v4l2_ctl(device, args):
    try:
        out = subprocess.run(['v4l2-ctl', '-d', device] + args,
                             capture_output=True, text=True, timeout=15)
        return out.returncode, out.stdout, out.stderr
    except Exception as exc:
        return 1, '', str(exc)


def read_ctrls(device):
    rc, out, _ = v4l2_ctl(device, ['--list-ctrls'])
    if rc != 0:
        return {}
    vals = {}
    keys = ('exposure_auto', 'exposure_absolute', 'exposure_auto_priority',
            'gain', 'brightness', 'white_balance_temperature_auto')
    for line in out.splitlines():
        line = line.strip()
        for key in keys:
            if line.startswith(key) and 'value=' in line:
                vals[key] = line.split('value=')[1].split()[0]
    return vals


def set_ctrls(device, **kwargs):
    ctrl = ','.join('%s=%s' % (k, v) for k, v in kwargs.items())
    rc, out, err = v4l2_ctl(device, ['-c', ctrl])
    return rc == 0, (err or out).strip()


# --- GStreamer / OpenCV ------------------------------------------------------

def gst_pipeline(dev, fmt, w, h, fps, decoder):
    head = 'v4l2src device=%s ! ' % dev
    if fmt == 'MJPG':
        caps = 'image/jpeg,width=%d,height=%d,framerate=%d/1' % (w, h, fps)
        dec = 'nvjpegdec' if decoder == 'hw' else 'jpegdec'
        chain = '%s ! jpegparse ! %s' % (caps, dec)
    else:
        chain = 'video/x-raw,format=YUY2,width=%d,height=%d,framerate=%d/1' % (w, h, fps)
    return (head + chain +
            ' ! videoconvert ! video/x-raw,format=BGR'
            ' ! appsink drop=1 max-buffers=2 sync=false')


def open_capture(backend, dev, fmt, w, h, fps):
    if backend == 'v4l2':
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if not cap.isOpened():
            return None, 'khong mo duoc %s' % dev
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fmt))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(w))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(h))
        cap.set(cv2.CAP_PROP_FPS, float(fps))
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return cap, ''

    decoder = 'hw' if backend == 'gst-hw' else 'sw'
    pipeline = gst_pipeline(dev, fmt, w, h, fps, decoder)
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        return None, 'pipeline GStreamer khong mo duoc (%s)' % backend
    return cap, ''


def run_trial(backend, dev, fmt, w, h, fps, duration, warmup_max, exposure_note):
    row = {
        'backend': backend, 'format': fmt,
        'req_w': w, 'req_h': h, 'req_fps': fps, 'exposure': exposure_note,
        'neg_w': '', 'neg_h': '', 'neg_fps': '', 'neg_fourcc': '',
        'ok': False, 'open_s': '', 'warmup_s': '', 'frames': 0,
        'duration_s': '', 'fps_actual': '',
        'interval_mean_ms': '', 'interval_std_ms': '',
        'interval_p50_ms': '', 'interval_p95_ms': '', 'interval_p99_ms': '',
        'interval_max_ms': '', 'cpu_percent': '', 'rss_mb': '',
        'frame_shape': '', 'note': '',
    }

    t0 = time.perf_counter()
    cap, note = open_capture(backend, dev, fmt, w, h, fps)
    t_open = time.perf_counter()
    row['open_s'] = round(t_open - t0, 4)
    if note:
        row['note'] = note
    if cap is None:
        return row

    try:
        if backend == 'v4l2':
            row['neg_w'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            row['neg_h'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            row['neg_fps'] = round(cap.get(cv2.CAP_PROP_FPS), 2)
            fc = int(cap.get(cv2.CAP_PROP_FOURCC))
            row['neg_fourcc'] = ''.join(chr((fc >> (8 * i)) & 0xFF) for i in range(4))

        ok, frame = False, None
        while time.perf_counter() - t_open < warmup_max:
            ok, frame = cap.read()
            if ok and frame is not None:
                break
        row['warmup_s'] = round(time.perf_counter() - t_open, 4)
        if not ok or frame is None:
            row['note'] = (row['note'] + ' | khong doc duoc frame dau tien').strip(' |')
            return row

        row['frame_shape'] = '%dx%dx%d' % (frame.shape[0], frame.shape[1], frame.shape[2])

        cpu0 = cpu_ticks()
        t_meas = time.perf_counter()
        marks = []
        while True:
            ok, frame = cap.read()
            now = time.perf_counter()
            if not ok or frame is None:
                row['note'] = (row['note'] + ' | read() that bai giua chung').strip(' |')
                break
            marks.append(now)
            if now - t_meas >= duration:
                break
        t_end = time.perf_counter()
        cpu1 = cpu_ticks()
        rss = rss_mb()
        wall = t_end - t_meas

        row['frames'] = len(marks)
        if len(marks) >= 2:
            span = marks[-1] - marks[0]
            row['duration_s'] = round(span, 4)
            row['fps_actual'] = round((len(marks) - 1) / span, 2) if span > 0 else ''
            iv = [(marks[i + 1] - marks[i]) * 1000.0 for i in range(len(marks) - 1)]
            row['interval_mean_ms'] = round(statistics.mean(iv), 3)
            row['interval_std_ms'] = round(statistics.pstdev(iv), 3) if len(iv) > 1 else 0.0
            row['interval_p50_ms'] = round(percentile(iv, 0.50), 3)
            row['interval_p95_ms'] = round(percentile(iv, 0.95), 3)
            row['interval_p99_ms'] = round(percentile(iv, 0.99), 3)
            row['interval_max_ms'] = round(max(iv), 3)
        if wall > 0:
            row['cpu_percent'] = round((cpu1 - cpu0) / CLK_TCK / wall * 100.0, 2)
        row['rss_mb'] = round(rss, 1)
        row['ok'] = True
        return row
    finally:
        try:
            cap.release()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser(description='Benchmark camera tren Jetson.')
    ap.add_argument('--device', default='/dev/video0')
    ap.add_argument('--duration', type=float, default=5.0)
    ap.add_argument('--warmup-max', type=float, default=10.0)
    ap.add_argument('--settle', type=float, default=1.5)
    ap.add_argument('--out-dir', default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument('--only-format', choices=['YUYV', 'MJPG'], default=None)
    ap.add_argument('--only-backend', default=None)
    ap.add_argument('--lock-exposure', action='store_true',
                    help='bat phoi sang thu cong de tranh auto-exposure keo tut fps')
    ap.add_argument('--exposure', type=int, default=156,
                    help='gia tri exposure_absolute khi --lock-exposure')
    ap.add_argument('--list', action='store_true')
    args = ap.parse_args()

    if args.list:
        for fmt, w, h, fps in MATRIX:
            print('%-4s %4dx%-5d @%-3d  backends: %s'
                  % (fmt, w, h, fps, ', '.join(backends_for(fmt))))
        return 0

    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    t_all = time.perf_counter()

    expo_before = read_ctrls(args.device)
    exposure_note = 'auto'
    locked = False

    print('== Benchmark camera: %s | %.1fs/config ==' % (args.device, args.duration), flush=True)
    print('== cv2 %s ==' % cv2.__version__, flush=True)

    try:
        if args.lock_exposure:
            ok, msg = set_ctrls(args.device, exposure_auto=1,
                                exposure_auto_priority=0,
                                exposure_absolute=args.exposure)
            locked = ok
            exposure_note = 'locked=%d' % args.exposure if ok else ('lock FAILED(%s)' % msg)
            print('== Lock exposure: %s ==' % exposure_note, flush=True)
            time.sleep(1.0)

        for fmt, w, h, fps in MATRIX:
            if args.only_format and fmt != args.only_format:
                continue
            for backend in backends_for(fmt):
                if args.only_backend and backend != args.only_backend:
                    continue
                label = '%s %dx%d@%d [%s]' % (fmt, w, h, fps, backend)
                print('[RUN ] ' + label, flush=True)
                row = run_trial(backend, args.device, fmt, w, h, fps,
                                args.duration, args.warmup_max, exposure_note)
                rows.append(row)
                if row['ok']:
                    print('[DONE] %s -> %.1f fps | CPU %.1f%% | RSS %.0f MB | std %.2f ms | p99 %.2f ms'
                          % (label, row['fps_actual'], row['cpu_percent'], row['rss_mb'],
                             row['interval_std_ms'], row['interval_p99_ms']), flush=True)
                else:
                    print('[FAIL] %s -> %s' % (label, row['note'] or 'unknown'), flush=True)
                time.sleep(args.settle)
    finally:
        if locked:
            set_ctrls(args.device, exposure_auto=expo_before.get('exposure_auto', '3'),
                      exposure_auto_priority=expo_before.get('exposure_auto_priority', '1'))
            print('== Da tra exposure ve auto ==', flush=True)

    csv_path = os.path.join(args.out_dir, 'benchmark_results.csv')
    with open(csv_path, 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction='ignore')
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, '') for k in COLUMNS})

    report = {
        'script': os.path.basename(__file__),
        'device': args.device,
        'duration_s': args.duration,
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_elapsed_s': round(time.perf_counter() - t_all, 1),
        'cv2_version': cv2.__version__,
        'exposure_before': expo_before,
        'exposure_during': exposure_note,
        'results': rows,
    }
    json_path = os.path.join(args.out_dir, 'benchmark_report.json')
    with open(json_path, 'w') as fh:
        json.dump(report, fh, indent=2)

    print('')
    print('== TOM TAT (exposure: %s) ==' % exposure_note)
    print('%-30s %8s %8s %9s %9s' % ('config', 'fps', 'CPU%', 'std(ms)', 'p99(ms)'))
    for r in rows:
        cfg = '%s %dx%d@%d %s' % (r['format'], r['req_w'], r['req_h'], r['req_fps'], r['backend'])
        if r['ok']:
            print('%-30s %8.1f %8.1f %9.2f %9.2f'
                  % (cfg, r['fps_actual'], r['cpu_percent'],
                     r['interval_std_ms'], r['interval_p99_ms']))
        else:
            print('%-30s %8s %8s %9s %9s' % (cfg, '-', '-', '-', '-'))
    print('')
    print('CSV : %s' % csv_path)
    print('JSON: %s' % json_path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
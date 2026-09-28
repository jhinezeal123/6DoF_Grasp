#!/usr/bin/env bash
#
# run_web.sh - Khoi dong stack ROS 2 + Web UI dieu khien myArm M750.
#
#   bash run_web.sh        (chay duoc tu bat ky thu muc nao)
#
# Script lo ca vong doi stack:
#   - Stack CHUA chay  -> tu khoi dong, va TAT khi thoat (Ctrl-C).
#   - Stack DANG chay  -> dung lai, KHONG tat khi thoat (co the la cua nguoi khac).
#
# Buoc "dang chay thi thoi" nay la bat buoc: khong co no thi moi lan Ctrl-C roi
# chay lai se de lai mot stack mo coi, lan sau se co 2-3 stack chay trung nhau
# tranh nhau dieu khien robot. Da tung gap that.
#
# Code chay qua package m750 (pip install -e . - mot lan, offline duoc vi
# dependencies de trong pyproject). `python -m m750.webui.web_control` chu
# khong phai chay file truc tiep: chay file truc tiep thi sys.path[0] la thu
# muc file chu khong phai goc repo -> ModuleNotFoundError.
#
# Moi thu doi duoc qua bien moi truong, mac dinh la duong dan co dinh cua ktmt.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LAB="${MYARM_LAB:-/home/ktmt-agx-xv/Data/khoanhd/MyArmM750_Controller_Lab}"
PY="${MYARM_PY:-$LAB/myarm_venv/bin/python}"
ROS_SETUP="${MYARM_ROS_SETUP:-/opt/ros/foxy/setup.bash}"
WS_SETUP="$LAB/myarm_sdk/ros2_ws/install/setup.bash"

STACK_LOG="${MYARM_STACK_LOG:-/tmp/myarm_stack.log}"
READY_TIMEOUT_S="${MYARM_READY_TIMEOUT_S:-90}"
# Cac node cua stack deu chay tu thu muc install nay -> mot mau pgrep la du.
STACK_PATTERN="myarm_sdk/ros2_ws/install"

for f in "$PY" "$ROS_SETUP" "$WS_SETUP"; do
  if [ ! -e "$f" ]; then
    echo "run_web.sh: khong thay $f" >&2
    exit 1
  fi
done

# setup.bash cua ROS 2 KHONG an toan voi `set -u`: no doc cac bien chua khai bao
# (vd AMENT_TRACE_SETUP_FILES) nen se chet ngay dong dau. Phai tat -u khi source.
set +u
# shellcheck disable=SC1090
source "$ROS_SETUP"
# shellcheck disable=SC1090
source "$WS_SETUP"
set -u

# Device trong ros_bridge.py la co dinh; domain phai khop stack.
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-10}"
# cv2/pymycobot can libgomp; thieu la chet ngay khi import.
export LD_PRELOAD="${LD_PRELOAD:-/lib/aarch64-linux-gnu/libgomp.so.1}"
# Jetson khong co man hinh -> render headless bang EGL.
export MUJOCO_GL="${MUJOCO_GL:-egl}"

stack_running() { pgrep -f "$STACK_PATTERN" >/dev/null 2>&1; }

# Bat dien servo TRUOC khi launch stack.
#
# Vi sao can: driver KHONG the khoi dong khi servo mat dien. connect() coi
# set_fresh_mode that bai (tra -1 khi servo khong phan hoi) la loi ket noi, roi
# _disconnect_vendor_after_failed_connect() xoa luon self._vendor. Driver chi
# ghi log "startup connection failed" roi return, khong retry. Ma power_on()
# cua adapter mo dau bang _require_vendor() -> "MyArmM750RobotArm is
# disconnected". Nen khi tay mat dien thi service /myarm/robot/power_on KHONG
# con duong nao goi duoc: khong co handle serial nao de gui lenh.
#
# Do do phai bat dien tu ngoai, bang chinh pymycobot, VA phai tra port lai
# truoc khi driver mo (chi mot tien trinh duoc giu /dev/ttyACM1).
#
# ponytail: day la workaround co y thuc. Goc nam o connect() cua adapter
# (myarm_m750_robot_arm.py) — no tu choi hoan tat khi tay chua co dien, nen
# trang thai "connected + unpowered" ma chinh adapter da dinh nghia
# (_require_powered_for_motion) khong bao gio dat toi duoc.
#
# Best-effort: loi thi chi canh bao roi di tiep, de stack tu bao loi that su.
ensure_servo_power() {
  "$PY" - "$REPO_ROOT" <<'PYEOF' || true
import sys, time
sys.path.insert(0, sys.argv[1] + "/src")
from m750.spec import DEFAULT_PORT as SERIAL_PORT

# Phai khop plugin_adapter/robot_arm/config/myarm_m750_robot_arm.yaml cua lab.
# MyArmMControl mac dinh 115200, khong phai 1000000 -> se khong noi duoc voi tay.
BAUDRATE = 1000000

try:
    from pymycobot import MyArmMControl
except Exception as e:
    print("run_web.sh: khong import duoc pymycobot (%s), bo qua pre-flight." % e)
    raise SystemExit(0)

try:
    arm = MyArmMControl(SERIAL_PORT, baudrate=BAUDRATE)
    time.sleep(0.5)
    if arm.is_powered_on() == 1:
        print("run_web.sh: servo da co dien.")
        raise SystemExit(0)
    print("run_web.sh: servo mat dien -> bat nguon...")
    # power_on() tra ve -1 ke ca khi THANH CONG (API vendor) -> phai doc lai
    # is_powered_on() moi biet ket qua that.
    arm.power_on()
    time.sleep(2.5)
    print("run_web.sh: is_powered_on =", arm.is_powered_on())
except SystemExit:
    raise
except Exception as e:
    print("run_web.sh: pre-flight nguon servo loi (%s), bo qua." % e)
PYEOF
  # Nha tty cho OS truoc khi driver mo lai.
  sleep 1
}

started_by_us=0
ui_pid=""
cleaned=0
cleanup() {
  [ "$cleaned" = "1" ] && return
  cleaned=1
  # UI chay nen chu khong exec: Ctrl-C o terminal thi python cung nhan SIGINT va
  # tu thoat, nhung khi script bi kill -INT tu noi khac thi chi bash nhan duoc ->
  # phai tu don con cua minh, neu khong no mo coi va giu port 8080.
  if [ -n "$ui_pid" ] && kill -0 "$ui_pid" 2>/dev/null; then
    kill -9 "$ui_pid" 2>/dev/null || true
  fi
  if [ "$started_by_us" = "1" ]; then
    echo
    echo "run_web.sh: tat stack ROS..."
    pkill -9 -f "$STACK_PATTERN" 2>/dev/null || true
    pkill -9 -f "ros2 launch myarm" 2>/dev/null || true
    pkill -9 -f robot_state_publisher 2>/dev/null || true
  fi
}
# INT/TERM -> thoat, roi EXIT trap lo phan don. Khong dat cleanup thang vao INT
# vi sau trap script se chay tiep chu khong dung.
trap cleanup EXIT
trap 'exit 130' INT TERM

if stack_running; then
  echo "run_web.sh: stack ROS da chay san, dung lai (se KHONG tat khi thoat)."
else
  # Chi chay khi stack CHUA chay: neu driver dang giu port thi mo se that bai.
  ensure_servo_power
  echo "run_web.sh: khoi dong stack ROS... (log: $STACK_LOG)"
  # setsid: tach khoi process group cua script, de Ctrl-C khong giet nua chung
  # stack giua luc dang khoi dong. Viec don dep do cleanup() lo.
  setsid nohup ros2 launch myarm_bringup myarm_system.launch.py \
      > "$STACK_LOG" 2>&1 < /dev/null &
  started_by_us=1
fi

cd "$REPO_ROOT"

# Cai package m750 vao venv (idempotent, offline: dependencies de trong).
# Khong cai lai moi lan chi phi dang ke; pip nhanh khi da cai.
"$PY" -m pip install -e "$REPO_ROOT" --no-deps --quiet 2>/dev/null \
  || "$PY" -m pip install -e "$REPO_ROOT" --no-deps

# Cho stack THAT SU phat du lieu, khong chi cho tien trinh song.
#
# KHONG dung `ros2 node list` de kiem tra: no di qua daemon cua ros2 CLI, ma daemon
# giu cache graph cu. Sau khi stack cu bi kill, daemon van bao node cu trong vai
# giay -> script tuong stack da san sang roi mo UI len truoc khi co du lieu.
# rclpy noi thang DDS nen khong dinh cache, va cho dung thu UI can: feedback khop.
echo "run_web.sh: cho stack phat feedback khop (toi da ${READY_TIMEOUT_S}s)..."
READY_ERR="$(mktemp)"
if ! "$PY" - "$REPO_ROOT" "$READY_TIMEOUT_S" >/dev/null 2>"$READY_ERR" <<'PYEOF'
import sys
sys.path.insert(0, sys.argv[1] + "/src")
from m750.ros.bridge import get_bridge
raise SystemExit(0 if get_bridge().wait_for_state(float(sys.argv[2])) else 1)
PYEOF
then
  echo "run_web.sh: stack khong phat feedback khop sau ${READY_TIMEOUT_S}s." >&2
  tail -5 "$READY_ERR" >&2 || true
  echo "--- 20 dong cuoi $STACK_LOG ---" >&2
  tail -20 "$STACK_LOG" >&2 || true
  rm -f "$READY_ERR"
  exit 1
fi
rm -f "$READY_ERR"
echo "run_web.sh: stack da san sang."

echo "run_web.sh: http://localhost:8080  (Ctrl-C de tat)"
# Chay nen + wait (khong dung `exec`) de con kip don dep khi thoat.
"$PY" -u -m m750.webui.web_control &
ui_pid=$!
wait "$ui_pid" || true

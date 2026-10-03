#!/usr/bin/env bash
# Dùng đúng environment của 6DoF, không dùng environment inference.
set -euo pipefail
robot_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
robot_python="${M750_PYTHON:-$robot_root/.venv/bin/python}"
if [[ ! -x "$robot_python" ]]; then
  echo "Thiếu Python của 6DoF. Chạy: bash scripts/setup-env.sh" >&2
  exit 1
fi
exec "$robot_python" "$robot_root/robot.py" "$@"

#!/usr/bin/env python3
"""Chạy CLI từ root checkout, kể cả khi chưa cài package editable."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from m750.robot.cli import main

if __name__ == "__main__":
    raise SystemExit(main())

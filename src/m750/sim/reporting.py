"""Đọc provenance và ghi báo cáo; không điều khiển robot hay chạy inference."""

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np


def commit(path):
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def json_default(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    raise TypeError(type(x).__name__)


def save_report(data, out):
    out.mkdir(parents=True, exist_ok=True)
    path = out / "report.json"
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8"
    )
    return path


def candidate_dict(g):
    return {
        "score": float(g.score),
        "width_m": float(g.width_m),
        "position_camera_m": list(g.position_m),
        "quaternion_xyzw": list(g.quaternion_xyzw),
        "metadata": dict(g.metadata),
    }

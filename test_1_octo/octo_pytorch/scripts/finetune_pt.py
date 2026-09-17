"""Compatibility wrapper for the old PyTorch finetune entrypoint.

Use the same optimized backend as `scripts/train_pt.py`.
Example:
  python scripts/finetune_pt.py     --config=scripts/configs/finetune_pt_myarm_stage_a.py     --dataset_json=scripts/datasets/myarm_stage_a_single_dataset.json
"""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from absl import app
import train_pt as _train_pt


def main(_):
    _train_pt.main(_)


if __name__ == "__main__":
    app.run(main)

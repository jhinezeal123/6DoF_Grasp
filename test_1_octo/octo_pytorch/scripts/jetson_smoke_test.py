"""Torch-only smoke test for Jetson.

Usage:
  python scripts/jetson_smoke_test.py --ckpt /path/to/torch-checkpoint

This verifies:
  * the repo can be imported without JAX/TF
  * torch sees CUDA
  * model forward works on CUDA with a dummy batch (shapes from example_batch)
"""

import argparse
import copy
import os
import sys
from pathlib import Path

# Keep this torch-only: prevent Transformers from importing TF/Flax/JAX.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TRANSFORMERS_NO_JAX", "1")

# Ensure we import local repo, not an older editable install.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from octo.model.octo_model_pt import OctoModelPt
from octo.utils.train_utils_pt import _to_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Torch checkpoint directory produced by conversion")
    args = p.parse_args()

    assert torch.cuda.is_available(), "CUDA not available in this PyTorch build"
    device = torch.device("cuda")
    print("CUDA device:", torch.cuda.get_device_name(0))

    model = OctoModelPt.load_pretrained(args.ckpt)["octo_model"].to(device)
    model.eval()

    # Build dummy inputs from example_batch (correct shapes)
    example_obs = copy.deepcopy(model.example_batch["observation"])
    example_task = copy.deepcopy(model.example_batch["task"])
    tmask = example_obs["timestep_pad_mask"]

    # Move to device
    example_obs = _to_device(example_obs, device)
    example_task = _to_device(example_task, device)
    tmask = tmask.to(device)

    # Run the same path used by inference: forward() with transformer_only=True.
    # OctoModelPt exposes `forward` (via __call__), not `run_transformer`.
    with torch.inference_mode():
        transformer_outputs, _ = model(
            observations=example_obs,
            tasks=example_task,
            timestep_pad_mask=tmask,
            train=False,
            transformer_only=True,
            save_attention_mask=False,
        )

    # Basic sanity check: ensure we got the action readout token group.
    assert "readout_action" in transformer_outputs, "Missing readout_action in transformer outputs"

    print("OK: transformer forward ran on CUDA")


if __name__ == "__main__":
    main()

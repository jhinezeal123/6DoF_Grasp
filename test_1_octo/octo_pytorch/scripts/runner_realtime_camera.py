"""Realtime camera inference runner (OpenCV/V4L2) for Torch Octo checkpoints.

This is meant for Jetson-style deployments where you want a simple rollout loop:
- Maintain an observation window (default=2)
- Read camera frames continuously
- Run `model.sample_actions()` each step
- Print the predicted action chunk (horizon x dim)

Notes
-----
* OpenCV on Jetson is often built without CUDA (that's fine for capture/resize).
* This runner is torch-only. It disables TF/Flax/JAX backends in Transformers.

Examples
--------
1) Language-conditioned, primary camera only:

  python scripts/runner_realtime_camera.py \
    --ckpt ./checkpoints/octo-small-1.5-torch \
    --text "pick up the fork" \
    --dataset bridge_dataset \
    --cam_primary 0 \
    --window 2

2) Goal-conditioned (provide goal images), primary + wrist cameras:

  python scripts/runner_realtime_camera.py \
    --ckpt ./checkpoints/octo-small-1.5-torch \
    --goal_primary ./assets/goal_primary.png \
    --goal_wrist ./assets/goal_wrist.png \
    --dataset bridge_dataset \
    --cam_primary 0 \
    --cam_wrist 2 \
    --window 2

Press Ctrl+C to stop.
"""

import argparse
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Deque, Optional, Tuple

# Keep this torch-only: prevent Transformers from importing TF/Flax/JAX.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TRANSFORMERS_NO_JAX", "1")

# Ensure we import local repo, not an older editable install.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from octo.model.octo_model_pt import OctoModelPt
from octo.utils.train_utils_pt import _to_device


def _load_rgb_uint8(path: str, size_wh: Tuple[int, int]) -> np.ndarray:
    from PIL import Image

    img = Image.open(path).convert("RGB")
    if size_wh is not None:
        img = img.resize(size_wh, Image.LANCZOS)
    arr = np.asarray(img, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected RGB image at {path}, got shape {arr.shape}")
    return arr


def _frame_to_rgb_chw_uint8(frame_bgr: np.ndarray, size_wh: Tuple[int, int]) -> np.ndarray:
    """BGR HWC -> RGB CHW uint8 resized."""
    import cv2

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    if size_wh is not None:
        frame_rgb = cv2.resize(frame_rgb, size_wh, interpolation=cv2.INTER_LINEAR)
    # HWC -> CHW
    return frame_rgb.transpose(2, 0, 1).astype(np.uint8, copy=False)


def _make_observation_from_example(
    model: OctoModelPt,
    window: int,
    primary_seq_chw: np.ndarray,
    wrist_seq_chw: Optional[np.ndarray],
    device: torch.device,
) -> dict:
    """Build observation dict from model.example_batch, overriding images and masks."""
    ex_obs = model.example_batch["observation"]
    out = {}

    # Allocate zeros like example (batch=1, time=window)
    for k, v in ex_obs.items():
        if k == "pad_mask_dict":
            continue
        if k == "timestep_pad_mask":
            out[k] = torch.ones((1, window), dtype=torch.bool)
            continue
        if isinstance(v, torch.Tensor):
            out[k] = torch.zeros((1, window, *v.shape[2:]), dtype=v.dtype)

    # Override images
    out["image_primary"] = torch.from_numpy(primary_seq_chw).unsqueeze(0).to(dtype=torch.uint8)
    if wrist_seq_chw is not None and "image_wrist" in ex_obs:
        out["image_wrist"] = torch.from_numpy(wrist_seq_chw).unsqueeze(0).to(dtype=torch.uint8)

    # Proper pad masks
    pad_mask_dict = {
        "image_primary": torch.ones((1, window), dtype=torch.bool),
        "timestep": torch.ones((1, window), dtype=torch.bool),
    }
    if "image_wrist" in out:
        pad_mask_dict["image_wrist"] = torch.ones((1, window), dtype=torch.bool)
    out["pad_mask_dict"] = pad_mask_dict

    return _to_device(out, device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Torch checkpoint directory")
    ap.add_argument("--dataset", type=str, default=None, help="Dataset key for unnormalization stats")
    ap.add_argument("--window", type=int, default=2)
    ap.add_argument("--cam_primary", type=int, default=0, help="OpenCV camera index for primary")
    ap.add_argument("--cam_wrist", type=int, default=None, help="Optional OpenCV camera index for wrist")
    ap.add_argument("--cap_width", type=int, default=640)
    ap.add_argument("--cap_height", type=int, default=480)
    ap.add_argument("--cap_fps", type=int, default=30)
    ap.add_argument("--text", type=str, default=None, help="Language instruction")
    ap.add_argument("--goal_primary", type=str, default=None, help="Path to goal primary image")
    ap.add_argument("--goal_wrist", type=str, default=None, help="Optional path to goal wrist image")
    ap.add_argument("--fp16", action="store_true", help="Use autocast fp16 for inference")
    ap.add_argument("--max_steps", type=int, default=0, help="0 = run forever")
    ap.add_argument("--print_every", type=int, default=1)
    args = ap.parse_args()

    if args.text is None and args.goal_primary is None:
        raise SystemExit("Provide at least one of --text or --goal_primary")

    import cv2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("CUDA device:", torch.cuda.get_device_name(0))
    else:
        print("Running on CPU")

    model = OctoModelPt.load_pretrained(Path(args.ckpt))["octo_model"].to(device)
    model.eval()

    # Determine expected input sizes from example batch
    ex_obs = model.example_batch["observation"]
    if "image_primary" not in ex_obs:
        raise KeyError("Model does not expect image_primary in observations")
    _, _, _, hp, wp = ex_obs["image_primary"].shape
    primary_size_wh = (wp, hp)

    wrist_size_wh = None
    if args.cam_wrist is not None:
        if "image_wrist" not in ex_obs:
            raise KeyError("Model does not expect image_wrist in observations")
        _, _, _, hw, ww = ex_obs["image_wrist"].shape
        wrist_size_wh = (ww, hw)

    # Build tasks ONCE (constant across rollout)
    goals = None
    if args.goal_primary is not None:
        ex_task = model.example_batch["task"]
        if "image_primary" not in ex_task:
            raise KeyError("Model does not expect goal image_primary in tasks")
        _, _, gh, gw = ex_task["image_primary"].shape
        goal_hw = (gw, gh)
        goals = {"image_primary": _load_rgb_uint8(args.goal_primary, goal_hw)[None]}
        if args.goal_wrist is not None:
            if "image_wrist" not in ex_task:
                raise KeyError("Model does not expect goal image_wrist in tasks")
            _, _, wh, ww = ex_task["image_wrist"].shape
            goals["image_wrist"] = _load_rgb_uint8(args.goal_wrist, (ww, wh))[None]

    texts = [args.text] if args.text is not None else None
    task = model.create_tasks(goals=goals, texts=texts, device=device)

    # Optional unnormalization
    stats = None
    if args.dataset is not None:
        if args.dataset not in model.dataset_statistics:
            print("Available dataset_statistics keys:", list(model.dataset_statistics.keys()))
            raise KeyError(f"Unknown dataset key: {args.dataset}")
        stats = model.dataset_statistics[args.dataset]["action"]

    # Setup cameras
    cap_p = cv2.VideoCapture(args.cam_primary)
    cap_p.set(cv2.CAP_PROP_FRAME_WIDTH, args.cap_width)
    cap_p.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cap_height)
    cap_p.set(cv2.CAP_PROP_FPS, args.cap_fps)
    if not cap_p.isOpened():
        raise RuntimeError(f"Failed to open primary camera index {args.cam_primary}")

    cap_w = None
    if args.cam_wrist is not None:
        cap_w = cv2.VideoCapture(args.cam_wrist)
        cap_w.set(cv2.CAP_PROP_FRAME_WIDTH, args.cap_width)
        cap_w.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cap_height)
        cap_w.set(cv2.CAP_PROP_FPS, args.cap_fps)
        if not cap_w.isOpened():
            raise RuntimeError(f"Failed to open wrist camera index {args.cam_wrist}")

    buf_p: Deque[np.ndarray] = deque(maxlen=args.window)
    buf_w: Optional[Deque[np.ndarray]] = deque(maxlen=args.window) if cap_w is not None else None

    step = 0
    t0 = time.time()
    try:
        while True:
            okp, frame_p = cap_p.read()
            if not okp:
                print("WARN: failed to read primary frame")
                continue
            chw_p = _frame_to_rgb_chw_uint8(frame_p, primary_size_wh)
            buf_p.append(chw_p)

            chw_w = None
            if cap_w is not None and buf_w is not None:
                okw, frame_w = cap_w.read()
                if not okw:
                    print("WARN: failed to read wrist frame")
                else:
                    chw_w = _frame_to_rgb_chw_uint8(frame_w, wrist_size_wh)
                    buf_w.append(chw_w)

            if len(buf_p) < args.window:
                continue
            if buf_w is not None and len(buf_w) < args.window:
                continue

            primary_seq = np.stack(list(buf_p), axis=0)  # (T,3,H,W)
            wrist_seq = np.stack(list(buf_w), axis=0) if buf_w is not None else None

            obs = _make_observation_from_example(
                model=model,
                window=args.window,
                primary_seq_chw=primary_seq,
                wrist_seq_chw=wrist_seq,
                device=device,
            )

            with torch.inference_mode():
                if args.fp16 and device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        action = model.sample_actions(
                            obs,
                            task,
                            unnormalization_statistics=stats,
                        )
                else:
                    action = model.sample_actions(
                        obs,
                        task,
                        unnormalization_statistics=stats,
                    )

            if step % max(args.print_every, 1) == 0:
                dt = time.time() - t0
                fps = (step + 1) / max(dt, 1e-9)
                # action is (1, horizon, dim)
                a0 = action[0].detach().cpu().numpy()
                print(f"step={step} fps~{fps:.2f} action[0]=\n{a0}")

            step += 1
            if args.max_steps and step >= args.max_steps:
                break

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        cap_p.release()
        if cap_w is not None:
            cap_w.release()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Registers TFDS builders (Stage-A in particular).
import octo.data.stage_a_tfds  # noqa: F401


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Export a single episode from a TFDS/RLDS dataset to a debug video (MP4). "
            "Overlays instruction/timestamps/actions/states for debugging."
        )
    )
    ap.add_argument(
        "--builder_name",
        required=True,
        help=(
            "TFDS builder name, optionally with config (e.g. myarm_stage_a_dataset/commanded_no_noops)."
        ),
    )
    ap.add_argument(
        "--builder_data_dir",
        required=True,
        help=(
            "TFDS data_dir used during build (the same --builder_data_dir you passed to scripts/build_dataset.py)."
        ),
    )
    ap.add_argument(
        "--split",
        default="train",
        help="TFDS split to read (e.g. train or val). Default: train.",
    )

    sel = ap.add_mutually_exclusive_group(required=False)
    sel.add_argument(
        "--episode_index",
        type=int,
        default=0,
        help="Episode index (0-based) within the split. Default: 0.",
    )
    sel.add_argument(
        "--episode_id",
        default=None,
        help='Episode id to match (e.g. "episode_000100").',
    )

    ap.add_argument(
        "--view",
        choices=["primary", "wrist", "side_by_side"],
        default="primary",
        help="Which camera view to render. Default: primary.",
    )
    ap.add_argument(
        "--robot_config",
        default=None,
        help=(
            "Optional robot config JSON (used to pick output fps via rollout.control_rate_hz or robot.control_period_s)."
        ),
    )
    ap.add_argument(
        "--fps",
        type=float,
        default=None,
        help=(
            "Override output FPS. If not provided, uses robot config control rate (if given) "
            "otherwise infers from timestamps."
        ),
    )
    ap.add_argument(
        "--output",
        required=True,
        help="Output video path (.mp4 recommended).",
    )
    ap.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Optional cap on number of frames exported.",
    )
    ap.add_argument(
        "--resize_height",
        type=int,
        default=None,
        help="Optional resize output frames to this height while preserving aspect ratio.",
    )
    return ap.parse_args()


def _read_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _maybe_decode_str(value: Any) -> str:
    if value is None:
        return ""
    # tf.Tensor scalar
    if hasattr(value, "numpy"):
        value = value.numpy()
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _maybe_to_numpy(value: Any) -> Any:
    if hasattr(value, "numpy"):
        return value.numpy()
    return value


def _ensure_uint8_rgb(image: Any) -> np.ndarray:
    image = _maybe_to_numpy(image)

    if isinstance(image, (bytes, bytearray)):
        img = Image.open(io.BytesIO(image)).convert("RGB")
        return np.asarray(img, dtype=np.uint8)

    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
        raise ValueError(f"Unexpected image shape: {arr.shape}")
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _stack_side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.shape[0] != right.shape[0]:
        # pad shorter image to match height
        h = max(left.shape[0], right.shape[0])
        left = _pad_to_height(left, h)
        right = _pad_to_height(right, h)
    return np.concatenate([left, right], axis=1)


def _pad_to_height(image: np.ndarray, height: int) -> np.ndarray:
    if image.shape[0] >= height:
        return image
    pad = height - image.shape[0]
    top = pad // 2
    bottom = pad - top
    return np.pad(image, ((top, bottom), (0, 0), (0, 0)), mode="constant", constant_values=0)


def _resize_to_height(image: np.ndarray, height: int) -> np.ndarray:
    if height is None or height <= 0 or image.shape[0] == height:
        return image
    img = Image.fromarray(image)
    w = int(round(image.shape[1] * (height / float(image.shape[0]))))
    img = img.resize((w, height), resample=Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def _draw_overlay(image: np.ndarray, lines: List[str]) -> np.ndarray:
    base = Image.fromarray(image).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    margin = 8
    line_h = 14
    # rough width estimate
    text = "\n".join(lines)
    # Pillow versions vary in API; try bbox first, then fall back.
    try:
        bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=2)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        w, h = draw.multiline_textsize(text, font=font, spacing=2)
    rect = (margin - 4, margin - 4, margin + w + 6, margin + h + 6)
    draw.rectangle(rect, fill=(0, 0, 0, 160))
    draw.multiline_text((margin, margin), text, fill=(255, 255, 255, 255), font=font, spacing=2)

    out = Image.alpha_composite(base, overlay).convert("RGB")
    return np.asarray(out, dtype=np.uint8)


def _infer_hz_from_timestamps(timestamps_s: List[float]) -> Optional[float]:
    if len(timestamps_s) < 3:
        return None
    ts = np.asarray(timestamps_s, dtype=np.float64)
    dt = np.diff(ts)
    dt = dt[np.isfinite(dt) & (dt > 1e-6)]
    if dt.size == 0:
        return None
    return float(1.0 / np.median(dt))


def _control_hz_from_robot_config(robot_cfg: Dict[str, Any]) -> Optional[float]:
    rollout = robot_cfg.get("rollout") if isinstance(robot_cfg, dict) else None
    if isinstance(rollout, dict) and "control_rate_hz" in rollout:
        try:
            return float(rollout["control_rate_hz"])
        except Exception:
            pass

    robot = robot_cfg.get("robot") if isinstance(robot_cfg, dict) else None
    if isinstance(robot, dict):
        if "control_period_s" in robot:
            try:
                period = float(robot["control_period_s"])
                if period > 1e-9:
                    return float(1.0 / period)
            except Exception:
                pass
        if "control_hz" in robot:
            try:
                return float(robot["control_hz"])
            except Exception:
                pass
    return None


def _format_vec(vec: Any, precision: int = 4) -> str:
    arr = np.asarray(vec, dtype=np.float64).reshape(-1)
    return np.array2string(arr, precision=precision, separator=", ")


def _pick_episode(
    episodes: Iterable[Any],
    *,
    want_index: Optional[int],
    want_id: Optional[str],
) -> Tuple[int, Any]:
    if want_id is not None:
        want_id = str(want_id)

    for idx, ep in enumerate(episodes):
        if want_id is None and want_index is not None and idx == want_index:
            return idx, ep

        if want_id is not None:
            meta = ep.get("episode_metadata", {})
            ep_id = _maybe_decode_str(meta.get("episode_id"))
            if ep_id == want_id:
                return idx, ep

    raise ValueError(
        f"Episode not found. want_index={want_index} want_id={want_id}. "
        "Check --split or the episode_id values in TFDS."
    )


def _iter_steps(step_ds: Any, *, max_steps: Optional[int]) -> List[Dict[str, Any]]:
    steps: List[Dict[str, Any]] = []
    for i, step in enumerate(step_ds):
        if max_steps is not None and i >= int(max_steps):
            break
        # Keep tensors as-is; decode later.
        steps.append(step)
    return steps


def main() -> None:
    args = _parse_args()

    import tensorflow as tf  # local import to keep import errors readable
    import tensorflow_datasets as tfds
    import imageio.v2 as imageio

    builder_data_dir = Path(args.builder_data_dir).resolve()
    builder = tfds.builder(args.builder_name, data_dir=str(builder_data_dir))

    split = str(args.split)
    ds = builder.as_dataset(split=split, shuffle_files=False)

    want_index = int(args.episode_index) if args.episode_id is None else None
    want_id = str(args.episode_id) if args.episode_id is not None else None

    # TFDS returns nested datasets for RLDS `steps`; eager iteration is simplest.
    ep_idx, episode = _pick_episode(ds, want_index=want_index, want_id=want_id)

    meta = episode.get("episode_metadata", {})
    episode_id = _maybe_decode_str(meta.get("episode_id"))
    instruction = _maybe_decode_str(meta.get("instruction"))

    steps = _iter_steps(episode["steps"], max_steps=args.max_steps)
    if not steps:
        raise ValueError(f"Episode {episode_id or ep_idx} has zero steps")

    timestamps_s: List[float] = []
    for step in steps:
        t = _maybe_to_numpy(step.get("timestamp"))
        if t is None:
            continue
        try:
            timestamps_s.append(float(t))
        except Exception:
            pass

    robot_cfg = _read_json(args.robot_config) if args.robot_config else None
    hz_cfg = _control_hz_from_robot_config(robot_cfg) if robot_cfg else None
    hz_ts = _infer_hz_from_timestamps(timestamps_s) if timestamps_s else None

    fps_out = float(args.fps) if args.fps is not None else (hz_cfg or hz_ts or 5.0)
    if fps_out <= 0:
        raise ValueError(f"Invalid fps: {fps_out}")

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        writer = imageio.get_writer(str(output_path), fps=fps_out)
    except Exception as e:
        raise RuntimeError(
            f"Failed to open video writer for {output_path}: {e}. "
            "If you are exporting .mp4, make sure ffmpeg is installed and imageio-ffmpeg is available."
        )

    try:
        prev_t: Optional[float] = None
        for step_idx, step in enumerate(steps):
            obs = step.get("observation", {})

            img_primary = obs.get("image_primary")
            img_wrist = obs.get("image_wrist")

            if args.view == "wrist":
                frame = _ensure_uint8_rgb(img_wrist)
            elif args.view == "side_by_side":
                left = _ensure_uint8_rgb(img_primary)
                right = _ensure_uint8_rgb(img_wrist)
                frame = _stack_side_by_side(left, right)
            else:
                frame = _ensure_uint8_rgb(img_primary)

            state = _maybe_to_numpy(obs.get("state"))
            commanded_state = _maybe_to_numpy(obs.get("commanded_state"))
            action = _maybe_to_numpy(step.get("action"))

            t = _maybe_to_numpy(step.get("timestamp"))
            t_s = float(t) if t is not None else float(step_idx) / fps_out
            dt_s = float(t_s - prev_t) if prev_t is not None else 0.0
            prev_t = t_s

            lines: List[str] = []
            lines.append(f"{builder.info.full_name} | split={split} | ep_idx={ep_idx}")
            if episode_id:
                lines.append(f"episode_id: {episode_id}")
            if instruction:
                lines.append(f"instruction: {instruction}")
            lines.append(
                f"step {step_idx + 1}/{len(steps)} | t={t_s:.3f}s | dt={dt_s:.3f}s | fps_out={fps_out:.3f}"
            )
            if hz_cfg is not None:
                lines.append(f"control_hz(config): {hz_cfg:.3f}")
            if hz_ts is not None:
                lines.append(f"approx_hz(timestamps): {hz_ts:.3f}")

            if state is not None:
                lines.append(f"achieved_state_7d: {_format_vec(state, precision=4)}")
            if commanded_state is not None:
                lines.append(f"commanded_state_7d: {_format_vec(commanded_state, precision=4)}")
                if state is not None:
                    err = np.asarray(state, dtype=np.float64) - np.asarray(commanded_state, dtype=np.float64)
                    lines.append(f"tracking_error: {_format_vec(err, precision=4)}")
                    err_xyz = float(np.linalg.norm(err[:3]))
                    err_rot = float(np.linalg.norm(err[3:6]))
                    lines.append(f"|err_xyz|={err_xyz:.6f} m | |err_rot|={err_rot:.6f} rad")

            if action is not None:
                lines.append(f"action_7d (dxyz,drpy,grip): {_format_vec(action, precision=4)}")
                act = np.asarray(action, dtype=np.float64).reshape(-1)
                if act.size >= 6:
                    act_xyz = float(np.linalg.norm(act[:3]))
                    act_rot = float(np.linalg.norm(act[3:6]))
                    lines.append(f"|dxyz|={act_xyz:.6f} m | |drpy|={act_rot:.6f} rad")

            frame = _draw_overlay(frame, lines)
            frame = _resize_to_height(frame, args.resize_height)
            writer.append_data(frame)

    finally:
        writer.close()

    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()

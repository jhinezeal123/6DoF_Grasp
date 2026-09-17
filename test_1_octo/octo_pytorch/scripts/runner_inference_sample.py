"""Sample inference runner for a Torch-converted Octo checkpoint.

This script is meant to run on Jetson (torch-only) or any CUDA machine.

Examples
--------

Language-conditioned (no goal image):

  python scripts/runner_inference_sample.py \
    --ckpt ./checkpoints/octo-small-1.5-torch \
    --obs_primary ./assets/obs_primary.png \
    --text "pick up the fork" \
    --dataset bridge_dataset \
    --window 2

Goal-conditioned:

  python scripts/runner_inference_sample.py \
    --ckpt ./checkpoints/octo-small-1.5-torch \
    --obs_primary ./assets/obs_primary.png \
    --goal_primary ./assets/goal_primary.png \
    --dataset bridge_dataset \
    --window 2

Notes
-----
* If you provide a single observation image but window>1, the image is repeated.
* For maximum compatibility, this script fills any missing observation keys with zeros
  using shapes from model.example_batch.
"""

import argparse
import gc
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Prevent HuggingFace Transformers from importing TensorFlow/Flax/JAX backends.
# This keeps the runner torch-only and avoids TF/XLA side-effect logs.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TRANSFORMERS_NO_JAX", "1")

# Ensure we import the *local* repo (avoid accidentally using an older editable install).
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from octo.model.octo_model_pt import OctoModelPt
from octo.model.components.attention_rules import AttentionRule
from octo.model.components.base_pt import TokenGroupPt
from octo.model.components.block_transformer_pt import PrefixGroupPt, TimestepGroupPt
from octo.utils.train_utils_pt import _to_device


def _parse_paths(s: Optional[str]) -> Optional[List[str]]:
    if s is None:
        return None
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return parts if parts else None


def _load_rgb_uint8(path: str, size_wh: Tuple[int, int]) -> np.ndarray:
    from PIL import Image

    img = Image.open(path).convert("RGB")
    if size_wh is not None:
        img = img.resize(size_wh, Image.LANCZOS)
    # Ensure the underlying memory is writable before converting to torch.
    # PIL-backed arrays can be read-only which triggers PyTorch warnings.
    arr = np.asarray(img, dtype=np.uint8).copy()
    # HWC
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected RGB image at {path}, got shape {arr.shape}")
    return arr


def _make_image_sequence(paths: List[str], t: int, size_wh: Tuple[int, int]) -> np.ndarray:
    if len(paths) == 1:
        paths = paths * t
    if len(paths) != t:
        raise ValueError(f"Need {t} images for the window, got {len(paths)}")
    imgs = [_load_rgb_uint8(p, size_wh=size_wh) for p in paths]
    seq = np.stack(imgs, axis=0)  # (T,H,W,3)
    # (T,3,H,W)
    return seq.transpose(0, 3, 1, 2)


def build_observation_from_example(
    model: OctoModelPt,
    window: int,
    obs_primary_paths: Optional[List[str]],
    obs_wrist_paths: Optional[List[str]],
    device: torch.device,
) -> dict:
    """Create an observation dict for batch_size=1 using model.example_batch shapes."""
    ex_obs = model.example_batch["observation"]
    out = {}

    # Determine target sizes from example batch if present
    primary_size_wh = None
    wrist_size_wh = None
    if "image_primary" in ex_obs:
        _, _, _, h, w = ex_obs["image_primary"].shape
        primary_size_wh = (w, h)
    if "image_wrist" in ex_obs:
        _, _, _, h, w = ex_obs["image_wrist"].shape
        wrist_size_wh = (w, h)

    for k, v in ex_obs.items():
        if k == "timestep_pad_mask":
            out[k] = torch.ones((1, window), dtype=torch.bool)
            continue

        if k == "pad_mask_dict":
            # We'll regenerate a proper pad_mask_dict below.
            continue

        if not isinstance(v, torch.Tensor):
            continue

        # Create zeros of the right shape (batch=1, time=window)
        target_shape = (1, window) + tuple(v.shape[2:])
        out[k] = torch.zeros(target_shape, dtype=v.dtype)

    # Inject observation images if provided
    provided_primary = False
    provided_wrist = False
    if obs_primary_paths is not None:
        if primary_size_wh is None:
            raise KeyError("Model does not expect image_primary, but --obs_primary was provided")
        seq = _make_image_sequence(obs_primary_paths, window, primary_size_wh)  # (T,3,H,W)
        out["image_primary"] = torch.from_numpy(seq).unsqueeze(0).to(dtype=torch.uint8)
        provided_primary = True

    if obs_wrist_paths is not None:
        if wrist_size_wh is None:
            raise KeyError("Model does not expect image_wrist, but --obs_wrist was provided")
        seq = _make_image_sequence(obs_wrist_paths, window, wrist_size_wh)
        out["image_wrist"] = torch.from_numpy(seq).unsqueeze(0).to(dtype=torch.uint8)
        provided_wrist = True

    # Add pad_mask_dict for proper masking inside tokenizers.
    # Expected shapes: (B, T) for per-timestep modalities.
    # IMPORTANT: If we auto-filled a modality with zeros (because it exists in example_batch)
    # but the user did not provide data for it, mark it as masked out. This avoids attention
    # seeing "valid" all-zero frames for missing cameras.
    pad_mask_dict = {}
    if "image_primary" in out:
        pad_mask_dict["image_primary"] = (
            torch.ones((1, window), dtype=torch.bool) if provided_primary
            else torch.zeros((1, window), dtype=torch.bool)
        )
    if "image_wrist" in out:
        pad_mask_dict["image_wrist"] = (
            torch.ones((1, window), dtype=torch.bool) if provided_wrist
            else torch.zeros((1, window), dtype=torch.bool)
        )
    # Timestep mask: always valid for provided timesteps.
    pad_mask_dict["timestep"] = torch.ones((1, window), dtype=torch.bool)
    out["pad_mask_dict"] = pad_mask_dict

    return _to_device(out, device)


def _bytes_to_mib(n: int) -> float:
    return float(n) / (1024.0 * 1024.0)


def _tensor_nbytes(t: Optional[torch.Tensor]) -> int:
    if t is None:
        return 0
    return int(t.numel() * t.element_size())


def _module_state_nbytes(m: torch.nn.Module) -> int:
    total = 0
    for p in m.parameters(recurse=True):
        total += int(p.numel() * p.element_size())
    for b in m.buffers(recurse=True):
        total += int(b.numel() * b.element_size())
    return total


def _cuda_mem_snapshot() -> Dict[str, int]:
    return {
        "alloc": int(torch.cuda.memory_allocated()),
        "reserved": int(torch.cuda.memory_reserved()),
        "peak_alloc": int(torch.cuda.max_memory_allocated()),
        "peak_reserved": int(torch.cuda.max_memory_reserved()),
    }


def _reset_cuda_peaks_if_needed(device: torch.device) -> None:
    if device.type != "cuda":
        return
    try:
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _sync_if_cuda(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.cuda.synchronize()


def _measure_segment(name: str, fn, device: torch.device) -> Tuple[object, Dict[str, object]]:
    """Measure wall-time + CUDA peak memory for one callable."""
    if device.type == "cuda":
        _sync_if_cuda(device)
        _reset_cuda_peaks_if_needed(device)
        start = _cuda_mem_snapshot()

        t0 = time.perf_counter()
        out = fn()
        _sync_if_cuda(device)
        t1 = time.perf_counter()

        end = _cuda_mem_snapshot()
        metrics = {
            "name": name,
            "ms": (t1 - t0) * 1000.0,
            "start_alloc": start["alloc"],
            "start_reserved": start["reserved"],
            "end_alloc": end["alloc"],
            "end_reserved": end["reserved"],
            "peak_alloc": end["peak_alloc"],
            "peak_reserved": end["peak_reserved"],
        }
        return out, metrics

    t0 = time.perf_counter()
    out = fn()
    t1 = time.perf_counter()
    metrics = {
        "name": name,
        "ms": (t1 - t0) * 1000.0,
        "start_alloc": 0,
        "start_reserved": 0,
        "end_alloc": 0,
        "end_reserved": 0,
        "peak_alloc": 0,
        "peak_reserved": 0,
    }
    return out, metrics


def _embodiment_action_dim_from_stats(stats: Optional[Dict]) -> Optional[int]:
    if stats is None:
        return None
    for k in ("mean", "p01", "p99"):
        if k in stats and isinstance(stats[k], torch.Tensor) and stats[k].ndim >= 1:
            return int(stats[k].shape[0])
    return None


def _run_transformer_breakdown_once(
    model: OctoModelPt,
    observation: Dict,
    task: Dict,
    device: torch.device,
    save_attention_mask: bool,
) -> Tuple[Dict[str, TokenGroupPt], Dict[str, Dict[str, object]]]:
    """Run tokenizers + transformer backbone, returning outputs plus per-module metrics."""
    octo_t = model.module.octo_transformer
    timestep_pad_mask = observation["timestep_pad_mask"]
    batch_size, horizon = timestep_pad_mask.shape

    task_attention_rules = {"task_*": AttentionRule.CAUSAL}
    observation_attention_rules = {
        "task_*": AttentionRule.CAUSAL,
        "obs_*": AttentionRule.CAUSAL,
    }

    all_prefix_groups: List[PrefixGroupPt] = []
    all_timestep_groups: List[TimestepGroupPt] = []
    metrics_by_name: Dict[str, Dict[str, object]] = {}

    # Task tokenizers
    for name, tok in octo_t.task_tokenizers.items():
        group_name = f"task_{name}"
        seg_name = f"task_tokenizer.{name}"

        def _fn_task_tok():
            tok_out = tok(observation, task, train=False)
            if tok_out is None:
                return None
            task_tokens = octo_t.task_projections[f"{group_name}_projection"](tok_out.tokens)
            task_tokens = octo_t._add_positional_embedding(
                task_tokens,
                getattr(octo_t, f"{group_name}_pos_embedding"),
            )
            return task_tokens, tok_out.mask

        out, m = _measure_segment(seg_name, _fn_task_tok, device=device)
        if out is None:
            m["skipped"] = True
            metrics_by_name[seg_name] = m
            continue

        task_tokens, task_mask = out
        all_prefix_groups.append(
            PrefixGroupPt(
                tokens=task_tokens,
                mask=task_mask,
                name=group_name,
                attention_rules=task_attention_rules,
            )
        )
        m["skipped"] = False
        m["out_shape"] = tuple(task_tokens.shape)
        m["out_dtype"] = str(task_tokens.dtype)
        m["out_mib"] = _bytes_to_mib(_tensor_nbytes(task_tokens))
        metrics_by_name[seg_name] = m

    # Observation tokenizers
    for name, tok in octo_t.observation_tokenizers.items():
        group_name = f"obs_{name}"
        seg_name = f"obs_tokenizer.{name}"

        def _fn_obs_tok():
            tok_out = tok(observation, task, train=False)
            if tok_out is None:
                return None
            obs_tokens = octo_t.obs_projections[f"{group_name}_projection"](tok_out.tokens)
            obs_tokens = octo_t._add_positional_embedding(
                obs_tokens,
                getattr(octo_t, f"{group_name}_pos_embedding"),
                history_dim=0,
            )
            obs_pad_mask = timestep_pad_mask.unsqueeze(-1) & tok_out.mask
            return obs_tokens, obs_pad_mask

        out, m = _measure_segment(seg_name, _fn_obs_tok, device=device)
        if out is None:
            m["skipped"] = True
            metrics_by_name[seg_name] = m
            continue

        obs_tokens, obs_mask = out
        all_timestep_groups.append(
            TimestepGroupPt(
                tokens=obs_tokens,
                mask=obs_mask,
                name=group_name,
                attention_rules=observation_attention_rules,
            )
        )
        m["skipped"] = False
        m["out_shape"] = tuple(obs_tokens.shape)
        m["out_dtype"] = str(obs_tokens.dtype)
        m["out_mib"] = _bytes_to_mib(_tensor_nbytes(obs_tokens))
        metrics_by_name[seg_name] = m

    # Optional: repeat task tokens into timestep stream
    if octo_t.repeat_task_tokens and len(all_prefix_groups) > 0:
        for task_group in list(all_prefix_groups):
            task_tokens = task_group.tokens.unsqueeze(1).expand(-1, horizon, -1, -1)
            task_pad_mask = task_group.mask.unsqueeze(1).expand(-1, horizon, -1)
            all_timestep_groups.append(
                TimestepGroupPt(
                    tokens=task_tokens,
                    mask=task_pad_mask,
                    name=f"obs_{task_group.name}",
                    attention_rules=observation_attention_rules,
                )
            )

    # Readout tokens
    readout_names = list(octo_t.readouts.keys())
    for readout_name in readout_names:
        group_name = f"readout_{readout_name}"
        seg_name = f"readout.{readout_name}"
        n_tokens_for_readout = int(octo_t.readouts[readout_name])

        def _fn_readout():
            readout_tokens = torch.zeros(
                (batch_size, horizon, n_tokens_for_readout, octo_t.token_embedding_size),
                device=device,
            )
            readout_tokens = octo_t._add_positional_embedding(
                readout_tokens,
                getattr(octo_t, f"{group_name}_pos_embedding"),
                history_dim=0,
            )
            readout_mask = torch.ones(
                (batch_size, horizon, n_tokens_for_readout),
                device=device,
                dtype=torch.bool,
            )
            return readout_tokens, readout_mask

        out, m = _measure_segment(seg_name, _fn_readout, device=device)
        readout_tokens, readout_mask = out
        readout_attention_rules = {
            "task_*": AttentionRule.CAUSAL,
            "obs_*": AttentionRule.CAUSAL,
            group_name: AttentionRule.CAUSAL,
        }
        all_timestep_groups.append(
            TimestepGroupPt(
                tokens=readout_tokens,
                mask=readout_mask,
                name=group_name,
                attention_rules=readout_attention_rules,
            )
        )
        m["out_shape"] = tuple(readout_tokens.shape)
        m["out_dtype"] = str(readout_tokens.dtype)
        m["out_mib"] = _bytes_to_mib(_tensor_nbytes(readout_tokens))
        metrics_by_name[seg_name] = m

    # Transformer backbone
    def _fn_backbone():
        return octo_t.block_transformer(
            all_prefix_groups,
            all_timestep_groups,
            train=False,
            verbose=False,
            save_attention_mask=save_attention_mask,
        )

    (prefix_outputs, timestep_outputs), m_backbone = _measure_segment(
        "transformer_backbone", _fn_backbone, device=device
    )
    metrics_by_name["transformer_backbone"] = m_backbone

    # Assemble outputs in the same format as OctoTransformerPt.forward
    outputs: Dict[str, TokenGroupPt] = {}
    outputs.update({g.name: TokenGroupPt(g.tokens, g.mask) for g in prefix_outputs})
    outputs.update({g.name: TokenGroupPt(g.tokens, g.mask) for g in timestep_outputs})

    if len(prefix_outputs) > 0:
        outputs["task"] = TokenGroupPt.concatenate(
            [TokenGroupPt(g.tokens, g.mask) for g in prefix_outputs]
        )

    outputs["obs"] = TokenGroupPt.concatenate(
        [
            TokenGroupPt(g.tokens, g.mask)
            for g in timestep_outputs
            if g.name.startswith("obs_")
        ],
        axis=-2,
    )

    return outputs, metrics_by_name


def _profile_octo(
    model: OctoModelPt,
    observation: Dict,
    task: Dict,
    stats: Optional[Dict],
    device: torch.device,
    seed: int,
    warmup: int,
    iters: int,
) -> torch.Tensor:
    octo_t = model.module.octo_transformer
    action_head = model.module.heads["action"]

    print("\n== Octo profiling (high-level modules) ==")
    print(f"device={device.type}  window={tuple(observation['timestep_pad_mask'].shape)}")

    if device.type == "cuda":
        _sync_if_cuda(device)
        base_alloc = torch.cuda.memory_allocated()
        base_res = torch.cuda.memory_reserved()
        print(
            f"baseline_vram: alloc={_bytes_to_mib(base_alloc):.1f} MiB  reserved={_bytes_to_mib(base_res):.1f} MiB"
        )

    # Parameter/buffer VRAM footprint (static)
    pos_emb_bytes = 0
    for pe_name in getattr(octo_t, "pos_embeddings_names", []):
        pe = getattr(octo_t, pe_name, None)
        if isinstance(pe, torch.Tensor):
            pos_emb_bytes += _tensor_nbytes(pe)

    print("\n[static weights (params+buffers)]")
    for name, tok in octo_t.task_tokenizers.items():
        print(f"task_tokenizer.{name:16s} {_bytes_to_mib(_module_state_nbytes(tok)):.1f} MiB")
    for name, tok in octo_t.observation_tokenizers.items():
        print(f"obs_tokenizer.{name:17s} {_bytes_to_mib(_module_state_nbytes(tok)):.1f} MiB")
    print(f"projections+pos_emb         {_bytes_to_mib(_module_state_nbytes(octo_t.task_projections) + _module_state_nbytes(octo_t.obs_projections) + pos_emb_bytes):.1f} MiB")
    print(f"transformer_backbone        {_bytes_to_mib(_module_state_nbytes(octo_t.block_transformer)):.1f} MiB")
    print(f"action_head                 {_bytes_to_mib(_module_state_nbytes(action_head)):.1f} MiB")
    print(f"TOTAL(model)                {_bytes_to_mib(_module_state_nbytes(model)):.1f} MiB")

    # Warmup to populate CUDA caches and (importantly) cache the transformer attention mask.
    if warmup > 0:
        for i in range(warmup):
            gen = torch.Generator(device=device).manual_seed(seed)
            with torch.inference_mode():
                _ = model.sample_actions(
                    observation,
                    task,
                    unnormalization_statistics=stats,
                    generator=gen,
                )
            _sync_if_cuda(device)
        gc.collect()

    # One end-to-end measurement (to include any remaining overhead not in the breakdown)
    def _fn_e2e():
        gen = torch.Generator(device=device).manual_seed(seed)
        return model.sample_actions(
            observation,
            task,
            unnormalization_statistics=stats,
            generator=gen,
        )

    action_e2e, m_e2e = _measure_segment("e2e.sample_actions", _fn_e2e, device=device)
    if device.type == "cuda":
        print(
            f"\n[e2e] {m_e2e['ms']:.2f} ms  peak_alloc={_bytes_to_mib(m_e2e['peak_alloc']):.1f} MiB  peak_reserved={_bytes_to_mib(m_e2e['peak_reserved']):.1f} MiB"
        )
    else:
        print(f"\n[e2e] {m_e2e['ms']:.2f} ms")

    # Breakdown (tokenizers -> backbone -> action head)
    acc: Dict[str, Dict[str, object]] = {}

    def _accum(seg_name: str, m: Dict[str, object]) -> None:
        rec = acc.setdefault(
            seg_name,
            {
                "ms": [],
                "delta_peak_alloc": [],
                "delta_peak_reserved": [],
                "peak_alloc": [],
                "peak_reserved": [],
                "out_shape": None,
                "out_dtype": None,
                "out_mib": None,
                "skipped": 0,
            },
        )
        rec["ms"].append(float(m["ms"]))
        rec["peak_alloc"].append(int(m["peak_alloc"]))
        rec["peak_reserved"].append(int(m["peak_reserved"]))
        rec["delta_peak_alloc"].append(int(m["peak_alloc"]) - int(m["start_alloc"]))
        rec["delta_peak_reserved"].append(int(m["peak_reserved"]) - int(m["start_reserved"]))
        if bool(m.get("skipped", False)):
            rec["skipped"] += 1
        if "out_shape" in m and m["out_shape"] is not None:
            rec["out_shape"] = m["out_shape"]
            rec["out_dtype"] = m.get("out_dtype")
            rec["out_mib"] = m.get("out_mib")

    diffusion_steps = getattr(action_head, "diffusion_steps", None)

    for _ in range(max(iters, 1)):
        # Tokenizers + transformer backbone
        with torch.inference_mode():
            transformer_outputs, seg_metrics = _run_transformer_breakdown_once(
                model=model,
                observation=observation,
                task=task,
                device=device,
                save_attention_mask=True,
            )

            for k, v in seg_metrics.items():
                _accum(k, v)

            # Action head
            embodiment_dim = _embodiment_action_dim_from_stats(stats)

            def _fn_head():
                gen = torch.Generator(device=device).manual_seed(seed)
                return action_head.predict_action(
                    transformer_outputs,
                    train=False,
                    embodiment_action_dim=embodiment_dim,
                    generator=gen,
                )

            action_pred, m_head = _measure_segment("action_head", _fn_head, device=device)
            m_head["out_shape"] = tuple(action_pred.shape)
            m_head["out_dtype"] = str(action_pred.dtype)
            m_head["out_mib"] = _bytes_to_mib(_tensor_nbytes(action_pred))
            _accum("action_head", m_head)

        # Best-effort cleanup
        del transformer_outputs
        del action_pred
        _sync_if_cuda(device)

    print("\n[breakdown: mean latency, max peak VRAM delta]")
    ordered = sorted(acc.keys())
    for seg_name in ordered:
        rec = acc[seg_name]
        ms_mean = float(np.mean(rec["ms"])) if rec["ms"] else 0.0
        delta_peak_alloc_max = max(rec["delta_peak_alloc"]) if rec["delta_peak_alloc"] else 0
        delta_peak_res_max = max(rec["delta_peak_reserved"]) if rec["delta_peak_reserved"] else 0

        extra = ""
        if rec["out_shape"] is not None:
            extra = f"  out={rec['out_shape']} {rec['out_dtype']} ({rec['out_mib']:.2f} MiB)"
        if seg_name == "action_head" and diffusion_steps is not None:
            extra = f"  diffusion_steps={diffusion_steps}" + extra
        if rec["skipped"]:
            extra = extra + f"  skipped={rec['skipped']}/{len(rec['ms'])}"

        if device.type == "cuda":
            print(
                f"{seg_name:28s}  {ms_mean:8.2f} ms  peakΔ alloc={_bytes_to_mib(delta_peak_alloc_max):7.1f} MiB  peakΔ res={_bytes_to_mib(delta_peak_res_max):7.1f} MiB{extra}"
            )
        else:
            print(f"{seg_name:28s}  {ms_mean:8.2f} ms{extra}")

    return action_e2e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Torch checkpoint directory")
    ap.add_argument("--window", type=int, default=2, help="Observation window size")
    ap.add_argument(
        "--obs_primary",
        type=str,
        default=None,
        help="Path(s) to primary obs image. Comma-separated for a sequence",
    )
    ap.add_argument(
        "--obs_wrist",
        type=str,
        default=None,
        help="Path(s) to wrist obs image. Comma-separated for a sequence",
    )
    ap.add_argument("--goal_primary", type=str, default=None, help="Path to goal primary image")
    ap.add_argument("--goal_wrist", type=str, default=None, help="Optional path to goal wrist image")
    ap.add_argument("--text", type=str, default=None, help="Language instruction")
    ap.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset key for unnormalization stats, e.g. bridge_dataset",
    )
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument(
        "--profile",
        action="store_true",
        help="Print per-module latency + VRAM (tokenizers / backbone / head)",
    )
    ap.add_argument("--profile_warmup", type=int, default=2)
    ap.add_argument("--profile_iters", type=int, default=5)

    args = ap.parse_args()

    assert args.window >= 1
    if args.text is None and args.goal_primary is None:
        raise SystemExit("Provide at least one of --text or --goal_primary")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("CUDA device:", torch.cuda.get_device_name(0))
    else:
        print("Running on CPU")

    model = OctoModelPt.load_pretrained(Path(args.ckpt))["octo_model"].to(device)
    model.eval()

    obs_primary_paths = _parse_paths(args.obs_primary)
    obs_wrist_paths = _parse_paths(args.obs_wrist)

    observation = build_observation_from_example(
        model=model,
        window=args.window,
        obs_primary_paths=obs_primary_paths,
        obs_wrist_paths=obs_wrist_paths,
        device=device,
    )

    # Build task dict
    goals = None
    if args.goal_primary is not None:
        # Resize goal to the model's expected goal image size. Use task example shape.
        ex_task = model.example_batch["task"]
        if "image_primary" not in ex_task:
            raise KeyError("Model does not expect goal image_primary in tasks")
        # ex_task[image_primary] is (B, C, H, W) in torch checkpoint
        _, _, gh, gw = ex_task["image_primary"].shape
        goal_hw = (gw, gh)
        goal_img = _load_rgb_uint8(args.goal_primary, size_wh=goal_hw)  # HWC
        goals = {"image_primary": goal_img[None]}  # add batch

        if args.goal_wrist is not None:
            if "image_wrist" not in ex_task:
                raise KeyError("Model does not expect goal image_wrist in tasks")
            _, _, wh, ww = ex_task["image_wrist"].shape
            wrist_hw = (ww, wh)
            wrist_img = _load_rgb_uint8(args.goal_wrist, size_wh=wrist_hw)
            goals["image_wrist"] = wrist_img[None]

    texts = [args.text] if args.text is not None else None
    task = model.create_tasks(goals=goals, texts=texts, device=device)

    # Ensure pad_mask_dict values are torch tensors (some older checkpoints/scripts may create numpy masks).
    if "pad_mask_dict" in task:
        for k, v in list(task["pad_mask_dict"].items()):
            if isinstance(v, np.ndarray):
                task["pad_mask_dict"][k] = torch.from_numpy(v).to(device=device)

    # Optional unnormalization
    stats = None
    if args.dataset is not None:
        if args.dataset not in model.dataset_statistics:
            print("Available dataset_statistics keys:", list(model.dataset_statistics.keys()))
            raise KeyError(f"Unknown dataset key: {args.dataset}")
        stats = model.dataset_statistics[args.dataset]["action"]

    if args.profile:
        action = _profile_octo(
            model=model,
            observation=observation,
            task=task,
            stats=stats,
            device=device,
            seed=args.seed,
            warmup=max(0, args.profile_warmup),
            iters=max(1, args.profile_iters),
        )
    else:
        gen = torch.Generator(device=device).manual_seed(args.seed)
        with torch.inference_mode():
            action = model.sample_actions(
                observation,
                task,
                unnormalization_statistics=stats,
                generator=gen,
            )

    print("\nAction shape:", tuple(action.shape))
    print("Action[0]:", action[0])


if __name__ == "__main__":
    main()

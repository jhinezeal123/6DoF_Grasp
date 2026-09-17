"""Compare JAX Octo and PyTorch Octo outputs on the same input.

This script is useful after converting a JAX checkpoint to a Torch checkpoint, or when
loading Torch weights directly from a JAX checkpoint.

Examples
--------

Compare a local torch checkpoint against the original JAX checkpoint:

  python scripts/compare_jax_torch_outputs.py \
    --jax_ckpt hf://rail-berkeley/octo-small-1.5 \
    --torch_ckpt ./checkpoints/octo-small-1.5-torch \
    --text "pick up the fork" \
    --obs_primary ./assets/obs_primary.png \
    --dataset bridge_dataset \
    --window 2

Compare without a converted torch checkpoint by instantiating the PyTorch model directly
from the JAX checkpoint:

  python scripts/compare_jax_torch_outputs.py \
    --jax_ckpt hf://rail-berkeley/octo-small-1.5 \
    --pt_from_jax \
    --text "pick up the fork" \
    --window 2

Advanced usage
--------------
You can provide arbitrary observation tensors through an .npz file with keys that match
Octo observation names, for example ``image_primary``, ``image_wrist``, ``proprio``.
Observation arrays should be stored without the batch dimension and with the first axis
representing the time window.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TRANSFORMERS_NO_JAX", "0")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from octo.model.octo_model import OctoModel
from octo.model.octo_model_pt import OctoModelPt
from octo.utils.compare_utils import (
    build_goal_overrides,
    build_jax_observation,
    compare_arrays,
    compare_bool_arrays,
    ensure_torch_task_on_device,
    format_metric_row,
    jax_obs_to_torch,
    parse_csv_paths,
    run_shared_noise_diffusion_jax,
    run_shared_noise_diffusion_torch,
    save_json,
    token_group_to_numpy,
    unnormalize_action_numpy,
)


def _summarize_transformer_outputs(jax_outputs: Dict, pt_outputs: Dict) -> Dict:
    summary = {"common": {}, "jax_only": [], "torch_only": []}
    common_keys = sorted(set(jax_outputs.keys()) & set(pt_outputs.keys()))
    summary["jax_only"] = sorted(set(jax_outputs.keys()) - set(pt_outputs.keys()))
    summary["torch_only"] = sorted(set(pt_outputs.keys()) - set(jax_outputs.keys()))

    for key in common_keys:
        jax_tokens, jax_mask = token_group_to_numpy(jax_outputs[key])
        pt_tokens, pt_mask = token_group_to_numpy(pt_outputs[key])
        summary["common"][key] = {
            "tokens": compare_arrays(jax_tokens, pt_tokens),
            "mask": compare_bool_arrays(jax_mask, pt_mask),
        }
    return summary


def _get_dataset_stats(jax_model, pt_model, dataset_name: str):
    jax_stats = None
    pt_stats = None
    if dataset_name is None:
        return None, None
    if dataset_name not in jax_model.dataset_statistics:
        raise KeyError(
            f"Dataset '{dataset_name}' not found in JAX dataset_statistics. Available: {list(jax_model.dataset_statistics.keys())}"
        )
    jax_stats = jax_model.dataset_statistics[dataset_name]["action"]
    if dataset_name not in pt_model.dataset_statistics:
        raise KeyError(
            f"Dataset '{dataset_name}' not found in Torch dataset_statistics. Available: {list(pt_model.dataset_statistics.keys())}"
        )
    pt_stats = pt_model.dataset_statistics[dataset_name]["action"]
    return jax_stats, pt_stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jax_ckpt", required=True, help="JAX checkpoint directory or hf:// path")
    ap.add_argument("--torch_ckpt", default=None, help="Converted Torch checkpoint directory")
    ap.add_argument("--pt_from_jax", action="store_true", help="Instantiate PyTorch weights directly from the JAX checkpoint")
    ap.add_argument("--window", type=int, default=2, help="Observation window size")
    ap.add_argument("--text", type=str, default=None, help="Language instruction")
    ap.add_argument("--obs_primary", type=str, default=None, help="Comma-separated primary observation image path(s)")
    ap.add_argument("--obs_wrist", type=str, default=None, help="Comma-separated wrist observation image path(s)")
    ap.add_argument("--goal_primary", type=str, default=None, help="Goal primary image path")
    ap.add_argument("--goal_wrist", type=str, default=None, help="Goal wrist image path")
    ap.add_argument("--obs_npz", type=str, default=None, help="Optional NPZ with observation tensors keyed by observation name")
    ap.add_argument("--goal_npz", type=str, default=None, help="Optional NPZ with goal tensors keyed by task image name")
    ap.add_argument("--dataset", type=str, default=None, help="Dataset key for action unnormalization, e.g. bridge_dataset")
    ap.add_argument("--seed", type=int, default=0, help="Seed for shared-noise action comparison")
    ap.add_argument("--save_dir", type=str, default="artifacts/jax_torch_compare", help="Directory to save comparison outputs")
    ap.add_argument("--device", type=str, default=None, help="Torch device, default: cuda if available else cpu")
    args = ap.parse_args()

    if args.torch_ckpt is None and not args.pt_from_jax:
        raise SystemExit("Provide either --torch_ckpt or --pt_from_jax.")
    if args.text is None and args.goal_primary is None and args.goal_npz is None:
        raise SystemExit("Provide at least one task input: --text, --goal_primary, or --goal_npz.")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Loading JAX model from: {args.jax_ckpt}")
    jax_model = OctoModel.load_pretrained(args.jax_ckpt)

    if args.torch_ckpt is not None:
        print(f"Loading Torch model from: {args.torch_ckpt}")
        pt_model = OctoModelPt.load_pretrained(Path(args.torch_ckpt))["octo_model"].to(device)
    else:
        print(f"Loading Torch weights directly from JAX checkpoint: {args.jax_ckpt}")
        pt_model = OctoModelPt.load_pretrained_from_jax(args.jax_ckpt)["octo_model"].to(device)
    pt_model.eval()

    obs_override = {}
    if args.obs_npz is not None:
        with np.load(args.obs_npz, allow_pickle=False) as data:
            obs_override = {str(k): np.asarray(v) for k, v in data.items()}

    goal_override = {}
    if args.goal_npz is not None:
        with np.load(args.goal_npz, allow_pickle=False) as data:
            goal_override = {str(k): np.asarray(v) for k, v in data.items()}

    image_sequences = {
        "image_primary": parse_csv_paths(args.obs_primary),
        "image_wrist": parse_csv_paths(args.obs_wrist),
    }
    image_sequences = {k: v for k, v in image_sequences.items() if v is not None}

    goal_images = {
        "image_primary": args.goal_primary,
        "image_wrist": args.goal_wrist,
    }
    goal_images = {k: v for k, v in goal_images.items() if v is not None}

    jax_observation = build_jax_observation(
        example_observation=jax_model.example_batch["observation"],
        window=args.window,
        observation_overrides=obs_override,
        image_sequences=image_sequences,
    )

    goal_inputs = build_goal_overrides(
        example_task=jax_model.example_batch["task"],
        goal_overrides=goal_override,
        goal_images=goal_images,
    )
    jax_goals = {k: v[None] for k, v in goal_inputs.items()} if goal_inputs else None
    texts = [args.text] if args.text is not None else None

    jax_task = jax_model.create_tasks(goals=jax_goals, texts=texts)
    pt_observation = jax_obs_to_torch(jax_observation, device=device)
    pt_task = ensure_torch_task_on_device(pt_model.create_tasks(goals=jax_goals, texts=texts, device=device), device=device)

    print("Running deterministic transformer forward pass for JAX ...")
    jax_transformer_outputs = jax_model.run_transformer(
        jax_observation,
        jax_task,
        jax_observation["timestep_pad_mask"],
        train=False,
    )

    print("Running deterministic transformer forward pass for Torch ...")
    pt_transformer_outputs, _ = pt_model(
        pt_observation,
        pt_task,
        pt_observation["timestep_pad_mask"],
        train=False,
        transformer_only=True,
        save_attention_mask=False,
    )

    transformer_summary = _summarize_transformer_outputs(jax_transformer_outputs, pt_transformer_outputs)

    print("\nTransformer token comparison")
    for key, payload in transformer_summary["common"].items():
        print(format_metric_row(f"{key}.tokens", payload["tokens"]))
        print(format_metric_row(f"{key}.mask", payload["mask"]))
    if transformer_summary["jax_only"]:
        print("JAX-only groups:", ", ".join(transformer_summary["jax_only"]))
    if transformer_summary["torch_only"]:
        print("Torch-only groups:", ", ".join(transformer_summary["torch_only"]))

    jax_action_stats, pt_action_stats = _get_dataset_stats(jax_model, pt_model, args.dataset)
    action_summary = None
    native_action_summary = None

    try:
        pt_head = pt_model.module.heads["action"]
        action_dim = int(pt_head.action_dim)
        action_horizon = int(pt_head.action_horizon)
        diffusion_steps = int(pt_head.diffusion_steps)
        max_action = float(pt_head.max_action)
        embodiment_action_dim = None if pt_action_stats is None else int(len(pt_action_stats["mean"]))

        print("\nRunning shared-noise diffusion action comparison ...")
        readout_key = str(pt_head.readout_key)
        jax_action = run_shared_noise_diffusion_jax(
            model=jax_model,
            transformer_outputs=jax_transformer_outputs,
            diffusion_steps=diffusion_steps,
            action_horizon=action_horizon,
            action_dim=action_dim,
            max_action=max_action,
            seed=args.seed,
            embodiment_action_dim=embodiment_action_dim,
            readout_key=readout_key,
        )
        pt_action = run_shared_noise_diffusion_torch(
            action_head=pt_head,
            transformer_outputs=pt_transformer_outputs,
            seed=args.seed,
            embodiment_action_dim=embodiment_action_dim,
        )

        jax_action = unnormalize_action_numpy(jax_action, jax_action_stats)
        pt_stats_np = None
        if pt_action_stats is not None:
            pt_stats_np = {
                k: (v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v))
                for k, v in pt_action_stats.items()
            }
        pt_action = unnormalize_action_numpy(pt_action, pt_stats_np)

        action_summary = compare_arrays(jax_action, pt_action)
        print(format_metric_row("shared_noise_action", action_summary))
    except Exception as exc:
        print(f"Shared-noise action comparison skipped: {exc}")

    if args.dataset is not None:
        print("\nRunning native sample_actions comparison ...")
        try:
            import jax

            jax_native_action = np.asarray(
                jax_model.sample_actions(
                    jax_observation,
                    jax_task,
                    unnormalization_statistics=jax_action_stats,
                    rng=jax.random.PRNGKey(args.seed),
                )
            )
            generator = torch.Generator(device=device)
            generator.manual_seed(args.seed)
            pt_native_action = (
                pt_model.sample_actions(
                    pt_observation,
                    pt_task,
                    unnormalization_statistics=pt_action_stats,
                    generator=generator,
                    save_attention_mask=False,
                )
                .detach()
                .cpu()
                .numpy()
            )
            native_action_summary = compare_arrays(jax_native_action, pt_native_action)
            print(format_metric_row("native_sample_action", native_action_summary))
        except Exception as exc:
            print(f"Native sampled action comparison skipped: {exc}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save per-group token dumps for deeper offline inspection.
    for key in sorted(set(jax_transformer_outputs.keys()) & set(pt_transformer_outputs.keys())):
        jax_tokens, jax_mask = token_group_to_numpy(jax_transformer_outputs[key])
        pt_tokens, pt_mask = token_group_to_numpy(pt_transformer_outputs[key])
        np.savez_compressed(
            save_dir / f"{key}.npz",
            jax_tokens=jax_tokens,
            pt_tokens=pt_tokens,
            jax_mask=jax_mask,
            pt_mask=pt_mask,
        )

    if action_summary is not None:
        np.savez_compressed(save_dir / "shared_noise_action.npz", jax_action=jax_action, pt_action=pt_action)
    if native_action_summary is not None:
        np.savez_compressed(save_dir / "native_sample_action.npz", jax_action=jax_native_action, pt_action=pt_native_action)

    payload = {
        "args": vars(args),
        "transformer_outputs": transformer_summary,
        "shared_noise_action": action_summary,
        "native_sample_action": native_action_summary,
    }
    save_json(save_dir / "comparison_summary.json", payload)
    print(f"\nSaved comparison artifacts to: {save_dir}")
    print(f"Summary JSON: {save_dir / 'comparison_summary.json'}")


if __name__ == "__main__":
    main()

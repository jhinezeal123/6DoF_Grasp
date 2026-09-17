"""Chia e2e sample_actions de tim phan thoi gian "khong giai thich duoc".

`runner_inference_sample.py --profile` do tung module RIENG LE, nhung tong cac
module do (511 ms) thap hon e2e (850 ms) tan ~340 ms. Khoang tan do khong nam o
module nao ca, nen phai do dung cai ma e2e lam.

sample_actions lam dung 3 viec:
  A. module.forward : task tokenizers -> projections -> pos-emb -> backbone -> readouts
  B. action_head    : diffusion sampling
  C. postprocess    : dua action ve CPU + unnormalize + kiem tra huu han

Script nay do A, B, C rieng roi doi chieu voi e2e. Neu A >> tong cac manh nho
ben trong no (tokenizers + backbone + readouts) thi 340 ms nam o phan "keo keo"
(projections, positional embedding, dung mask, ghep token) — dung thu ma
profiler cu khong do.

Chay:
  python scripts/profile_split_e2e.py --ckpt <dir> --obs_primary assets/obs_primary.png \
      --text "pick up the red cube" --window 2 --dataset bridge_dataset
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))  # de import duoc goi `octo`
sys.path.insert(0, str(_HERE))         # de import duoc runner_inference_sample

from octo.model.octo_model_pt import OctoModelPt  # noqa: E402


def _load_runner():
    """Nap runner_inference_sample.py nhu mot module (no co `if __name__` guard)."""
    spec = importlib.util.spec_from_file_location(
        "runner_inference_sample", _HERE / "runner_inference_sample.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _median_ms(fn, device, iters):
    """Trung vi thoi gian (ms) cua fn, do bang dong ho host + dong bo CUDA."""
    samples = []
    out = None
    for _ in range(iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    return samples[len(samples) // 2], out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Torch checkpoint directory")
    ap.add_argument("--window", type=int, default=2)
    ap.add_argument("--obs_primary", type=str, default=None)
    ap.add_argument("--obs_wrist", type=str, default=None)
    ap.add_argument("--text", type=str, default=None)
    ap.add_argument("--dataset", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    args = ap.parse_args()

    runner = _load_runner()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA device:", torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu")

    model = OctoModelPt.load_pretrained(Path(args.ckpt))["octo_model"].to(device)
    model.eval()

    observation = runner.build_observation_from_example(
        model=model,
        window=args.window,
        obs_primary_paths=runner._parse_paths(args.obs_primary),
        obs_wrist_paths=runner._parse_paths(args.obs_wrist),
        device=device,
    )
    task = model.create_tasks(
        goals=None, texts=[args.text] if args.text is not None else None, device=device
    )

    stats = None
    if args.dataset is not None:
        stats = model.dataset_statistics[args.dataset]["action"]

    module = model.module
    head = module.heads["action"]
    tpm = observation["timestep_pad_mask"]
    embodiment_dim = len(stats["mean"]) if stats is not None else None

    def _e2e():
        gen = torch.Generator(device=device).manual_seed(args.seed)
        return model.sample_actions(
            observation, task, unnormalization_statistics=stats, generator=gen
        )

    def _module_forward():
        return model(
            observation, task, tpm, train=False,
            transformer_only=True, save_attention_mask=True,
        )

    # Warmup: nap CUDA cache + cache attention mask (nhu --profile van lam).
    with torch.inference_mode():
        for _ in range(max(0, args.warmup)):
            _e2e()
    if device.type == "cuda":
        torch.cuda.synchronize()

    with torch.inference_mode():
        e2e_ms, _ = _median_ms(_e2e, device, args.iters)
        fwd_ms, fwd_out = _median_ms(_module_forward, device, args.iters)

        transformer_outputs = fwd_out[0]

        def _head():
            gen = torch.Generator(device=device).manual_seed(args.seed)
            return head.predict_action(
                transformer_outputs,
                train=False,
                embodiment_action_dim=embodiment_dim,
                generator=gen,
            )

        head_ms, _ = _median_ms(_head, device, args.iters)

    residual = e2e_ms - fwd_ms - head_ms

    print("\n== Chia e2e sample_actions ==")
    print(f"{'e2e sample_actions':32s} {e2e_ms:8.2f} ms")
    print(f"{'  A. module.forward':32s} {fwd_ms:8.2f} ms   ({fwd_ms / e2e_ms * 100:.1f}%)")
    print(f"{'  B. action_head':32s} {head_ms:8.2f} ms   ({head_ms / e2e_ms * 100:.1f}%)")
    print(f"{'  C. postprocess (con lai)':32s} {residual:8.2f} ms   ({residual / e2e_ms * 100:.1f}%)")
    print(
        "\nSo sanh voi --profile: A duoc chia nho tiep thanh "
        "task/obs tokenizers + transformer_backbone + readout."
    )
    print("Neu A lon hon nhieu so voi tong cac manh do => thoi gian nam o phan keo keo")
    print("(projections, positional embedding, dung attention mask, ghep token).")


if __name__ == "__main__":
    main()

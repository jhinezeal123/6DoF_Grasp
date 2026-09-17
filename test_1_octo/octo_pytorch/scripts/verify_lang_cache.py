"""Chung minh cache embedding cau lenh (task tokenizer) vua DUNG vua NHANH.

Cau lenh khong doi trong suot episode, nhung t5-base van chay lai moi buoc.
Cache theo noi dung trong OctoTransformerPt._run_task_tokenizer.

Script nay kiem 3 dieu:
  1. Cache co that su hit khong (dem hit/miss).
  2. Nhanh hon bao nhieu (do median co cache vs tat cache).
  3. Co lam SAI ket qua khong (so action tung phan tu, doi hoi giong het).

Chay:
  python scripts/verify_lang_cache.py --ckpt <dir> [--iters 4]
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))

from octo.model.octo_model_pt import OctoModelPt  # noqa: E402
from octo.model.octo_module_pt import (  # noqa: E402
    OctoTransformerPt,
    _content_digest,
)


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "runner_inference_sample", _HERE / "runner_inference_sample.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--text", default="pick up the red cube")
    ap.add_argument("--dataset", default="bridge_dataset")
    ap.add_argument("--window", type=int, default=2)
    ap.add_argument("--iters", type=int, default=4)
    args = ap.parse_args()

    runner = _load_runner()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = OctoModelPt.load_pretrained(Path(args.ckpt))["octo_model"].to(device)
    model.eval()
    observation = runner.build_observation_from_example(
        model=model,
        window=args.window,
        obs_primary_paths=runner._parse_paths("assets/obs_primary.png"),
        obs_wrist_paths=runner._parse_paths(None),
        device=device,
    )
    task = model.create_tasks(goals=None, texts=[args.text], device=device)
    stats = model.dataset_statistics[args.dataset]["action"]

    def run_once():
        gen = torch.Generator(device=device).manual_seed(0)
        with torch.inference_mode():
            return model.sample_actions(
                observation, task, unnormalization_statistics=stats, generator=gen
            )

    def median_ms(n):
        ts = []
        for _ in range(n):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            run_once()
            if device.type == "cuda":
                torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000.0)
        ts.sort()
        return ts[len(ts) // 2]

    # --- 1+2: co cache, dem hit/miss ---
    counts = {"hit": 0, "miss": 0}
    original = OctoTransformerPt._run_task_tokenizer

    def counting(self, name, tokenizer, observations, tasks, train):
        if not train:
            key = (name, _content_digest(tasks))
            counts["hit" if key in self._task_tokenizer_cache else "miss"] += 1
        return original(self, name, tokenizer, observations, tasks, train)

    OctoTransformerPt._run_task_tokenizer = counting
    action_cached = run_once()          # lan dau: miss
    counts_after_first = dict(counts)
    ms_cached = median_ms(args.iters)
    hits_total = dict(counts)
    OctoTransformerPt._run_task_tokenizer = original

    # --- 3: tat cache, luon tinh lai ---
    OctoTransformerPt._run_task_tokenizer = (
        lambda self, name, tokenizer, observations, tasks, train:
        tokenizer(observations, tasks, train=train)
    )
    action_nocache = run_once()
    ms_nocache = median_ms(args.iters)

    print("\n== Ket qua kiem tra cache cau lenh ==")
    print(f"  lan chay dau tien      : {counts_after_first}  (mong doi miss=1)")
    print(f"  sau {1 + args.iters} lan chay    : {hits_total}")
    print(f"  co cache   (median)    : {ms_cached:8.2f} ms")
    print(f"  khong cache(median)    : {ms_nocache:8.2f} ms")
    print(f"  tiet kiem moi buoc     : {ms_nocache - ms_cached:8.2f} ms")
    same = torch.equal(action_cached, action_nocache)
    print(f"  action GIONG HET nhau  : {same}")
    print(f"  sai khac tuyet doi max : {(action_cached - action_nocache).abs().max().item():.3e}")

    ok = counts_after_first.get("miss") == 1 and hits_total.get("hit", 0) >= args.iters and same
    print(f"\n  => {'DAT' if ok else 'KHONG DAT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

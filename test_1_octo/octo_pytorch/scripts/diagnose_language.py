"""Chan doan: cau lenh (language) co that su anh huong toi action khong?

Cach lam: nap model MOT lan, tao hai task voi hai cau lenh khac han nhau,
reset generator ve cung seed truoc moi lan goi, roi so action.

- Neu token ngon ngu KHAC nhau nhung action GIONG nhau => tang ngon ngu bi vo
  hieu o dau do phia sau (conditioning chet).
- Neu action khac nhau => ngon ngu co tac dung, van de nam cho khac.

Chay:
    python scripts/diagnose_language.py --ckpt <dir> --obs_primary <anh.jpg>
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TRANSFORMERS_NO_JAX", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from octo.robot.policy import OctoTorchPolicy  # noqa: E402

TEXT_A = "pick up the blue block"
TEXT_B = "fly the rocket to mars immediately"


def _load_uint8(path: str, hw) -> np.ndarray:
    import cv2

    img = cv2.imread(path)
    if img is None:
        raise SystemExit("khong doc duoc anh: %s" % path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = hw
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return img.astype(np.uint8)


def _token_summary(task: dict) -> str:
    """In shape + tong tri tuyet doi cua token ngon ngu trong task dict."""
    lines = []
    for k, v in task.items():
        if isinstance(v, torch.Tensor):
            lines.append("    %-42s %-22s sum=%.6f" % (k, tuple(v.shape), float(v.float().abs().sum())))
        elif isinstance(v, dict):
            for k2, v2 in v.items():
                if isinstance(v2, torch.Tensor):
                    lines.append(
                        "    %-42s %-22s sum=%.6f"
                        % (k + "/" + k2, tuple(v2.shape), float(v2.float().abs().sum()))
                    )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", default="myarm_stage_a_dataset")
    ap.add_argument("--obs_primary", required=True)
    ap.add_argument("--obs_wrist", default=None)
    ap.add_argument("--window", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    policy = OctoTorchPolicy.from_checkpoint(
        checkpoint_dir=args.ckpt,
        device="cuda",
        dataset_key=args.dataset,
        fp16=False,
        seed=args.seed,
    )
    sizes = policy.expected_image_sizes()
    print("Model mong doi modality:", sizes)

    window = args.window
    obs = {}
    if "primary" in sizes:
        img = _load_uint8(args.obs_primary, sizes["primary"])
        obs["image_primary"] = np.stack([img] * window, axis=0)
    if "wrist" in sizes and args.obs_wrist:
        img = _load_uint8(args.obs_wrist, sizes["wrist"])
        obs["image_wrist"] = np.stack([img] * window, axis=0)
    obs["timestep"] = np.arange(window, dtype=np.int32)
    obs["timestep_pad_mask"] = np.ones((window,), dtype=np.bool_)

    task_a = policy.create_task(text=TEXT_A)
    task_b = policy.create_task(text=TEXT_B)

    print("\n=== TASK A: %r ===" % TEXT_A)
    print(_token_summary(task_a))
    print("=== TASK B: %r ===" % TEXT_B)
    print(_token_summary(task_b))

    key = "task/language_instruction/input_ids"
    ta = task_a.get("language_instruction", {}).get("input_ids")
    tb = task_b.get("language_instruction", {}).get("input_ids")
    if ta is None:
        ta = task_a.get("input_ids")
        tb = task_b.get("input_ids")
    if ta is not None and tb is not None:
        same_ids = bool(torch.equal(ta, tb))
        print("\ninput_ids A == B ? %s" % same_ids)
        print("  A:", ta[0].tolist()[:12] if ta.ndim > 1 else ta.tolist()[:12])
        print("  B:", tb[0].tolist()[:12] if tb.ndim > 1 else tb.tolist()[:12])
    print("(key tham chieu cho chan doan: %s)" % key)

    a_a = None
    a_b = None
    with torch.inference_mode():
        policy.generator.manual_seed(args.seed)
        a_a = policy.sample_action_chunk(obs, task_a)
        policy.generator.manual_seed(args.seed)
        a_b = policy.sample_action_chunk(obs, task_b)

    print("\n=== ACTION[0] ===")
    print("A:", np.array2string(a_a[0], precision=6, suppress_small=False))
    print("B:", np.array2string(a_b[0], precision=6, suppress_small=False))

    diff = float(np.max(np.abs(a_a - a_b)))
    scale = float(np.max(np.abs(a_a)))
    print("\nmax |A - B| = %.3e   (bien do action = %.3e)" % (diff, scale))
    print("ti le      = %.3e" % (diff / max(scale, 1e-12)))
    if diff / max(scale, 1e-12) < 1e-3:
        print("\nKET LUAN: NGON NGU VO HIEU - doi cau lenh khong doi action.")
    else:
        print("\nKET LUAN: ngon ngu CO tac dung.")


if __name__ == "__main__":
    main()

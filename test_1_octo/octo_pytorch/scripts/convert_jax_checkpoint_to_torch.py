"""Convert a JAX/Flax Octo checkpoint into a torch checkpoint for Jetson.

Run this script on a PC where you have JAX/Flax/TensorFlow installed.

Example:
  python scripts/convert_jax_checkpoint_to_torch.py \
    --src hf://rail-berkeley/octo-small-1.5 \
    --dst ./checkpoints/octo-small-1.5-torch \
    --save-step 0

Then on Jetson (torch-only install):
  from octo.model.octo_model_pt import OctoModelPt
  model = OctoModelPt.load_pretrained('./checkpoints/octo-small-1.5-torch')['octo_model']

Notes:
  * This conversion copies all JAX params except the language tokenizer weights.
    The language encoder (T5) is loaded from HuggingFace by LanguageTokenizerPt.
"""

import argparse
from pathlib import Path

from octo.model.octo_model_pt import OctoModelPt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        required=True,
        help="Source JAX checkpoint path or HF ref (e.g. hf://rail-berkeley/octo-small-1.5)",
    )
    parser.add_argument(
        "--dst",
        required=True,
        help="Destination directory for the torch checkpoint (created if missing)",
    )
    parser.add_argument(
        "--jax-step",
        type=int,
        default=None,
        help="Optional JAX step to load (defaults to latest)",
    )
    parser.add_argument(
        "--save-step",
        type=int,
        default=0,
        help="Step index to use inside the torch checkpoint folder (default: 0)",
    )
    args = parser.parse_args()

    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    load = OctoModelPt.load_pretrained_from_jax(args.src, step=args.jax_step)
    model = load["octo_model"]

    model.save_pretrained(step=args.save_step, checkpoint_path=str(dst))
    print(f"Saved torch checkpoint to: {dst} (step={args.save_step})")


if __name__ == "__main__":
    main()

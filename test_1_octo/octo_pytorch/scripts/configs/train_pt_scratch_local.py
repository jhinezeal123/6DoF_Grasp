"""Local PyTorch pretraining config (from scratch).

Usage:
  accelerate launch scripts/train_pt.py \
    --config scripts/configs/train_pt_scratch_local.py \
    --name scratch

Edit `save_dir`, `dataset_kwargs.oxe_kwargs.data_dir`, and optionally `data_mix`.

If you want to train on a custom TFDS/RLDS dataset without modifying this file,
prefer passing:
  --dataset_json=/path/to/single_dataset.json
or
  --dataset_mix_json=/path/to/mixture.json
and (if needed) --tfds_import=your.custom.tfds_builder_module
"""

from copy import deepcopy
import importlib.util
import os

from ml_collections import ConfigDict


def _load_base_get_config():
  base_path = os.path.join(os.path.dirname(__file__), "octo_pretrain_config.py")
  spec = importlib.util.spec_from_file_location("octo_pretrain_config", base_path)
  if spec is None or spec.loader is None:
    raise ImportError(f"Could not load config module from {base_path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module.get_config


_get_base = _load_base_get_config()


def get_config(config_string=None):
    cfg = deepcopy(_get_base(config_string))

    # Local output directory (no automatic nesting unless append_wandb_id_to_save_dir=True)
    cfg.save_dir = "/home/ceec/khoanhd/VLAR_Lab/octo_weights/runs/octo-base-1.5-pt-torch-scratch-libero-10"
    cfg.append_wandb_id_to_save_dir = True
    cfg.overwrite_output = False

    # Train from scratch
    cfg.pretrained_path = None
    cfg.pretrained_format = "auto"
    cfg.pretrained_step = None
    cfg.load_optimizer_state = False
    cfg.start_step = 0

    # Checkpointing behavior
    cfg.save_initial_checkpoint = True
    cfg.save_final_checkpoint = True
    cfg.save_optimizer_state = False

    return ConfigDict(cfg)

"""Local PyTorch pretraining config (resume training from a torch checkpoint).

Usage:
  accelerate launch scripts/train_pt.py \
    --config scripts/configs/train_pt_resume_local.py \
    --name resume

Set `pretrained_path` to an existing torch checkpoint folder (format like octo-base-1.5-torch).
If `start_step` is None, train_pt.py will default start_step to the loaded checkpoint step.
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

    # Local output directory for the *new* run.
    cfg.save_dir = "/home/ceec/khoanhd/VLAR_Lab/octo_weights/runs/octo_pt_resume"
    cfg.append_wandb_id_to_save_dir = True
    cfg.overwrite_output = False

    # Resume from torch checkpoint
    cfg.pretrained_path = "/home/ceec/khoanhd/VLAR_Lab/octo_weights/pretrained_pytorch/octo-base-1.5-torch"
    cfg.pretrained_format = "torch"
    cfg.pretrained_step = None  # None => load latest numeric step in pretrained_path
    cfg.load_optimizer_state = True

    # If None => defaults to loaded step; if you want to restart schedule at 0, set 0.
    cfg.start_step = None

    # Checkpointing behavior
    cfg.save_initial_checkpoint = False
    cfg.save_final_checkpoint = True
    cfg.save_optimizer_state = True

    return ConfigDict(cfg)

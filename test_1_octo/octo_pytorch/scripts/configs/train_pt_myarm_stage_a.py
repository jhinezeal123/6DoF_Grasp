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
    cfg.num_steps = 20000
    cfg.log_interval = 20
    cfg.eval_interval = 1000
    cfg.save_interval = 1000
    cfg.grad_accum_steps = 1
    cfg.save_dir = "/tmp/octo_runs/train_pt_myarm_stage_a"
    cfg.append_wandb_id_to_save_dir = True
    cfg.overwrite_output = False
    cfg.dataset_kwargs.batch_size = 64
    cfg.dataset_kwargs.dataloader_num_workers = 4
    cfg.dataset_kwargs.dataloader_prefetch_factor = 4
    cfg.dataset_kwargs.dataloader_persistent_workers = True
    cfg.dataset_kwargs.rlds_iterator_prefetch = 8
    cfg.dataset_kwargs.tf_intra_op_threads = 8
    cfg.dataset_kwargs.tf_inter_op_threads = 8
    cfg.dataset_kwargs.copy_non_writable_arrays = False
    cfg.pretrained_path = None
    cfg.pretrained_format = "auto"
    cfg.pretrained_step = None
    cfg.load_optimizer_state = False
    cfg.start_step = 0
    return ConfigDict(cfg)

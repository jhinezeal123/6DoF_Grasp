"""Single-GPU optimized config for LIBERO spatial finetuning (Torch -> LIBERO).

Designed for:
- 1x RTX 4090 (24GB)
- camera: primary only
- window_size=2, action_horizon=4
- no gradient accumulation (grad_accum_steps=1)

Use with:
  python3 scripts/finetune_pt.py \
    --config scripts/configs/finetune_pt_single_gpu_libero_spatial.py \
    --dataset_json scripts/datasets/libero_spatial_no_noops_from_directory.json

Set `cfg.pretrained_path` to your converted PyTorch checkpoint directory.
"""

from copy import deepcopy
import importlib.util
import os

from ml_collections import ConfigDict


def _load_base_get_config():
    base_path = os.path.join(
        os.path.dirname(__file__), "octo_small_1_5_scratch_libero_spatial_config.py"
    )
    spec = importlib.util.spec_from_file_location(
        "octo_small_1_5_scratch_libero_spatial_config", base_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load config module from {base_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_config


_get_base = _load_base_get_config()


def get_config(config_string=None):
    cfg = deepcopy(_get_base(config_string))

    # Output
    cfg.save_dir = "/home/ceec/khoanhd/VLAR_Lab/octo_weights/runs/octo-small-1.5-torch-finetune-libero_spatial"
    cfg.append_wandb_id_to_save_dir = True
    cfg.overwrite_output = False

    # Finetune from torch checkpoint
    cfg.pretrained_path = "/absolute/path/to/pretrained_pt_checkpoint_dir"
    cfg.pretrained_format = "torch"
    cfg.pretrained_step = None
    cfg.load_optimizer_state = False

    # Finetune typically restarts the schedule at step 0.
    cfg.start_step = 0

    # Train schedule
    cfg.num_steps = 10000
    cfg.log_interval = 50
    cfg.eval_interval = 500
    cfg.save_interval = 2000

    # No gradient accumulation
    cfg.grad_accum_steps = 1

    # Checkpoints
    cfg.save_initial_checkpoint = True
    cfg.save_final_checkpoint = True
    cfg.save_optimizer_state = True

    # Dataset perf (single GPU)
    cfg.dataset_kwargs.per_device_batch_size = 32
    cfg.dataset_kwargs.batch_size = 32  # fallback
    cfg.dataset_kwargs.shuffle_buffer_size = 2000
    cfg.dataset_kwargs.traj_transform_threads = 16
    cfg.dataset_kwargs.traj_read_threads = 16
    cfg.dataset_kwargs.copy_non_writable_arrays = False
    cfg.dataset_kwargs.tf_intra_op_threads = 16
    cfg.dataset_kwargs.tf_inter_op_threads = 16
    cfg.dataset_kwargs.dataloader_num_workers = 4
    cfg.dataset_kwargs.dataloader_prefetch_factor = 4
    cfg.dataset_kwargs.dataloader_persistent_workers = True
    cfg.dataset_kwargs.dataloader_multiprocessing_context = "spawn"
    cfg.dataset_kwargs.rlds_iterator_prefetch = 4

    if "frame_transform_kwargs" in cfg.dataset_kwargs:
        cfg.dataset_kwargs.frame_transform_kwargs.num_parallel_calls = 16

    cfg.dataset_kwargs.balance_weights = False

    # Profiling to diagnose bottlenecks (data_wait vs compute)
    cfg.profile_timing = True
    cfg.profile_timing_window = 20

    # Optimizer
    cfg.optimizer.learning_rate.peak_value = 1e-4
    cfg.optimizer.learning_rate.warmup_steps = 1000
    cfg.optimizer.weight_decay = 0.05
    cfg.optimizer.clip_gradient = 1.0

    return ConfigDict(cfg)

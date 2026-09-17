"""Single-GPU optimized config for LIBERO spatial (RLDS/TFDS) scratch training.

Designed for:
- 1x RTX 4090 (24GB)
- camera: primary only
- window_size=2, action_horizon=4
- no gradient accumulation (grad_accum_steps=1)
- effective batch == per-device batch (single GPU)

Use with:
  python3 scripts/train_pt.py \
    --config scripts/configs/train_pt_single_gpu_libero_spatial.py \
    --scratch \
    --dataset_json scripts/datasets/libero_spatial_no_noops_from_directory.json

Note:
- Effective batch (single GPU) = per_device_batch_size.
"""

from copy import deepcopy
import importlib.util
import os

from ml_collections import ConfigDict


def _load_base_get_config():
    base_path = os.path.join(os.path.dirname(__file__), "octo_small_1_5_scratch_libero_spatial_config.py")
    spec = importlib.util.spec_from_file_location("octo_small_1_5_scratch_libero_spatial_config", base_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load config module from {base_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_config


_get_base = _load_base_get_config()


def get_config(config_string=None):
    cfg = deepcopy(_get_base(config_string))

    # Output
    cfg.save_dir = "/home/ceec/khoanhd/VLAR_Lab/octo_weights/runs/octo-small-1.5-torch-scratch-libero_spatial"
    cfg.append_wandb_id_to_save_dir = True
    cfg.overwrite_output = False

    # Scratch
    cfg.pretrained_path = None
    cfg.pretrained_format = "auto"
    cfg.pretrained_step = None
    cfg.load_optimizer_state = True
    cfg.start_step = 0

    # Checkpoints
    cfg.save_initial_checkpoint = True
    cfg.save_final_checkpoint = True
    cfg.save_optimizer_state = True

    # Train schedule
    cfg.num_steps = 50000
    cfg.log_interval = 50
    cfg.eval_interval = 500
    cfg.save_interval = 10000

    # No gradient accumulation (optimizer steps are cfg.num_steps)
    cfg.grad_accum_steps = 1

    # Dataset perf (single GPU)
    # We will pass dataset via --dataset_json; these values control batching/threads/shuffle.
    # Try to push VRAM closer to ~16GB on 4090.
    # If you still see low VRAM/GPU (<20%), try per_device_batch_size=48 or 64.
    cfg.dataset_kwargs.per_device_batch_size = 32
    cfg.dataset_kwargs.batch_size = 32  # fallback
    cfg.dataset_kwargs.shuffle_buffer_size = 4096
    cfg.dataset_kwargs.traj_transform_threads = 16
    cfg.dataset_kwargs.traj_read_threads = 16

    # If you see warnings about non-writable numpy arrays from torch.as_tensor,
    # you can set this True to copy those arrays (slower but silences warning).
    cfg.dataset_kwargs.copy_non_writable_arrays = False

    # Optional: TF thread pools (0/None means leave TF defaults).
    cfg.dataset_kwargs.tf_intra_op_threads = 16
    cfg.dataset_kwargs.tf_inter_op_threads = 16

    # Optional: PyTorch DataLoader multiprocessing.
    # If you see hangs/crashes, set dataloader_num_workers=0.
    cfg.dataset_kwargs.dataloader_num_workers = 4
    # Keep this small; large values can explode RAM without improving throughput.
    cfg.dataset_kwargs.dataloader_prefetch_factor = 4
    cfg.dataset_kwargs.dataloader_persistent_workers = True
    cfg.dataset_kwargs.dataloader_multiprocessing_context = "spawn"

    # TF/dlimp iterator prefetch (enables TF-side prefetch threads).
    cfg.dataset_kwargs.rlds_iterator_prefetch = 4

    # Profiling to diagnose bottlenecks (data_wait vs compute)
    cfg.profile_timing = True
    cfg.profile_timing_window = 20

    # TF frame transforms parallelism (kept modest to avoid CPU thrash).
    if "frame_transform_kwargs" in cfg.dataset_kwargs:
        cfg.dataset_kwargs.frame_transform_kwargs.num_parallel_calls = 16

    # With a single dataset, balancing doesn't change anything.
    cfg.dataset_kwargs.balance_weights = False

    # Optimizer
    cfg.optimizer.learning_rate.peak_value = 3e-4
    cfg.optimizer.learning_rate.warmup_steps = 2000
    cfg.optimizer.weight_decay = 0.1
    cfg.optimizer.clip_gradient = 1.0

    return ConfigDict(cfg)

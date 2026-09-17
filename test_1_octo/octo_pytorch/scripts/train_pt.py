"""PyTorch training script for Octo on TFDS/RLDS datasets.

This mirrors the high-level data pipeline of `scripts/train.py` (JAX):
- Builds an interleaved RLDS dataset mixture via `octo.data.dataset.make_interleaved_dataset`
- Wraps it for PyTorch with `octo.utils.torch_rlds_dataset.TorchRLDSDataset`
- Trains an `OctoModelPt` checkpoint (or initializes from config)

Notes
-----
- RLDS/TFDS loading uses TensorFlow; we disable TF GPU visibility so TF does not
  reserve GPU memory needed by PyTorch.
- For multi-GPU, we rely on `accelerate.PartialState` + DDP.
- Dataset sharding across processes is best-effort: we set distinct TF seeds per
  process so shuffle/sample ops diverge.
"""

from __future__ import annotations

# WARNING: importing tensorflow too late can silence important logging
import os

os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TRANSFORMERS_NO_JAX", "1")

import datetime
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import importlib

import numpy as np
import tensorflow as tf
import torch
import tqdm
try:
	import wandb  # type: ignore
except Exception:  # pragma: no cover
	wandb = None
from absl import app, flags, logging
from accelerate import PartialState
from ml_collections import config_flags, ConfigDict
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

# Ensure `octo` is importable even when this script is launched outside the repo root.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(_PROJECT_ROOT))

from octo.data.dataset import make_interleaved_dataset
from octo.data.oxe import make_oxe_dataset_kwargs_and_weights
from octo.model.octo_model_pt import OctoModelPt
from octo.utils.spec import ModuleSpec
from octo.utils.torch_rlds_dataset import TorchRLDSDataset
from octo.utils.train_utils_common import filter_eval_datasets, format_name_with_config
from octo.utils.train_utils_pt import (
	_jax_config_to_pt_config,
	_np2pt,
	_to_device,
	freeze_weights_pt,
)
from octo.utils.rlds_dataset_factory import RLDSDatasetFactory, to_builtins


FLAGS = flags.FLAGS

flags.DEFINE_string("name", "experiment", "Experiment name.")
flags.DEFINE_bool("debug", False, "Debug config (no wandb logging)")

flags.DEFINE_bool(
	"use_wandb",
	False,
	"If True, log metrics to Weights & Biases (requires wandb). "
	"If False (default), logs to console and optionally to metrics.jsonl in save_dir.",
)
flags.DEFINE_string(
	"metrics_jsonl",
	None,
	"Optional path to a JSONL file for metrics logging. If not set and save_dir is set, "
	"writes to <save_dir>/metrics.jsonl.",
)

# Dataset override options (for TFDS/RLDS).
# These are designed for convenience when you want to train from scratch on a custom dataset
# without editing Python config files.
flags.DEFINE_string(
	"dataset_json",
	None,
	"Path to JSON file describing a SINGLE dataset kwargs (for make_dataset_from_rlds). "
	"Will be wrapped as dataset_kwargs_list=[...], sample_weights=[1.0]. "
	"Tip: use 'builder_dir' to load a TFDS dataset from an exported directory (tfds.builder_from_directory), "
	"and 'dataset_statistics_key' if your dataset_statistics JSON contains multiple datasets.",
)
flags.DEFINE_string(
	"dataset_mix_json",
	None,
	"Path to JSON file describing a dataset mixture. Expected keys: dataset_kwargs_list (list) and "
	"optionally sample_weights (list). You may also include batch_size/shuffle_buffer_size/etc to override.",
)
flags.DEFINE_multi_string(
	"tfds_import",
	[],
	"Extra python modules to import before loading TFDS builders (useful for custom datasets). "
	"Example: --tfds_import=octo.data.stage_a_tfds",
)
flags.DEFINE_bool(
	"scratch",
	False,
	"Force train-from-scratch: ignores pretrained_path, sets start_step=0. (Config can still set model/dataset.)",
)
flags.DEFINE_bool(
	"allow_cpu",
	False,
	"Allow running on CPU when CUDA is not available (useful for dataset/debug runs; training will be slow).",
)

flags.DEFINE_string(
	"resume_from",
	None,
	"Resume training in-place from an existing PyTorch checkpoint directory (the folder containing config.json). "
	"This overrides config.save_dir and config.pretrained_*.",
)
flags.DEFINE_integer(
	"resume_step",
	-1,
	"Checkpoint step directory to resume from. Use -1 to pick the latest available step.",
)
flags.DEFINE_bool(
	"resume_force",
	False,
	"Allow resuming from an earlier step even if later checkpoints exist in the same directory (may overwrite checkpoints).",
)

config_dir = os.path.join(os.path.dirname(__file__), "configs")
config_flags.DEFINE_config_file(
	"config",
	os.path.join(config_dir, "octo_pretrain_config.py"),
	"File path to the training hyperparameter configuration.",
	lock_config=False,
)


def _is_gcs_path(path: str) -> bool:
	return str(path).startswith("gs://")


def _infer_latest_step(checkpoint_path: str) -> int:
	ckpt = Path(checkpoint_path)
	if not ckpt.exists() or not ckpt.is_dir():
		raise FileNotFoundError(f"Checkpoint dir not found: {checkpoint_path}")
	steps = []
	for p in ckpt.iterdir():
		if p.is_dir() and p.name.isdigit():
			steps.append(int(p.name))
	if not steps:
		return 0
	return int(max(steps))


def _load_json(path: str) -> Dict[str, Any]:
	with tf.io.gfile.GFile(path, "r") as f:
		return json.load(f)


def _apply_dataset_override_from_flags(config: ConfigDict) -> None:
	"""Mutate config.dataset_kwargs based on --dataset_json / --dataset_mix_json."""
	if FLAGS.dataset_json and FLAGS.dataset_mix_json:
		raise ValueError("Pass only one of --dataset_json or --dataset_mix_json")

	if not FLAGS.dataset_json and not FLAGS.dataset_mix_json:
		return

	# Remove OXE auto-generation if present.
	if "oxe_kwargs" in config.dataset_kwargs:
		try:
			del config.dataset_kwargs["oxe_kwargs"]
		except Exception:
			pass

	if FLAGS.dataset_json:
		single = _load_json(FLAGS.dataset_json)
		config.dataset_kwargs["dataset_kwargs_list"] = [single]
		config.dataset_kwargs["sample_weights"] = [1.0]
		return

	mixture = _load_json(FLAGS.dataset_mix_json)
	if "dataset_kwargs_list" not in mixture:
		raise KeyError("dataset_mix_json must contain key 'dataset_kwargs_list'")
	config.dataset_kwargs["dataset_kwargs_list"] = mixture["dataset_kwargs_list"]
	if "sample_weights" in mixture:
		config.dataset_kwargs["sample_weights"] = mixture["sample_weights"]
	# Allow overriding top-level dataset_kwargs fields via the mix JSON.
	for k in (
		"batch_size",
		"shuffle_buffer_size",
		"balance_weights",
		"traj_transform_kwargs",
		"frame_transform_kwargs",
		"traj_transform_threads",
		"traj_read_threads",
	):
		if k in mixture:
			config.dataset_kwargs[k] = mixture[k]


def _detect_pretrained_format(pretrained_path: str, explicit_format: Optional[str]) -> str:
	if explicit_format not in (None, "", "auto"):
		return str(explicit_format)
	if pretrained_path.startswith("hf://"):
		return "jax"
	ckpt = Path(pretrained_path)
	if (ckpt / "example_batch.pickle").exists():
		return "torch"
	if (ckpt / "example_batch.msgpack").exists():
		return "jax"
	return "torch"


def _broadcast_str(distributed_state: PartialState, value: str) -> str:
	"""Broadcast a python string from rank0 to all ranks."""
	if distributed_state.num_processes <= 1:
		return value
	obj_list = [value]
	if torch.distributed.is_available() and torch.distributed.is_initialized():
		torch.distributed.broadcast_object_list(obj_list, src=0)
	return str(obj_list[0])


def _maybe_gfile_mkdir(path: str) -> None:
	# Supports local paths + gs:// via TF gfile.
	tf.io.gfile.makedirs(path)


def _configure_tf_threads_from_config(config: ConfigDict) -> None:
	"""Optionally configure TF intra/inter-op thread pools.

	This helps when you observe TF pipeline using only 1-2 cores.
	Set via:
	  config.dataset_kwargs.tf_intra_op_threads
	  config.dataset_kwargs.tf_inter_op_threads
	"""
	try:
		dk = config.get("dataset_kwargs", {})
		intra = dk.get("tf_intra_op_threads", None)
		inter = dk.get("tf_inter_op_threads", None)
		if intra not in (None, "", 0):
			tf.config.threading.set_intra_op_parallelism_threads(int(intra))
		if inter not in (None, "", 0):
			tf.config.threading.set_inter_op_parallelism_threads(int(inter))
	except Exception as exc:
		logging.warning("Could not set TF threading config: %s", exc)


def _read_cgroup_cpu_quota() -> Optional[float]:
	"""Best-effort: return effective CPU quota (cores) if running under cgroups."""
	# cgroup v2
	try:
		path = "/sys/fs/cgroup/cpu.max"
		if os.path.exists(path):
			content = Path(path).read_text().strip().split()
			if len(content) == 2:
				quota, period = content
				if quota != "max":
					q = float(quota)
					p = float(period)
					if p > 0:
						return q / p
	except Exception:
		pass
	# cgroup v1
	try:
		quota_path = "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"
		period_path = "/sys/fs/cgroup/cpu/cpu.cfs_period_us"
		if os.path.exists(quota_path) and os.path.exists(period_path):
			q = float(Path(quota_path).read_text().strip())
			p = float(Path(period_path).read_text().strip())
			if q > 0 and p > 0:
				return q / p
	except Exception:
		pass
	return None


class _MetricsLogger:
	def __init__(
		self,
		*,
		use_wandb: bool,
		debug: bool,
		wandb_config: Optional[Dict[str, Any]],
		run_name: str,
		run_id: str,
		jsonl_path: Optional[str],
		enable_console: bool,
	):
		self._use_wandb = bool(use_wandb)
		self._debug = bool(debug)
		self._wandb_config = wandb_config or {}
		self._run_name = str(run_name)
		self._run_id = str(run_id)
		self._jsonl_path = jsonl_path
		self._enable_console = bool(enable_console)

		if self._use_wandb and wandb is None:
			raise RuntimeError("--use_wandb=True but wandb is not installed/importable")

	def init(self, config: Dict[str, Any]) -> None:
		if self._use_wandb:
			assert wandb is not None
			wandb.init(
				config=config,
				id=self._run_id,
				name=self._run_name,
				mode="disabled" if self._debug else None,
				**self._wandb_config,
			)
		else:
			# No-op.
			pass

	def update_config(self, updates: Dict[str, Any]) -> None:
		if self._use_wandb:
			assert wandb is not None
			wandb.config.update(dict(updates), allow_val_change=True)

	def log(self, metrics: Dict[str, Any], *, step: int) -> None:
		# JSONL
		if self._jsonl_path:
			row = {"step": int(step), **metrics}
			with open(self._jsonl_path, "a", encoding="utf-8") as f:
				f.write(json.dumps(row) + "\n")

		# Console (lightweight)
		if self._enable_console:
			# Keep it compact.
			keys = [k for k in ("train/loss", "train/lr", "val/loss") if k in metrics]
			if keys:
				msg = " | ".join([f"{k}={metrics[k]:.6g}" for k in keys])
				logging.info("step=%d | %s", int(step), msg)

		# wandb
		if self._use_wandb:
			assert wandb is not None
			wandb.log(metrics, step=step)


def _summarize_batch_shapes(batch: Dict[str, Any]) -> Dict[str, Any]:
	def _shape(x):
		if hasattr(x, "shape"):
			return tuple(x.shape)
		return None

	out: Dict[str, Any] = {}
	for k, v in batch.items():
		if isinstance(v, dict):
			out[k] = {kk: _shape(vv) for kk, vv in v.items()}
		else:
			out[k] = _shape(v)
	return out


def _create_lr_scheduler(
	optimizer: torch.optim.Optimizer,
	lr_cfg: Dict[str, Any],
	*,
	last_epoch: int = -1,
) -> LambdaLR:
	"""Torch version of Octo's JAX learning-rate schedules.

	Expects a dict like:
	  {name, init_value, peak_value, warmup_steps, timescale/decay_steps}
	"""
	name = str(lr_cfg.get("name", "cosine"))
	init_value = float(lr_cfg.get("init_value", 0.0))
	peak_value = float(lr_cfg.get("peak_value", 3e-4))
	warmup_steps = int(lr_cfg.get("warmup_steps", 0))

	# Avoid divide-by-zero in factor computation.
	denom = peak_value if abs(peak_value) > 0 else 1.0

	if name == "constant":
		def lr_lambda(step: int) -> float:
			if step < warmup_steps:
				return (init_value + (peak_value - init_value) * (step / max(1, warmup_steps))) / denom
			return peak_value / denom

	elif name == "rsqrt":
		timescale = float(lr_cfg.get("timescale", 10000))

		def lr_lambda(step: int) -> float:
			if step < warmup_steps:
				return (init_value + (peak_value - init_value) * (step / max(1, warmup_steps))) / denom
			return (peak_value / np.sqrt((step + timescale) / timescale)) / denom

	elif name == "cosine":
		decay_steps = int(lr_cfg.get("decay_steps", lr_cfg.get("num_training_steps", 0)))
		if decay_steps <= 0:
			# If not given, fall back to a no-decay schedule.
			decay_steps = warmup_steps + 1

		def lr_lambda(step: int) -> float:
			if step < warmup_steps:
				return (init_value + (peak_value - init_value) * (step / max(1, warmup_steps))) / denom
			progress = (step - warmup_steps) / max(1, decay_steps - warmup_steps)
			progress = float(np.clip(progress, 0.0, 1.0))
			lr = 0.5 * peak_value * (1.0 + np.cos(np.pi * progress))
			return lr / denom

	else:
		raise ValueError(f"Unsupported learning_rate schedule: {name!r}")

	# PyTorch requires `initial_lr` when constructing a scheduler with last_epoch != -1.
	# This happens when resuming training from a nonzero step.
	if last_epoch != -1:
		for group in optimizer.param_groups:
			group.setdefault("initial_lr", group["lr"])

	return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def _loss_from_head_outputs(head_outputs: Dict[str, Any]) -> torch.Tensor:
	# By convention in this repo, action head returns (loss, metrics_dict)
	# and loss is stored at index 0.
	if "action" not in head_outputs:
		raise KeyError(f"Expected 'action' head in head_outputs, got keys={list(head_outputs.keys())}")
	loss = head_outputs["action"][0]
	if not isinstance(loss, torch.Tensor):
		loss = torch.as_tensor(loss)
	return loss


def _build_interleaved_dataset_and_loader(
	config: ConfigDict,
	text_processor,
	distributed_state: PartialState,
) -> Tuple[DataLoader, Dict[str, Any]]:
	"""Builds training dataloader + returns a dict with mixture metadata."""

	# Expand OXE mixture config into explicit dataset list/weights.
	dataset_kwargs = config.dataset_kwargs
	if "oxe_kwargs" in dataset_kwargs:
		dkwargs_list, weights = make_oxe_dataset_kwargs_and_weights(**dataset_kwargs["oxe_kwargs"])
		dataset_kwargs["dataset_kwargs_list"] = dkwargs_list
		dataset_kwargs["sample_weights"] = weights
		del dataset_kwargs["oxe_kwargs"]

	# Always let PyTorch handle batching.
	# By default, `dataset_kwargs.batch_size` is treated as GLOBAL batch size.
	# If `dataset_kwargs.per_device_batch_size` is provided, we derive global batch size
	# as per_device_batch_size * world_size.
	per_device_batch_size = int(dataset_kwargs.get("per_device_batch_size", 0) or 0)
	if per_device_batch_size > 0:
		global_batch_size = per_device_batch_size * int(distributed_state.num_processes)
	else:
		global_batch_size = int(dataset_kwargs["batch_size"])
	if global_batch_size % distributed_state.num_processes != 0:
		raise ValueError(
			f"dataset_kwargs.batch_size ({global_batch_size}) must be divisible by world_size ({distributed_state.num_processes})"
		)
	per_rank_batch = global_batch_size // distributed_state.num_processes

	tf_seed = int(config.seed) + int(distributed_state.process_index)
	tf.random.set_seed(tf_seed)

	num_workers = int(dataset_kwargs.get("dataloader_num_workers", 0) or 0)
	if distributed_state.is_main_process and num_workers > 0:
		logging.warning(
			"DataLoader num_workers=%d enabled: each worker will build its own TF/dlimp input pipeline "
			"(spawn-safe). Start with 2-4 workers; 8+ can be counterproductive.",
			num_workers,
		)
	# Build once on the main process to obtain dataset statistics.
	factory = RLDSDatasetFactory(
		dataset_kwargs_list=to_builtins(dataset_kwargs["dataset_kwargs_list"]),
		sample_weights=to_builtins(dataset_kwargs.get("sample_weights", None)),
		train=True,
		shuffle_buffer_size=int(dataset_kwargs["shuffle_buffer_size"]),
		traj_transform_kwargs=to_builtins(dataset_kwargs.get("traj_transform_kwargs", {})),
		frame_transform_kwargs=to_builtins(dataset_kwargs.get("frame_transform_kwargs", {})),
		balance_weights=bool(dataset_kwargs.get("balance_weights", False)),
		traj_transform_threads=to_builtins(dataset_kwargs.get("traj_transform_threads", None)),
		traj_read_threads=to_builtins(dataset_kwargs.get("traj_read_threads", None)),
		seed=int(tf_seed),
		tfds_imports=tuple(FLAGS.tfds_import),
		tf_intra_op_threads=int(dataset_kwargs.get("tf_intra_op_threads", 0) or 0),
		tf_inter_op_threads=int(dataset_kwargs.get("tf_inter_op_threads", 0) or 0),
	)
	rlds_dataset = factory(worker_id=0, num_workers=1)

	# For multiprocessing DataLoader, do NOT pass DLataset (not picklable). Pass the factory instead.
	# Also avoid pickling instantiated text_processor; pass the ModuleSpec dict when available.
	text_for_dataset = text_processor
	if num_workers > 0 and isinstance(config.text_processor, dict):
		text_for_dataset = config.text_processor
	copy_non_writable = bool(dataset_kwargs.get("copy_non_writable_arrays", False))
	iterator_prefetch = int(
		dataset_kwargs.get("rlds_iterator_prefetch", config.get("prefetch_num_batches", 0)) or 0
	)
	pytorch_dataset = TorchRLDSDataset(
		factory if num_workers > 0 else rlds_dataset,
		text_for_dataset,
		train=True,
		copy_non_writable_arrays=copy_non_writable,
		iterator_prefetch=iterator_prefetch,
	)
	num_workers = int(dataset_kwargs.get("dataloader_num_workers", 0) or 0)
	prefetch_factor = int(dataset_kwargs.get("dataloader_prefetch_factor", 2) or 2)
	persistent_workers = bool(dataset_kwargs.get("dataloader_persistent_workers", True))
	if num_workers < 0:
		raise ValueError(f"dataset_kwargs.dataloader_num_workers must be >= 0, got {num_workers}")
	# WARNING: TF + multiprocessing can be fragile. Keep this opt-in.
	# Use 'spawn' when using workers to reduce fork-related issues.
	dl_kwargs: Dict[str, Any] = {}
	if num_workers > 0:
		dl_kwargs["prefetch_factor"] = prefetch_factor
		dl_kwargs["persistent_workers"] = bool(persistent_workers)
		dl_kwargs["multiprocessing_context"] = str(
			dataset_kwargs.get("dataloader_multiprocessing_context", "spawn")
		)
	dataloader = DataLoader(
		pytorch_dataset,
		batch_size=per_rank_batch,
		num_workers=num_workers,
		pin_memory=torch.cuda.is_available(),
		drop_last=True,
		**dl_kwargs,
	)
	meta = {
		"global_batch_size": global_batch_size,
		"per_rank_batch": per_rank_batch,
		"dataset_statistics": getattr(rlds_dataset, "dataset_statistics", None),
	}
	return dataloader, meta


def _build_val_loader_if_needed(
	config: ConfigDict,
	text_processor,
	distributed_state: PartialState,
) -> Optional[DataLoader]:
	dataset_kwargs = config.dataset_kwargs

	# If user did not pass eval datasets, default to no-eval (keeps script simple/fast).
	eval_datasets = config.get("eval_datasets", None)
	if eval_datasets is None:
		return None

	dkwargs_list, weights = filter_eval_datasets(
		dataset_kwargs["dataset_kwargs_list"],
		dataset_kwargs.get("sample_weights", None),
		eval_datasets,
	)
	if not dkwargs_list:
		return None

	val_kwargs = config.get("val_kwargs", {})
	val_shuffle = int(val_kwargs.get("val_shuffle_buffer_size", 1000))
	per_device_batch_size = int(dataset_kwargs.get("per_device_batch_size", 0) or 0)
	if per_device_batch_size > 0:
		global_batch_size = per_device_batch_size * int(distributed_state.num_processes)
	else:
		global_batch_size = int(dataset_kwargs["batch_size"])
	per_rank_batch = global_batch_size // distributed_state.num_processes

	tf.random.set_seed(int(config.seed) + 10000 + int(distributed_state.process_index))

	num_workers = int(dataset_kwargs.get("dataloader_num_workers", 0) or 0)
	factory = RLDSDatasetFactory(
		dataset_kwargs_list=to_builtins(dkwargs_list),
		sample_weights=to_builtins(weights),
		train=False,
		shuffle_buffer_size=int(val_shuffle),
		traj_transform_kwargs=to_builtins(dataset_kwargs.get("traj_transform_kwargs", {})),
		frame_transform_kwargs=to_builtins(dataset_kwargs.get("frame_transform_kwargs", {})),
		balance_weights=bool(dataset_kwargs.get("balance_weights", False)),
		traj_transform_threads=to_builtins(dataset_kwargs.get("traj_transform_threads", None)),
		traj_read_threads=to_builtins(dataset_kwargs.get("traj_read_threads", None)),
		seed=int(config.seed) + 10000 + int(distributed_state.process_index),
		tfds_imports=tuple(FLAGS.tfds_import),
		tf_intra_op_threads=int(dataset_kwargs.get("tf_intra_op_threads", 0) or 0),
		tf_inter_op_threads=int(dataset_kwargs.get("tf_inter_op_threads", 0) or 0),
	)
	rlds_val = factory(worker_id=0, num_workers=1)

	text_for_dataset = text_processor
	if num_workers > 0 and isinstance(config.text_processor, dict):
		text_for_dataset = config.text_processor
	copy_non_writable = bool(dataset_kwargs.get("copy_non_writable_arrays", False))
	iterator_prefetch = int(
		dataset_kwargs.get("rlds_iterator_prefetch", config.get("prefetch_num_batches", 0)) or 0
	)
	pytorch_val = TorchRLDSDataset(
		factory if num_workers > 0 else rlds_val,
		text_for_dataset,
		train=False,
		copy_non_writable_arrays=copy_non_writable,
		iterator_prefetch=iterator_prefetch,
	)
	num_workers = int(dataset_kwargs.get("dataloader_num_workers", 0) or 0)
	prefetch_factor = int(dataset_kwargs.get("dataloader_prefetch_factor", 2) or 2)
	persistent_workers = bool(dataset_kwargs.get("dataloader_persistent_workers", True))
	dl_kwargs: Dict[str, Any] = {}
	if num_workers > 0:
		dl_kwargs["prefetch_factor"] = prefetch_factor
		dl_kwargs["persistent_workers"] = bool(persistent_workers)
		dl_kwargs["multiprocessing_context"] = str(
			dataset_kwargs.get("dataloader_multiprocessing_context", "spawn")
		)
	return DataLoader(
		pytorch_val,
		batch_size=per_rank_batch,
		num_workers=num_workers,
		pin_memory=torch.cuda.is_available(),
		drop_last=True,
		**dl_kwargs,
	)


def main(_):
	distributed_state = PartialState()

	use_cuda = torch.cuda.is_available()
	if not use_cuda and not FLAGS.allow_cpu:
		raise RuntimeError(
			"CUDA is not available (torch.cuda.is_available() is False). "
			"If you intended to train on GPU, ensure your environment/container exposes CUDA. "
			"Otherwise, pass --allow_cpu to run on CPU for debugging/dataset sanity-checks."
		)
	if use_cuda:
		torch.cuda.set_device(distributed_state.local_process_index)
		torch.cuda.empty_cache()
		device: torch.device = torch.device(f"cuda:{distributed_state.local_process_index}")
	else:
		if distributed_state.num_processes > 1:
			raise RuntimeError("CPU training with multi-process DDP is not supported by this script.")
		device = torch.device("cpu")

	if distributed_state.is_main_process:
		logging.set_verbosity(logging.INFO)
	else:
		logging.set_verbosity(logging.ERROR)
	if torch.cuda.is_available():
		torch.backends.cudnn.benchmark = True
		try:
			torch.set_float32_matmul_precision("high")
		except Exception:
			pass


	# Prevent tensorflow from using GPUs.
	tf.config.set_visible_devices([], "GPU")
	_configure_tf_threads_from_config(FLAGS.config)
	if distributed_state.is_main_process:
		cpu_quota = _read_cgroup_cpu_quota()
		logging.info("Host os.cpu_count()=%s | cgroup_cpu_quota=%s", os.cpu_count(), cpu_quota)
	if distributed_state.is_main_process:
		dk = FLAGS.config.get("dataset_kwargs", {})
		logging.info(
			"Input pipeline knobs | workers=%s prefetch_factor=%s iterator_prefetch=%s tf_intra=%s tf_inter=%s",
			dk.get("dataloader_num_workers", 0),
			dk.get("dataloader_prefetch_factor", 2),
			dk.get("rlds_iterator_prefetch", FLAGS.config.get("prefetch_num_batches", 0)),
			dk.get("tf_intra_op_threads", None),
			dk.get("tf_inter_op_threads", None),
		)

	# Import any extra TFDS builders (useful for custom datasets).
	for mod in FLAGS.tfds_import:
		if mod:
			importlib.import_module(mod)

	# Build run id and logging.
	run_name = format_name_with_config(FLAGS.name, FLAGS.config.to_dict())
	run_id = f"{run_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
	if distributed_state.is_main_process:
		run_id = _broadcast_str(distributed_state, run_id)
	else:
		run_id = _broadcast_str(distributed_state, "")

	use_wandb = bool(FLAGS.use_wandb)

	# Output directory (PyTorch checkpoint format)
	# The output folder will match the structure:
	#   save_dir/config.json
	#   save_dir/example_batch.pickle
	#   save_dir/dataset_statistics.json
	#   save_dir/<step>/weights.pth
	resume_from = FLAGS.resume_from
	is_resume = resume_from not in (None, "")
	base_save_dir = FLAGS.config.get("save_dir", None)
	save_dir = None
	if is_resume:
		if _is_gcs_path(str(resume_from)):
			raise ValueError(
				"Resume requires a local filesystem path (torch checkpoints). "
				f"Got resume_from={resume_from!r}."
			)
		save_dir = str(resume_from)
		if distributed_state.is_main_process:
			ckpt_dir = Path(save_dir)
			if not ckpt_dir.exists():
				raise FileNotFoundError(f"Resume dir not found: {save_dir}")
			if not (ckpt_dir / "config.json").exists():
				raise FileNotFoundError(
					f"Resume dir does not look like a torch checkpoint (missing config.json): {save_dir}"
				)
			logging.info("Resuming training in-place from %s", save_dir)
	elif base_save_dir not in (None, ""):
		if _is_gcs_path(str(base_save_dir)):
			raise ValueError(
				"PyTorch checkpoint saving uses torch.save and requires a local filesystem path. "
				f"Got save_dir={base_save_dir!r}. Please set save_dir to a local path (e.g. /home/... )."
			)
		# By default, save directly into save_dir (no extra nesting)
		append_run_id = bool(FLAGS.config.get("append_wandb_id_to_save_dir", False))
		save_dir = os.path.join(str(base_save_dir), str(run_id)) if append_run_id else str(base_save_dir)
		if distributed_state.is_main_process:
			os.makedirs(save_dir, exist_ok=True)
			logging.info("Saving PyTorch checkpoints to %s", save_dir)
	else:
		if distributed_state.is_main_process:
			logging.info("save_dir not passed in, not saving checkpoints")

	# Initialize metrics logger (rank0 only).
	logger = None
	if distributed_state.is_main_process:
		metrics_jsonl = FLAGS.metrics_jsonl
		if metrics_jsonl in (None, "") and save_dir is not None:
			metrics_jsonl = os.path.join(save_dir, "metrics.jsonl")
		wandb_cfg = {}
		try:
			if hasattr(FLAGS.config, "wandb") and hasattr(FLAGS.config.wandb, "to_dict"):
				wandb_cfg = FLAGS.config.wandb.to_dict()
		except Exception:
			wandb_cfg = {}
		logger = _MetricsLogger(
			use_wandb=use_wandb,
			debug=bool(FLAGS.debug),
			wandb_config=wandb_cfg,
			run_name=run_name,
			run_id=run_id,
			jsonl_path=metrics_jsonl,
			enable_console=True,
		)
		logger.init(FLAGS.config.to_dict())
		if save_dir is not None:
			logger.update_config(dict(save_dir=save_dir))

	# Text processor
	if FLAGS.config.text_processor is None:
		text_processor = None
	else:
		text_processor = ModuleSpec.instantiate(FLAGS.config.text_processor)()

	# Dataset overrides (from JSON spec)
	_apply_dataset_override_from_flags(FLAGS.config)

	# Optional: force train-from-scratch.
	if FLAGS.scratch:
		if is_resume:
			raise ValueError("Cannot use --scratch together with --resume_from")
		FLAGS.config.pretrained_path = None
		FLAGS.config.pretrained_format = "auto"
		FLAGS.config.pretrained_step = None
		FLAGS.config.load_optimizer_state = False
		FLAGS.config.start_step = 0

	# Data
	train_loader, train_meta = _build_interleaved_dataset_and_loader(
		FLAGS.config, text_processor, distributed_state
	)
	val_loader = _build_val_loader_if_needed(FLAGS.config, text_processor, distributed_state)

	example_batch = next(iter(train_loader))
	if distributed_state.is_main_process:
		logging.info("Example batch shapes: %s", _summarize_batch_shapes(example_batch))

	# Build / load model
	pretrained_path = FLAGS.config.get("pretrained_path", None)
	if is_resume:
		pretrained_path = str(save_dir)
	pretrained_format = None
	loaded_optimizer_state = None
	raw_start_step = FLAGS.config.get("start_step", None)
	start_step: Optional[int]
	if raw_start_step in (None, ""):
		start_step = None
	else:
		start_step = int(raw_start_step)

	# Safety: avoid accidentally overwriting an existing checkpoint folder.
	if distributed_state.is_main_process and save_dir is not None:
		overwrite_output = bool(FLAGS.config.get("overwrite_output", False))
		if (Path(save_dir) / "config.json").exists() and not overwrite_output and not is_resume:
			raise FileExistsError(
				f"Output dir already contains a checkpoint (config.json exists): {save_dir}. "
				"Set config.overwrite_output=True or choose a new save_dir."
			)

	if pretrained_path not in (None, ""):
		explicit_format = "auto" if is_resume else FLAGS.config.get("pretrained_format", "auto")
		pretrained_format = _detect_pretrained_format(str(pretrained_path), explicit_format)
		if is_resume and pretrained_format != "torch":
			raise ValueError(
				f"--resume_from expects a torch checkpoint directory, but detected pretrained_format={pretrained_format!r} "
				f"for path={pretrained_path!r}"
			)
		if pretrained_format == "torch":
			# Determine which step we are loading so we can default start_step correctly.
			if is_resume:
				resume_step = int(FLAGS.resume_step)
				if resume_step < -1:
					raise ValueError(f"--resume_step must be >= -1, got {resume_step}")
				latest_step = _infer_latest_step(str(pretrained_path))
				loaded_step = latest_step if resume_step == -1 else resume_step

				# Safety: avoid clobbering existing later checkpoints unless explicitly forced.
				if not bool(FLAGS.resume_force) and latest_step > loaded_step:
					raise ValueError(
						f"Refusing to resume from step {loaded_step} because later checkpoints exist "
						f"(latest={latest_step}) in {pretrained_path}. "
						"Pass --resume_step=-1 to resume from latest, or pass --resume_force to overwrite later checkpoints."
					)

				load_opt_state = True
			else:
				pretrained_step = FLAGS.config.get("pretrained_step", None)
				loaded_step = (
					int(pretrained_step)
					if pretrained_step not in (None, "")
					else _infer_latest_step(str(pretrained_path))
				)
				load_opt_state = bool(FLAGS.config.get("load_optimizer_state", False))

			try:
				loaded = OctoModelPt.load_pretrained(
					str(pretrained_path),
					step=loaded_step,
					load_optimizer_state=load_opt_state,
				)
			except AssertionError as exc:
				if is_resume and load_opt_state:
					logging.warning(
						"Checkpoint at %s (step=%d) has no optimizer state; continuing with a fresh optimizer. "
						"To make future resumes exact, set config.save_optimizer_state=True when training. (%s)",
						str(pretrained_path),
						int(loaded_step),
						str(exc),
					)
					loaded = OctoModelPt.load_pretrained(
						str(pretrained_path),
						step=loaded_step,
						load_optimizer_state=False,
					)
				else:
					raise

			model = loaded["octo_model"]
			loaded_optimizer_state = loaded.get("optimizer_state_dict", None)
			if start_step is None:
				start_step = loaded_step
		elif pretrained_format == "jax":
			meta = OctoModelPt.load_config_and_meta_from_jax(
				str(pretrained_path), return_jax_meta=True
			)
			meta["config"]["model"] = _jax_config_to_pt_config(meta["config"]["model"])
			model = OctoModelPt.from_config(
				config=meta["config"],
				example_batch=example_batch,
				text_processor=text_processor,
				dataset_statistics=_np2pt(train_meta.get("dataset_statistics", {}) or {}),
			)
			model.load_weights_from_jax(
				str(pretrained_path),
				step=FLAGS.config.get("pretrained_step", None),
				skip_keys=FLAGS.config.to_dict().get("skip_keys", []),
				skip_keys_regex=FLAGS.config.to_dict().get("skip_keys_regex", ".*hf_model"),
				non_strict_keys=FLAGS.config.to_dict().get("non_strict_keys", []),
				non_strict_keys_regex=FLAGS.config.to_dict().get("non_strict_keys_regex", None),
			)
			# If the user didn't specify start_step, default to the requested pretrained_step (or 0).
			if start_step is None:
				pretrained_step = FLAGS.config.get("pretrained_step", None)
				start_step = int(pretrained_step) if pretrained_step not in (None, "") else 0
		else:
			raise ValueError(f"Unsupported pretrained_format: {pretrained_format!r}")

		# Update meta to match the actual training data.
		model.example_batch = example_batch
		if train_meta.get("dataset_statistics", None) is not None:
			model.dataset_statistics = _np2pt(train_meta["dataset_statistics"])
	else:
		# Initialize from config (JAX-style config converted to PT).
		config_dict = FLAGS.config.to_dict()
		config_dict["model"] = _jax_config_to_pt_config(config_dict["model"])
		model = OctoModelPt.from_config(
			config=config_dict,
			example_batch=example_batch,
			text_processor=text_processor,
			dataset_statistics=_np2pt(train_meta.get("dataset_statistics", {}) or {}),
			verbose=distributed_state.is_main_process,
		)
		if start_step is None:
			start_step = 0

	model = model.to(device)
	if distributed_state.num_processes > 1:
		# Only wrap with DDP when launched with accelerate/torchrun.
		# device_ids should only be set for CUDA; for CPU DDP (not supported here), it would be None.
		ddp_kwargs = dict(find_unused_parameters=True, gradient_as_bucket_view=True)
		if use_cuda:
			model = DDP(model, device_ids=[distributed_state.local_process_index], **ddp_kwargs)
		else:
			model = DDP(model, **ddp_kwargs)

	# Unwrap DDP for saving checkpoints. Note: OctoModelPt has attribute `.module` (OctoModulePt),
	# so we must only unwrap when the outer object is actually a DDP wrapper.
	model_to_save = model.module if isinstance(model, DDP) else model

	# Optimizer
	opt_cfg = FLAGS.config.optimizer.to_dict() if hasattr(FLAGS.config.optimizer, "to_dict") else dict(FLAGS.config.optimizer)
	lr_cfg = opt_cfg.get("learning_rate", {})
	peak_lr = float(lr_cfg.get("peak_value", 3e-4))
	weight_decay = float(opt_cfg.get("weight_decay", 0.0))
	clip_grad = float(opt_cfg.get("clip_gradient", 0.0) or 0.0)

	# Freeze weights (match JAX config semantics)
	frozen_keys = opt_cfg.get("frozen_keys", None) or ()
	freeze_weights_pt(model, list(frozen_keys))

	trainable_params = [p for p in model.parameters() if p.requires_grad]
	optimizer = AdamW(trainable_params, lr=peak_lr, weight_decay=weight_decay)
	scheduler = _create_lr_scheduler(
		optimizer,
		{**lr_cfg, "num_training_steps": int(FLAGS.config.num_steps)},
		last_epoch=start_step - 1,
	)

	if loaded_optimizer_state is not None:
		optimizer.load_state_dict(loaded_optimizer_state)
		if distributed_state.is_main_process:
			logging.info("Restored optimizer state from checkpoint")

	# Optionally save an initial checkpoint (step 0) for from-scratch runs.
	save_initial = bool(FLAGS.config.get("save_initial_checkpoint", True))
	if (
		distributed_state.is_main_process
		and save_dir is not None
		and save_initial
		and int(start_step or 0) == 0
	):
		logging.info("Saving initial checkpoint (step 0)")
		model_to_save.save_pretrained(step=0, checkpoint_path=save_dir, optimizer=None)

	# Train loop
	num_steps = int(FLAGS.config.num_steps)
	log_interval = int(FLAGS.config.get("log_interval", 100) or 0)
	save_interval = int(FLAGS.config.get("save_interval", 10000) or 0)
	eval_interval = int(FLAGS.config.get("eval_interval", 0) or 0)
	grad_accum_steps = int(FLAGS.config.get("grad_accum_steps", 1) or 1)
	if grad_accum_steps < 1:
		raise ValueError(f"grad_accum_steps must be >= 1, got {grad_accum_steps}")
	val_kwargs = FLAGS.config.get("val_kwargs", ConfigDict())
	num_val_batches = int(val_kwargs.get("num_val_batches", 0) or 0)

	if distributed_state.is_main_process:
		logging.info(
			"PT pretrain | world=%d | global_bs=%d | per_rank_bs=%d | steps=%d | pretrained=%s",
			distributed_state.num_processes,
			train_meta["global_batch_size"],
			train_meta["per_rank_batch"],
			num_steps,
			str(pretrained_path),
		)

	# Ensure reproducibility-ish: torch seed per rank.
	torch.manual_seed(int(FLAGS.config.seed) + int(distributed_state.process_index))

	train_iter = iter(train_loader)
	val_iter = iter(val_loader) if val_loader is not None else None

	# Lightweight profiler (enabled by default; logs at log_interval)
	prof_enabled = bool(FLAGS.config.get("profile_timing", True))
	prof_window = int(FLAGS.config.get("profile_timing_window", max(10, log_interval or 10)) or 10)
	prof = {"data_wait_s": 0.0, "h2d_s": 0.0, "compute_s": 0.0, "steps": 0}

	assert start_step is not None
	for step in tqdm.tqdm(range(int(start_step), num_steps), disable=not distributed_state.is_main_process, dynamic_ncols=True):
		model.train()
		optimizer.zero_grad(set_to_none=True)
		step_t0 = time.perf_counter()

		loss_for_step = torch.zeros((), device=device)
		for micro_step in range(grad_accum_steps):
			get_t0 = time.perf_counter()
			try:
				batch = next(train_iter)
			except StopIteration:
				train_iter = iter(train_loader)
				batch = next(train_iter)
			get_t1 = time.perf_counter()
			h2d_t0 = time.perf_counter()
			batch = _to_device(batch, device)
			h2d_t1 = time.perf_counter()

			# For DDP, avoid gradient sync on intermediate micro-steps.
			sync_ctx = nullcontext()
			if isinstance(model, DDP) and micro_step < grad_accum_steps - 1:
				sync_ctx = model.no_sync()
			with sync_ctx:
				comp_t0 = time.perf_counter()
				_, head_outputs = model(
					observations=batch["observation"],
					tasks=batch["task"],
					timestep_pad_mask=batch["observation"]["timestep_pad_mask"],
					action_pad_mask=batch.get("action_pad_mask", None),
					gt_actions=batch.get("action", None),
					train=True,
					verbose=False,
					save_attention_mask=True,
				)
				loss = _loss_from_head_outputs(head_outputs)
				loss = loss / float(grad_accum_steps)
				loss.backward()
				loss_for_step = loss_for_step + loss.detach()
				comp_t1 = time.perf_counter()

			if prof_enabled:
				prof["data_wait_s"] += (get_t1 - get_t0)
				prof["h2d_s"] += (h2d_t1 - h2d_t0)
				prof["compute_s"] += (comp_t1 - comp_t0)

		if clip_grad and clip_grad > 0:
			torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=clip_grad)

		optimizer.step()
		scheduler.step()
		if prof_enabled:
			prof["steps"] += 1

		# Reduce loss across ranks for clean logging.
		loss_for_log = loss_for_step
		if distributed_state.num_processes > 1 and torch.distributed.is_initialized():
			torch.distributed.all_reduce(loss_for_log, op=torch.distributed.ReduceOp.SUM)
			loss_for_log = loss_for_log / float(distributed_state.num_processes)

		if distributed_state.is_main_process and log_interval > 0 and (step + 1) % log_interval == 0:
			assert logger is not None
			logger.log(
				{
					"train/loss": float(loss_for_log.item()),
					"train/lr": float(scheduler.get_last_lr()[0]),
				},
				step=step,
			)
			if prof_enabled and prof["steps"] > 0:
				# Report average per *micro-step* (data/h2d/compute) and per optimizer step.
				avg_data = prof["data_wait_s"] / float(max(1, prof_window * grad_accum_steps))
				avg_h2d = prof["h2d_s"] / float(max(1, prof_window * grad_accum_steps))
				avg_comp = prof["compute_s"] / float(max(1, prof_window * grad_accum_steps))
				logging.info(
					"timing(avg per micro-step over ~%d steps): data_wait=%.4fs | h2d=%.4fs | compute=%.4fs",
					prof_window,
					avg_data,
					avg_h2d,
					avg_comp,
				)
				prof = {"data_wait_s": 0.0, "h2d_s": 0.0, "compute_s": 0.0, "steps": 0}

		if (
			distributed_state.is_main_process
			and save_dir is not None
			and save_interval > 0
			and (step + 1) % save_interval == 0
		):
			logging.info("Saving checkpoint at step %d", step)
			save_optimizer_state = bool(FLAGS.config.get("save_optimizer_state", False))
			model_to_save.save_pretrained(
				step=step,
				checkpoint_path=save_dir,
				optimizer=optimizer if save_optimizer_state else None,
			)

		# Basic evaluation loop (optional). Must run on *all* ranks to avoid DDP hangs.
		if (
			eval_interval > 0
			and val_iter is not None
			and num_val_batches > 0
			and (step + 1) % eval_interval == 0
		):
			model.eval()
			losses = []
			with torch.no_grad():
				for _ in range(num_val_batches):
					try:
						vbatch = next(val_iter)
					except StopIteration:
						val_iter = iter(val_loader)
						vbatch = next(val_iter)
					vbatch = _to_device(vbatch, device)
					# IMPORTANT: when train=False, OctoModulePt returns head *predictions*, not (loss, metrics).
					# So we compute val loss explicitly via action head loss.
					transformer_outputs, _ = model(
						observations=vbatch["observation"],
						tasks=vbatch["task"],
						timestep_pad_mask=vbatch["observation"]["timestep_pad_mask"],
						train=False,
						transformer_only=True,
						verbose=False,
						save_attention_mask=True,
					)
					# Unwrap DDP to access the action head.
					action_head = model_to_save.module.heads["action"]
					if "action" not in vbatch:
						raise KeyError(
							"Validation requires ground-truth actions in batch['action'] to compute val/loss."
						)
					vloss, _ = action_head.loss(
						transformer_outputs,
						vbatch["action"],
						vbatch["observation"]["timestep_pad_mask"],
						vbatch.get("action_pad_mask", None),
						train=False,
					)
					vloss = vloss.detach()
					losses.append(vloss)

			mean_vloss = torch.stack(losses).mean()
			if distributed_state.num_processes > 1 and torch.distributed.is_initialized():
				torch.distributed.all_reduce(mean_vloss, op=torch.distributed.ReduceOp.SUM)
				mean_vloss = mean_vloss / float(distributed_state.num_processes)
			if distributed_state.is_main_process:
				logger.log({"val/loss": float(mean_vloss.item())}, step=step)

	# Final checkpoint
	save_final = bool(FLAGS.config.get("save_final_checkpoint", True))
	if distributed_state.is_main_process and save_dir is not None and save_final:
		final_step = int(num_steps)
		logging.info("Saving final checkpoint (step %d)", final_step)
		save_optimizer_state = bool(FLAGS.config.get("save_optimizer_state", False))
		model_to_save.save_pretrained(
			step=final_step,
			checkpoint_path=save_dir,
			optimizer=optimizer if save_optimizer_state else None,
		)


if __name__ == "__main__":
	app.run(main)

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple
import importlib

import tensorflow as tf

try:
    from ml_collections import ConfigDict
except Exception:  # pragma: no cover
    ConfigDict = None  # type: ignore

from octo.data.dataset import make_interleaved_dataset


def to_builtins(x: Any) -> Any:
    """Convert ConfigDict-like objects into plain Python containers for pickling."""
    try:
        if ConfigDict is not None and isinstance(x, ConfigDict):
            x = x.to_dict()
        elif hasattr(x, "to_dict") and callable(getattr(x, "to_dict")):
            x = x.to_dict()
    except Exception:
        pass

    if isinstance(x, dict):
        return {k: to_builtins(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_builtins(v) for v in x]
    return x


@dataclass(frozen=True)
class RLDSDatasetFactory:
    """Picklable factory that builds a TF/dlimp RLDS dataset inside each worker process."""

    dataset_kwargs_list: Any
    sample_weights: Any
    train: bool
    shuffle_buffer_size: int
    traj_transform_kwargs: Any
    frame_transform_kwargs: Any
    balance_weights: bool
    traj_transform_threads: Any
    traj_read_threads: Any
    seed: int
    tfds_imports: Tuple[str, ...]
    tf_intra_op_threads: int
    tf_inter_op_threads: int

    def __call__(self, *, worker_id: int = 0, num_workers: int = 1):
        # Worker-local TF safety: keep TF off GPU and optionally set thread pools.
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass

        try:
            if self.tf_intra_op_threads and int(self.tf_intra_op_threads) > 0:
                tf.config.threading.set_intra_op_parallelism_threads(int(self.tf_intra_op_threads))
            if self.tf_inter_op_threads and int(self.tf_inter_op_threads) > 0:
                tf.config.threading.set_inter_op_parallelism_threads(int(self.tf_inter_op_threads))
        except Exception:
            pass

        for mod in self.tfds_imports:
            if mod:
                importlib.import_module(mod)

        # Distinct seed per worker to avoid identical shuffles.
        try:
            tf.random.set_seed(int(self.seed) + 1000 * int(worker_id))
        except Exception:
            pass

        return make_interleaved_dataset(
            self.dataset_kwargs_list,
            self.sample_weights,
            train=bool(self.train),
            shuffle_buffer_size=int(self.shuffle_buffer_size),
            traj_transform_kwargs=self.traj_transform_kwargs or {},
            frame_transform_kwargs=self.frame_transform_kwargs or {},
            batch_size=None,
            balance_weights=bool(self.balance_weights),
            traj_transform_threads=self.traj_transform_threads,
            traj_read_threads=self.traj_read_threads,
        )

"""Lightweight training utilities (JAX-free).

This module exists to support the PyTorch training / finetuning pipeline without
importing JAX/Flax (which can be heavy and can cause DataLoader workers created
with multiprocessing 'spawn' to spend a long time importing JAX-only deps).

Keep this file dependency-light on purpose.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple


def _maybe_to_dict(x: Any) -> Any:
    try:
        if hasattr(x, "to_dict") and callable(getattr(x, "to_dict")):
            return x.to_dict()
    except Exception:
        pass
    return x


def _flatten_dict(d: Dict[str, Any], parent: Tuple[str, ...] = ()) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    d = _maybe_to_dict(d)
    if not isinstance(d, dict):
        return out

    for k, v in d.items():
        key = parent + (str(k),)
        v = _maybe_to_dict(v)
        if isinstance(v, dict):
            out.update(_flatten_dict(v, key))
        else:
            out["_".join(key)] = v
    return out


def format_name_with_config(name: str, config: Dict[str, Any]) -> str:
    """Formats a name string with a (nested) config dict.

    Formatting keys may be specified as {key} or {full_path_to_key_with_underscores}.

    This mirrors the behavior of the JAX version but avoids importing flax.
    """
    flat = _flatten_dict(config)
    # allow short-hands like {batch_size} in addition to {dataset_kwargs_batch_size}
    final = {k.split("_")[-1]: v for k, v in flat.items()}
    format_dict = {**final, **flat}
    try:
        return str(name).format(**format_dict)
    except Exception:
        # If formatting fails (missing keys, bad placeholders), return name unchanged.
        return str(name)


def filter_eval_datasets(
    dataset_kwargs_list: Sequence[dict],
    sample_weights: Optional[Sequence[float]],
    eval_datasets: Optional[Sequence[str]] = None,
):
    """Filter a dataset mixture down to the requested eval datasets.

    Returns (filtered_dataset_kwargs_list, filtered_sample_weights).
    """
    if sample_weights is None:
        sample_weights = [1.0] * len(dataset_kwargs_list)

    if eval_datasets is None:
        return list(dataset_kwargs_list), list(sample_weights)

    if len(eval_datasets) == 0:
        return [], []

    filtered_kwargs: List[dict] = []
    filtered_weights: List[float] = []
    eval_set = set(eval_datasets)
    for dkwargs, w in zip(dataset_kwargs_list, sample_weights):
        if dkwargs.get("name") in eval_set:
            filtered_kwargs.append(dkwargs)
            filtered_weights.append(float(w))

    return filtered_kwargs, filtered_weights


def process_text(batch: Dict[str, Any], text_processor: Any) -> Dict[str, Any]:
    """Encode task language_instruction using the provided text_processor.

    If text_processor is None, removes language_instruction from batch['task'].

    Expects:
      batch['task']['language_instruction'] is a 1D sequence of bytes/strings.
    """
    task = batch.get("task", None)
    if not isinstance(task, dict):
        return batch

    if text_processor is None:
        task.pop("language_instruction", None)
        return batch

    if "language_instruction" not in task:
        return batch

    raw = task["language_instruction"]
    # raw is usually a numpy array of dtype=object/bytes or a list.
    strings: List[str] = []
    try:
        iterator = list(raw)
    except Exception:
        iterator = [raw]

    for s in iterator:
        if isinstance(s, (bytes, bytearray)):
            strings.append(s.decode("utf-8"))
        else:
            strings.append(str(s))

    task["language_instruction"] = text_processor.encode(strings)
    return batch

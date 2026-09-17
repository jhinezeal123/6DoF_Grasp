"""Lightweight typing helpers.

This repository includes both the original JAX/Flax Octo implementation and a
PyTorch port. For Jetson deployment we often want to import the PyTorch model
without installing JAX/Flax/TensorFlow.

To support that, this module *must not hard-require* JAX.
"""

from typing import Any, Mapping, Sequence, Union, Tuple

try:
    import jax  # type: ignore

    PRNGKey = jax.random.KeyArray
    PyTree = Union[jax.typing.ArrayLike, Mapping[str, "PyTree"]]
    Dtype = jax.typing.DTypeLike
except Exception:  # pragma: no cover
    # Fallback types when JAX is not installed (e.g., Jetson torch-only runtime).
    PRNGKey = Any
    PyTree = Any
    Dtype = Any

Config = Union[Any, Mapping[str, "Config"]]
Params = Mapping[str, PyTree]
Data = Mapping[str, PyTree]
Shape = Sequence[int]

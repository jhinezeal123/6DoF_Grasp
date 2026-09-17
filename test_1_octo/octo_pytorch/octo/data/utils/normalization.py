from enum import Enum


class NormalizationType(str, Enum):
    """Defines supported normalization schemes for action and proprio.

    This small enum is separated from `data_utils.py` so torch-only deployments
    (e.g. Jetson) do not need to import TensorFlow.
    """

    NORMAL = "normal"  # normalize to mean 0, std 1
    BOUNDS = "bounds"  # normalize to [-1, 1]
    BOUNDS_Q99 = "bounds_q99"  # normalize to [-1, 1] using q01/q99 bounds

"""octo-pytorch package.

This fork is frequently used in "torch-only" environments (e.g., Jetson).
To avoid importing heavy/optional backends via HuggingFace Transformers, we
default-disable TF/Flax/JAX backends unless the user explicitly enables them.
"""

import os

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TRANSFORMERS_NO_JAX", "1")


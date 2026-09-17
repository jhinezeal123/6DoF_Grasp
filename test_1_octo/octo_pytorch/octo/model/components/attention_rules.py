from enum import Enum


class AttentionRule(Enum):
    """Enum describing when to attend to another token group.

    This is duplicated from the JAX implementation but placed in a torch-only
    friendly module (no JAX/Flax imports), so Jetson deployments can import the
    PyTorch model without installing JAX/Flax.
    """

    NEVER = "never"
    CAUSAL = "other.timestep <= self.timestep"
    CURRENT = "other.timestep == self.timestep"
    STRICT_PAST = "other.timestep < self.timestep"
    ALL = "all"  # Breaks causal structure! Be careful

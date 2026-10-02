#!/usr/bin/env python3
"""Offline photo replay and simulation-only grasp validation.

The implementation lives in m750.sim; this launcher stays because the run book,
the acceptance checklist and the CI gate all invoke this exact path.
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

from m750.sim.cli import main

if __name__ == "__main__":
    raise SystemExit(main())

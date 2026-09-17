"""Chạy thử pipeline bằng ba module mock.

Từ thư mục ``SDK`` chạy:

    python -m pipeline.examples.mock_run
"""

from __future__ import annotations

import time

from ..adapters import DryRunSink, ModelPolicy
from ..interfaces import Policy, Source
from ..runner import Pipeline
from ..types import Action, ActionSpec, Episode, Observation


SPEC = ActionSpec("joint_position", ("joint_1",), ("rad",), "joint")


class MockSource(Source):
    def __init__(self) -> None:
        self.sequence = 0

    def reset(self, episode: Episode) -> None:
        self.sequence = 0

    def read(self, timeout_s: float) -> Observation:
        self.sequence += 1
        stamp = time.time_ns()
        return Observation(
            images={"front": "mock_rgb"},
            state=(0.0,),
            stamps_ns={"front": stamp, "state": stamp},
            valid_until=time.monotonic() + 1.0,
            sequence=self.sequence,
        )


class MockModel:
    def inference(self, payload):
        assert payload["instruction"]
        return (0.1,)


def main() -> None:
    sink = DryRunSink(SPEC)
    pipeline = Pipeline(
        source=MockSource(),
        policy=ModelPolicy(MockModel(), SPEC),
        sink=sink,
        rate_hz=100.0,
    )
    steps = pipeline.run("đặt khối đỏ vào hộp", max_steps=3)
    print(f"steps={steps}, actions={len(sink.actions)}, halts={sink.halt_reasons}")


if __name__ == "__main__":
    main()

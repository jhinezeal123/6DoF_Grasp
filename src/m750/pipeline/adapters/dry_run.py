"""Sink dùng để kiểm tra model và guard mà không gửi lệnh tới robot."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..interfaces import Sink
from ..types import Action, ActionSpec


@dataclass
class DryRunSink(Sink):
    """Lưu các action đã nhận để debug hoặc kiểm tra offline."""

    spec: ActionSpec
    actions: list[Action] = field(default_factory=list)
    halt_reasons: list[str] = field(default_factory=list)

    def write(self, action: Action) -> None:
        self.actions.append(action)

    def halt(self, reason: str) -> None:
        self.halt_reasons.append(reason)


__all__ = ["DryRunSink"]

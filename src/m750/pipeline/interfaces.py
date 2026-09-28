"""Interface nhỏ để thay Source, Policy và Sink mà không sửa runner."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .types import Action, ActionSpec, Episode, Observation


class Module:
    """Vòng đời mặc định no-op; adapter chỉ override phần mình sở hữu."""

    def open(self) -> None:
        pass

    def reset(self, episode: Episode) -> None:
        pass

    def close(self) -> None:
        pass


class Source(Module, ABC):
    @abstractmethod
    def read(self, timeout_s: float) -> Observation:
        """Trả snapshot mới hoặc ném TimeoutError trong thời gian cho phép."""


class Policy(Module, ABC):
    @abstractmethod
    def predict(self, observation: Observation, instruction: str) -> Action:
        """Suy luận một lệnh; mã hóa model thuộc về adapter."""


class Sink(Module, ABC):
    def __init__(self, spec: ActionSpec) -> None:
        self.spec = spec

    @abstractmethod
    def write(self, action: Action) -> None:
        """Bàn giao một lệnh."""

    @abstractmethod
    def halt(self, reason: str) -> None:
        """Dừng/hủy/giữ theo controller; phải idempotent."""


__all__ = ["Module", "Policy", "Sink", "Source"]

"""Các kiểu dữ liệu dùng chung của pipeline VLA.

Pipeline chỉ truyền các kiểu ở module này giữa Source, Policy và Sink. Các
adapter cụ thể của SDK có thể giữ dữ liệu gốc trong ``metadata`` nếu model cần
thêm thông tin ngoài ảnh và trạng thái khớp.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class Observation:
    """Một snapshot hoàn chỉnh, độc lập với snapshot kế tiếp."""

    images: Mapping[str, Any]
    state: tuple[float, ...]
    stamps_ns: Mapping[str, int]
    valid_until: float
    sequence: int
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ActionSpec:
    """Hợp đồng lệnh mà Sink nhận."""

    mode: str
    axes: tuple[str, ...]
    units: tuple[str, ...]
    frame_id: str

    def __post_init__(self) -> None:
        if not self.axes or len(self.axes) != len(self.units):
            raise ValueError("axes và units phải có cùng độ dài khác không")
        if len(set(self.axes)) != len(self.axes):
            raise ValueError("Action axes phải là các tên duy nhất")


@dataclass(frozen=True)
class Action:
    values: tuple[float, ...]
    spec: ActionSpec


@dataclass(frozen=True)
class Episode:
    id: str
    instruction: str


@dataclass(frozen=True)
class StepResult:
    episode: Episode
    index: int
    observation: Observation
    action: Action
    elapsed_s: float


__all__ = ["Action", "ActionSpec", "Episode", "Observation", "StepResult"]

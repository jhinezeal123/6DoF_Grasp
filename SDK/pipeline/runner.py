"""Runner đồng bộ, single-owner, cho các module VLA."""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable, Sequence
from typing import Any
from uuid import uuid4

from .interfaces import Module, Policy, Sink, Source
from .types import Action, Episode, Observation, StepResult


class PipelineStopped(RuntimeError):
    """Được ném trong step khi stop() đã được yêu cầu."""


class Pipeline:
    """Ghép Source → preprocess → Policy → postprocess/guard → Sink.

    Pipeline chỉ chạy trong thread đã mở context. ``stop()`` là ngoại lệ duy
    nhất được gọi từ thread khác và chỉ yêu cầu dừng hợp tác.
    """

    def __init__(
        self,
        *,
        source: Source,
        policy: Policy,
        sink: Sink,
        preprocess: Sequence[Callable[[Observation], Observation]] = (),
        postprocess: Sequence[Callable[[Action], Action]] = (),
        guard: Callable[[Observation, Action], Action] | None = None,
        hooks: Sequence[Callable[[StepResult], None]] = (),
        rate_hz: float = 5.0,
        read_timeout_s: float = 1.0,
    ) -> None:
        if not math.isfinite(rate_hz) or rate_hz <= 0:
            raise ValueError("rate_hz phải hữu hạn và dương")
        if not math.isfinite(read_timeout_s) or read_timeout_s <= 0:
            raise ValueError("read_timeout_s phải hữu hạn và dương")
        self.source, self.policy, self.sink = source, policy, sink
        self.preprocess = tuple(preprocess)
        self.postprocess = tuple(postprocess)
        self.guard, self.hooks = guard, tuple(hooks)
        self.rate_hz, self.read_timeout_s = rate_hz, read_timeout_s
        self._opened: list[Module] = []
        self._active = False
        self._owner: int | None = None
        self._episode: Episode | None = None
        self._index = 0
        self._last_sequence: int | None = None
        self._stop = threading.Event()
        self._stepping = False
        self._log = logging.getLogger(__name__)

    def __enter__(self) -> "Pipeline":
        if self._active:
            raise RuntimeError("Pipeline đã được mở")
        self._active, self._owner = True, threading.get_ident()
        self._stop.clear()
        try:
            # Một object có thể được dùng cho nhiều vai trò; mở đúng một lần.
            for module in (self.source, self.policy, self.sink):
                if not any(module is opened for opened in self._opened):
                    self._opened.append(module)
                    module.open()
        except BaseException:
            self._shutdown("open_failed", suppress=True)
            raise
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        self._require_owner()
        self._shutdown("error" if exc_type else "complete", suppress=exc_type is not None)
        return False

    def _require_owner(self) -> None:
        if not self._active or threading.get_ident() != self._owner:
            raise RuntimeError("Pipeline phải được dùng trong thread sở hữu context")

    def _shutdown(self, reason: str, *, suppress: bool) -> None:
        self._stop.set()
        errors: list[BaseException] = []
        operations: list[Callable[[], None]] = []
        if any(self.sink is module for module in self._opened):
            operations.append(lambda: self.sink.halt(reason))
        operations.extend(module.close for module in reversed(self._opened))
        for operation in operations:
            try:
                operation()
            except BaseException as error:
                errors.append(error)
                self._log.exception("Pipeline cleanup failed")
        self._opened.clear()
        self._active, self._owner, self._episode = False, None, None
        if errors and not suppress:
            raise errors[0]

    def _abort(self, reason: str) -> None:
        self._stop.set()
        try:
            self.sink.halt(reason)
        except BaseException:
            self._log.exception("Sink.halt failed while preserving original error")

    def reset(self, instruction: str) -> Episode:
        self._require_owner()
        if self._stepping:
            raise RuntimeError("Không thể reset trong step()")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction phải là chuỗi không rỗng")
        episode = Episode(uuid4().hex, instruction)
        self._episode = None
        self._stop.clear()
        try:
            # Sink.halt mặc định chỉ giữ/hủy lệnh; không tự coi đây là emergency stop.
            self.sink.halt("episode_reset")
            for module in self._opened:
                module.reset(episode)
        except BaseException:
            self._abort("reset_failed")
            raise
        self._episode, self._index, self._last_sequence = episode, 0, None
        return episode

    def stop(self) -> None:
        """Yêu cầu dừng hợp tác; không ngắt inference đang bị treo."""
        self._stop.set()

    def _check_stop(self) -> None:
        if self._stop.is_set():
            raise PipelineStopped("Đã yêu cầu dừng pipeline")

    @staticmethod
    def _check_fresh(observation: Observation) -> None:
        if not math.isfinite(observation.valid_until):
            raise ValueError("Observation deadline phải hữu hạn")
        if time.monotonic() >= observation.valid_until:
            raise TimeoutError("Observation đã hết hạn; không gửi lệnh")

    def _check_action(self, action: Action) -> None:
        if action.spec != self.sink.spec:
            raise ValueError("ActionSpec không khớp với Sink")
        if len(action.values) != len(self.sink.spec.axes):
            raise ValueError("Số chiều action không khớp Sink")
        if not all(math.isfinite(float(value)) for value in action.values):
            raise ValueError("Action chứa NaN hoặc infinity")

    def step(self) -> StepResult:
        self._require_owner()
        if self._episode is None:
            raise RuntimeError("Gọi reset(instruction) trước step()")
        if self._stepping:
            raise RuntimeError("step() không được gọi lồng nhau")
        self._stepping = True
        started = time.monotonic()
        try:
            self._check_stop()
            observation = self.source.read(self.read_timeout_s)
            self._check_stop()
            self._check_fresh(observation)
            if self._last_sequence is not None and observation.sequence <= self._last_sequence:
                raise ValueError("Source phải trả snapshot mới ở mỗi step")
            self._last_sequence = observation.sequence
            processed = observation
            for transform in self.preprocess:
                processed = transform(processed)
            self._check_stop()
            self._check_fresh(observation)
            action = self.policy.predict(processed, self._episode.instruction)
            for transform in self.postprocess:
                action = transform(action)
            if self.guard is not None:
                action = self.guard(observation, action)
            self._check_action(action)
            self._check_fresh(observation)  # preprocessing không kéo dài deadline
            self._check_stop()
            self.sink.write(action)
            result = StepResult(
                self._episode, self._index, observation, action,
                time.monotonic() - started,
            )
            self._index += 1
            for hook in self.hooks:
                hook(result)
            return result
        except BaseException:
            self._abort("step_failed")
            raise
        finally:
            self._stepping = False

    def run(self, instruction: str, *, max_steps: int | None = None) -> int:
        if max_steps is not None and (
            isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 0
        ):
            raise ValueError("max_steps phải là số nguyên không âm hoặc None")
        count = 0
        with self:
            self.reset(instruction)
            while not self._stop.is_set() and (max_steps is None or count < max_steps):
                started = time.monotonic()
                try:
                    self.step()
                except PipelineStopped:
                    if not self._stop.is_set():
                        raise
                    break
                count += 1
                if max_steps is not None and count >= max_steps:
                    break
                remaining = max(0.0, 1.0 / self.rate_hz - (time.monotonic() - started))
                self._stop.wait(remaining)
        return count


__all__ = ["Pipeline", "PipelineStopped"]

"""Bọc ``program.model.base.Model`` thành Policy của pipeline."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from ..interfaces import Policy
from ..types import Action, ActionSpec, Episode, Observation


def default_input_builder(observation: Observation, instruction: str) -> dict[str, Any]:
    """Payload ổn định, model có thể dùng trực tiếp hoặc thay bằng builder riêng."""

    return {
        "images": observation.images,
        "state": observation.state,
        "stamps_ns": observation.stamps_ns,
        "instruction": instruction,
        "observation": observation,
    }


class ModelPolicy(Policy):
    """Adapter cho object có ``inference(input)`` hoặc callable.

    ``output_parser`` nhận kết quả model và trả ``Action``. Parser mặc định chấp
    nhận Action sẵn có, mapping có khóa ``values`` hoặc một sequence số.
    """

    def __init__(
        self,
        model: Any,
        spec: ActionSpec,
        *,
        input_builder: Callable[[Observation, str], Any] = default_input_builder,
        output_parser: Callable[[Any], Action] | None = None,
    ) -> None:
        self.model = model
        self.spec = spec
        self.input_builder = input_builder
        self.output_parser = output_parser or self._parse_output

    def open(self) -> None:
        opener = getattr(self.model, "open", None)
        if callable(opener):
            opener()

    def reset(self, episode: Episode) -> None:
        resetter = getattr(self.model, "reset", None)
        if callable(resetter):
            resetter(episode)

    def close(self) -> None:
        closer = getattr(self.model, "close", None)
        if callable(closer):
            closer()

    def predict(self, observation: Observation, instruction: str) -> Action:
        payload = self.input_builder(observation, instruction)
        infer = getattr(self.model, "inference", None)
        result = infer(payload) if callable(infer) else self.model(payload)
        return self.output_parser(result)

    def _parse_output(self, result: Any) -> Action:
        if isinstance(result, Action):
            return result
        if isinstance(result, Mapping):
            if "values" not in result:
                raise TypeError("Model output mapping phải có khóa 'values'")
            values = result["values"]
        else:
            values = result
        try:
            return Action(tuple(float(value) for value in values), self.spec)
        except (TypeError, ValueError) as exc:
            raise TypeError("Model output phải là Action hoặc sequence số") from exc


__all__ = ["ModelPolicy", "default_input_builder"]

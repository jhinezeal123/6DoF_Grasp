"""Guard: chan lenh truoc khi ra tay that.

Cam vao khe ``guard`` cua Pipeline. Pipeline goi guard SAU postprocess va TRUOC
sink.write, va runner con tu kiem tra ActionSpec + NaN sau do nua. Guard o day
chi lo nhung thu runner khong biet: vung lam viec cua robot that, san do cao,
va muc do nhay cua tung buoc.

Guard KHONG duoc phep sua am tham roi di tiep neu viec sua do lam sai y do cua
policy. O day no chi cat (clamp) vao hop cho phep va ghi log - cat nho thi vo
hai, con day lui han thi nem loi de khong ai doan mo.
"""

from __future__ import annotations

import numpy as np

from m750.pipeline.types import Action, ActionSpec, Observation


class GuardError(RuntimeError):
    """Lenh vi pham gioi han an toan."""


class WorkspaceGuard:
    """Gioi han hop lam viec, san do cao, va do nhay moi buoc."""

    def __init__(
        self,
        spec: ActionSpec,
        *,
        origin,
        reach_m: float = 0.35,
        z_floor_m: float = 0.02,
        z_ceiling_m: float = 0.60,
        max_jump_m: float = 0.05,
        log=print,
    ) -> None:
        self.spec = spec
        self.origin = np.asarray(origin, dtype=np.float64).reshape(3)
        self.reach_m = float(reach_m)
        self.z_floor_m = float(z_floor_m)
        self.z_ceiling_m = float(z_ceiling_m)
        self.max_jump_m = float(max_jump_m)
        self.log = log
        self._previous: np.ndarray | None = None

    def reset_hint(self) -> None:
        """Quen vi tri truoc, dung khi bat dau episode moi."""
        self._previous = None

    def __call__(self, observation: Observation, action: Action) -> Action:
        if action.spec != self.spec:
            raise GuardError("ActionSpec khong khop guard")
        if len(action.values) != len(self.spec.axes):
            raise GuardError("so chieu action khong khop")

        values = np.asarray(action.values, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise GuardError("action chua NaN/inf")

        position = values[:3].copy()

        # 1. Hop lam viec quanh tu the xuat phat.
        offset = position - self.origin
        horizontal = float(np.linalg.norm(offset[:2]))
        if horizontal > self.reach_m:
            raise GuardError(
                "dich ra ngoai tam voi: lech %.0f mm > %.0f mm"
                % (horizontal * 1000, self.reach_m * 1000)
            )

        # 2. San va tran do cao. Cat mem: ha xuong dung san la chuyen binh thuong,
        #    khong phai loi.
        if position[2] < self.z_floor_m:
            self.log("  [guard] cat z %.0f -> %.0f mm (san)"
                     % (position[2] * 1000, self.z_floor_m * 1000))
            position[2] = self.z_floor_m
        if position[2] > self.z_ceiling_m:
            raise GuardError("dich cao hon tran %.0f mm" % (self.z_ceiling_m * 1000))

        # 3. Do nhay moi buoc. Nhay qua lon nghia la model doc sai hoac J loan.
        if self._previous is not None:
            jump = float(np.linalg.norm(position - self._previous))
            if jump > self.max_jump_m:
                raise GuardError(
                    "buoc nhay %.0f mm > %.0f mm; dung lai thay vi ra lenh loan"
                    % (jump * 1000, self.max_jump_m * 1000)
                )
        self._previous = position.copy()

        return Action(values=tuple(float(v) for v in np.concatenate([position, values[3:]])),
                      spec=self.spec)


__all__ = ["WorkspaceGuard", "GuardError"]

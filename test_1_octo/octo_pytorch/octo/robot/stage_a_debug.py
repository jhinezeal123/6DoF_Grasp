from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence

import numpy as np


@dataclass
class StageADebugPreviewConfig:
    enabled_modalities: Sequence[str]
    show_labels: bool = True
    show_timestamp: bool = True
    max_height: int = 360
    window_name: str = "stage_a_collection_debug"
    terminal_abort_enabled: bool = True
    terminal_abort_commands: Sequence[str] = ("q", "quit", "abort", "stop", "exit")


class StageADebugPreview:
    def __init__(self, config: StageADebugPreviewConfig):
        self.config = config
        self._available = True
        self._initialized = False

    def _resize_keep_aspect(self, image_rgb: np.ndarray, target_h: int) -> np.ndarray:
        import cv2

        h, w = image_rgb.shape[:2]
        if h <= 0 or w <= 0 or h == target_h:
            return image_rgb
        scale = float(target_h) / float(h)
        new_w = max(1, int(round(w * scale)))
        return cv2.resize(image_rgb, (new_w, int(target_h)), interpolation=cv2.INTER_AREA)

    def _annotate(self, image_rgb: np.ndarray, *, label: str, meta: Optional[dict]) -> np.ndarray:
        import cv2

        frame = image_rgb.copy()
        y = 24
        if self.config.show_labels:
            cv2.putText(frame, label.upper(), (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            y += 24
        if self.config.show_timestamp and meta is not None:
            age_s = float(meta.get("age_s", np.nan))
            cv2.putText(frame, f"age={age_s:.3f}s", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
        return frame

    def _build_canvas(
        self,
        *,
        episode_id: str,
        step_idx: int,
        frames: Mapping[str, np.ndarray],
        frame_meta: Optional[Mapping[str, dict]],
        status_lines: Sequence[str],
    ) -> Optional[np.ndarray]:
        import cv2

        frame_items = []
        frame_meta = frame_meta or {}
        preferred = [m for m in self.config.enabled_modalities]
        if not preferred:
            preferred = [key.replace("image_", "") for key in sorted(frames.keys())]

        for name in preferred:
            key = f"image_{name}"
            if key not in frames:
                continue
            frame_items.append((name, self._annotate(frames[key], label=name, meta=frame_meta.get(name))))
        if not frame_items:
            return None

        target_h = max(120, int(self.config.max_height))
        resized_frames = [self._resize_keep_aspect(frame, target_h) for _, frame in frame_items]
        if len(resized_frames) == 1:
            canvas = resized_frames[0]
        else:
            separator = np.zeros((target_h, 8, 3), dtype=np.uint8)
            canvas = resized_frames[0]
            for frame in resized_frames[1:]:
                canvas = np.concatenate([canvas, separator, frame], axis=1)

        footer_h = 26 + 22 * max(1, len(status_lines) + 1)
        footer = np.zeros((footer_h, canvas.shape[1], 3), dtype=np.uint8)
        y = 22
        cv2.putText(
            footer,
            f"{episode_id} step={step_idx:04d} | window: q/ESC abort | terminal: type 'q'+Enter",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        for line in status_lines:
            y += 20
            cv2.putText(footer, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)
        return np.concatenate([canvas, footer], axis=0)

    def _terminal_abort_requested(self) -> bool:
        if not bool(getattr(self.config, "terminal_abort_enabled", False)):
            return False
        try:
            import select
            import sys
        except Exception:
            return False

        stdin = sys.stdin
        if stdin is None or not getattr(stdin, "isatty", lambda: False)():
            return False

        try:
            readable, _, _ = select.select([stdin], [], [], 0)
        except Exception:
            return False
        if not readable:
            return False

        try:
            line = stdin.readline()
        except Exception:
            return False
        if not line:
            return False

        cmd = line.strip().lower()
        if cmd == "":
            return False

        allowed = {str(x).strip().lower() for x in getattr(self.config, "terminal_abort_commands", ())}
        return cmd in allowed

    def show_step(
        self,
        *,
        episode_id: str,
        step_idx: int,
        frames: Mapping[str, np.ndarray],
        frame_meta: Optional[Mapping[str, dict]],
        status_lines: Sequence[str],
    ) -> bool:
        if self._terminal_abort_requested():
            return False
        if not self._available:
            return True
        try:
            import cv2
        except Exception:
            self._available = False
            return True

        try:
            canvas = self._build_canvas(
                episode_id=episode_id,
                step_idx=step_idx,
                frames=frames,
                frame_meta=frame_meta,
                status_lines=status_lines,
            )
            if canvas is None:
                return True
            if not self._initialized:
                try:
                    cv2.namedWindow(self.config.window_name, cv2.WINDOW_NORMAL)
                except Exception:
                    pass
                self._initialized = True
            cv2.imshow(self.config.window_name, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                return False
            return True
        except Exception:
            self._available = False
            return True

    def close(self) -> None:
        if not self._available:
            return
        try:
            import cv2

            cv2.destroyWindow(self.config.window_name)
            cv2.waitKey(1)
        except Exception:
            pass

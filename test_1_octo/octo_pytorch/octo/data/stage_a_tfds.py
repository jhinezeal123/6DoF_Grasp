from __future__ import annotations

from enum import Enum
import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import tensorflow_datasets as tfds

_DESCRIPTION = """Stage-A MyArm dataset generated from scripted absolute 7D waypoint rollouts.

The raw format is produced by scripts/collect_stage_a_dataset.py or scripts/collect_waypoint_demos.py.
Each episode stores RGB frames, achieved TCP state in SI units, and action labels derived from next-state deltas.
"""

_CITATION = """@misc{octo_pytorch_stage_a_2026,
  title={Stage-A MyArm absolute-waypoint dataset for Octo PyTorch finetuning},
  year={2026}
}"""


class StageAActionSource(str, Enum):
    ACHIEVED_DELTA = "achieved_delta"
    COMMANDED_DELTA = "commanded_delta"


class MyarmStageAConfig(tfds.core.BuilderConfig):
    def __init__(
        self,
        *,
        action_source: StageAActionSource,
        filter_noops: bool = False,
        binarize_gripper: bool = True,
        invert_gripper: bool = False,
        noop_pos_eps_m: float = 1e-6,
        noop_rot_eps_rad: float = 1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.action_source = StageAActionSource(action_source)
        self.filter_noops = bool(filter_noops)
        self.binarize_gripper = bool(binarize_gripper)
        self.invert_gripper = bool(invert_gripper)
        self.noop_pos_eps_m = float(noop_pos_eps_m)
        self.noop_rot_eps_rad = float(noop_rot_eps_rad)


class MyarmStageADataset(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("0.0.6")
    RELEASE_NOTES = {
        "0.0.1": "Initial release for Stage-A single-camera finetuning.",
        "0.0.2": "Use delta actions derived from absolute get_coords samples without re-differencing in the transform.",
        "0.0.3": "Fix TFDS feature schema for boolean step flags with explicit numpy boolean dtypes.",
        "0.0.4": "Optional wrist camera support and richer episode metadata.",
        "0.0.5": "Normalized episode_xxxxxx naming plus Stage-A v0.2.5 metadata/debug workflow.",
        "0.0.6": "Add builder configs for commanded vs achieved deltas and optional no-op filtering.",
    }
    MANUAL_DOWNLOAD_INSTRUCTIONS = "Place raw episodes under <manual_dir>/raw/episode_xxxxxx before running download_and_prepare()."

    BUILDER_CONFIGS = [
        MyarmStageAConfig(
            name="commanded",
            description="Uses commanded_action.npy (delta xyz/rpy + absolute gripper_open at next state).",
            action_source=StageAActionSource.COMMANDED_DELTA,
            filter_noops=False,
            binarize_gripper=True,
            invert_gripper=False,
        ),
        MyarmStageAConfig(
            name="commanded_invert_gripper",
            description="Uses commanded_action.npy and inverts gripper (raw open=0/close=1 -> canonical open=1/close=0).",
            action_source=StageAActionSource.COMMANDED_DELTA,
            filter_noops=False,
            binarize_gripper=True,
            invert_gripper=True,
        ),
        MyarmStageAConfig(
            name="commanded_no_noops",
            description="Uses commanded_action.npy and filters delta-action no-ops (keeps gripper changes).",
            action_source=StageAActionSource.COMMANDED_DELTA,
            filter_noops=True,
            binarize_gripper=True,
            invert_gripper=False,
        ),
        MyarmStageAConfig(
            name="commanded_invert_gripper_no_noops",
            description="Uses commanded_action.npy, inverts gripper, and filters delta-action no-ops (keeps gripper changes).",
            action_source=StageAActionSource.COMMANDED_DELTA,
            filter_noops=True,
            binarize_gripper=True,
            invert_gripper=True,
        ),
        MyarmStageAConfig(
            name="achieved",
            description="Uses achieved delta actions (delta_action_from_get_coords.npy / action.npy).",
            action_source=StageAActionSource.ACHIEVED_DELTA,
            filter_noops=False,
            binarize_gripper=True,
            invert_gripper=False,
        ),
        MyarmStageAConfig(
            name="achieved_no_noops",
            description="Uses achieved delta actions and filters delta-action no-ops (keeps gripper changes).",
            action_source=StageAActionSource.ACHIEVED_DELTA,
            filter_noops=True,
            binarize_gripper=True,
            invert_gripper=False,
        ),
    ]
    DEFAULT_CONFIG_NAME = "commanded"

    def _info(self) -> tfds.core.DatasetInfo:
        return self.dataset_info_from_configs(
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": {
                                "image_primary": tfds.features.Image(shape=(None, None, 3), encoding_format="jpeg"),
                                "image_wrist": tfds.features.Image(shape=(None, None, 3), encoding_format="jpeg"),
                                "state": tfds.features.Tensor(shape=(7,), dtype=np.float32),
                                "proprio": tfds.features.Tensor(shape=(7,), dtype=np.float32),
                                "commanded_state": tfds.features.Tensor(shape=(7,), dtype=np.float32),
                            },
                            "action": tfds.features.Tensor(shape=(7,), dtype=np.float32),
                            "discount": np.float32,
                            "reward": np.float32,
                            "is_first": np.bool_,
                            "is_last": np.bool_,
                            "is_terminal": np.bool_,
                            "language_instruction": tfds.features.Text(),
                            "timestamp": np.float32,
                        }
                    ),
                    "episode_metadata": {
                        "episode_id": tfds.features.Text(),
                        "instruction": tfds.features.Text(),
                        "num_steps": np.int32,
                        "source_format": tfds.features.Text(),
                        "run_kind": tfds.features.Text(),
                    },
                }
            ),
            supervised_keys=None,
            homepage="https://github.com/octo-models/octo",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        manual_dir = Path(dl_manager.manual_dir)
        raw_dir = manual_dir / "raw"
        if not raw_dir.exists():
            raise FileNotFoundError(
                f"Expected raw episodes under {raw_dir}. Run scripts/collect_stage_a_dataset.py or collect_waypoint_demos.py first."
            )
        episodes = sorted(p for p in raw_dir.iterdir() if p.is_dir() and p.name.startswith("episode_"))
        if not episodes:
            raise FileNotFoundError(f"No raw episodes found in {raw_dir}")
        return {"train": self._generate_examples(episodes)}

    def _generate_examples(self, episodes: Iterable[Path]):
        for episode_dir in episodes:
            episode_id = episode_dir.name
            yield episode_id, self._build_episode_example(episode_dir)

    @staticmethod
    def _optional_frame_path(frame_dir: Path, idx: int, fallback: Optional[Path]) -> Path:
        frame_path = frame_dir / f"{idx:06d}.jpg"
        if frame_path.exists():
            return frame_path
        if fallback is not None and fallback.exists():
            return fallback
        raise FileNotFoundError(f"Missing frame {frame_path}")

    def _build_episode_example(self, episode_dir: Path):
        meta = json.loads((episode_dir / "meta.json").read_text(encoding="utf-8"))
        instruction = (episode_dir / "instruction.txt").read_text(encoding="utf-8").strip()

        achieved_state = np.load(episode_dir / "achieved_state.npy").astype(np.float32)
        commanded_state = np.load(episode_dir / "commanded_state.npy").astype(np.float32)
        cfg = self.builder_config
        if not isinstance(cfg, MyarmStageAConfig):
            raise TypeError(f"Unexpected builder_config type: {type(cfg)}")

        if cfg.action_source == StageAActionSource.COMMANDED_DELTA:
            action_path = episode_dir / "commanded_action.npy"
            if not action_path.exists():
                raise FileNotFoundError(
                    f"Missing {action_path}. This raw episode was likely recorded with an older collector that did not save commanded_action.npy."
                )
        else:
            action_path = episode_dir / "delta_action_from_get_coords.npy"
            if not action_path.exists():
                action_path = episode_dir / "action.npy"
        action = np.load(action_path).astype(np.float32)
        timestamps = np.load(episode_dir / "timestamps.npy").astype(np.float32)

        primary_dir = episode_dir / "frames" / "image_primary"
        wrist_dir = episode_dir / "frames" / "image_wrist"
        primary_paths = sorted(primary_dir.glob("*.jpg"))
        wrist_paths = sorted(wrist_dir.glob("*.jpg")) if wrist_dir.exists() else []

        if action.ndim != 2 or action.shape[-1] != 7:
            raise ValueError(f"Expected action shape (T, 7), got {action.shape} in {episode_dir}")
        if achieved_state.ndim != 2 or achieved_state.shape[-1] != 7:
            raise ValueError(f"Expected achieved_state shape (T, 7), got {achieved_state.shape} in {episode_dir}")
        if commanded_state.ndim != 2 or commanded_state.shape[-1] != 7:
            raise ValueError(f"Expected commanded_state shape (T, 7), got {commanded_state.shape} in {episode_dir}")

        max_state_steps = min(len(primary_paths), len(timestamps), len(achieved_state), len(commanded_state))
        usable_len = min(len(action), max_state_steps - 1)
        if usable_len <= 0:
            raise ValueError(f"Episode {episode_dir} is empty after alignment")

        # Canonicalize gripper to open=1 / closed=0, optionally inverting and binarizing.
        action = action.copy()
        action[:, 6] = np.clip(action[:, 6], 0.0, 1.0)
        if cfg.invert_gripper:
            action[:, 6] = 1.0 - action[:, 6]
        if cfg.binarize_gripper and len(action) > 0:
            action[:, 6] = self._binarize_gripper_np(action[:, 6])

        keep_indices = np.arange(usable_len, dtype=np.int64)
        if cfg.filter_noops and usable_len > 0:
            prev_gripper_open = float(
                commanded_state[0, 6]
                if cfg.action_source == StageAActionSource.COMMANDED_DELTA
                else achieved_state[0, 6]
            )
            prev_gripper_open = float(np.clip(prev_gripper_open, 0.0, 1.0))
            if cfg.invert_gripper:
                prev_gripper_open = 1.0 - prev_gripper_open
            if cfg.binarize_gripper:
                prev_gripper_open = float(
                    self._binarize_gripper_np(
                        np.asarray([prev_gripper_open], dtype=np.float32)
                    )[0]
                )
            keep_mask = self._keep_non_noop_indices(
                action=action[:usable_len],
                prev_gripper_open=prev_gripper_open,
                pos_eps_m=cfg.noop_pos_eps_m,
                rot_eps_rad=cfg.noop_rot_eps_rad,
            )
            keep_indices = keep_indices[keep_mask]
            if keep_indices.size == 0:
                raise ValueError(f"Episode {episode_dir} has zero steps after no-op filtering")

        steps = []
        for out_idx, i in enumerate(keep_indices.tolist()):
            primary_frame = primary_paths[i]
            wrist_frame = wrist_paths[i] if i < len(wrist_paths) else primary_frame
            steps.append(
                {
                    "observation": {
                        "image_primary": primary_frame,
                        "image_wrist": wrist_frame,
                        "state": achieved_state[i],
                        "proprio": achieved_state[i],
                        "commanded_state": commanded_state[i],
                    },
                    "action": action[i],
                    "discount": np.float32(1.0),
                    "reward": np.float32(0.0),
                    "is_first": np.bool_(out_idx == 0),
                    "is_last": np.bool_(out_idx == len(keep_indices) - 1),
                    "is_terminal": np.bool_(out_idx == len(keep_indices) - 1),
                    "language_instruction": instruction,
                    "timestamp": np.float32(timestamps[i]),
                }
            )

        return {
            "steps": steps,
            "episode_metadata": {
                "episode_id": episode_dir.name,
                "instruction": instruction,
                "num_steps": np.int32(len(keep_indices)),
                "source_format": str(meta.get("format_version", "stage_a_raw_v0_0_4")),
                "run_kind": str(meta.get("run_kind", "record")),
            },
        }

    @staticmethod
    def _binarize_gripper_np(actions_1d: np.ndarray) -> np.ndarray:
        actions_1d = np.asarray(actions_1d, dtype=np.float32)
        if actions_1d.ndim != 1:
            raise ValueError(f"Expected 1D gripper action array, got shape {actions_1d.shape}")
        if actions_1d.size == 0:
            return actions_1d

        open_mask = actions_1d > 0.95
        closed_mask = actions_1d < 0.05
        in_between_mask = ~(open_mask | closed_mask)

        out = np.empty_like(actions_1d, dtype=np.float32)
        carry = float(actions_1d[-1])
        for i in range(actions_1d.size - 1, -1, -1):
            if in_between_mask[i]:
                out[i] = carry
            else:
                carry = 1.0 if bool(open_mask[i]) else 0.0
                out[i] = carry
        return out

    @staticmethod
    def _keep_non_noop_indices(
        *,
        action: np.ndarray,
        prev_gripper_open: float,
        pos_eps_m: float,
        rot_eps_rad: float,
        gripper_change_eps: float = 1e-6,
    ) -> np.ndarray:
        """Returns a boolean mask selecting non-noop steps.

        No-op is defined as: delta xyz/rpy is (near) zero AND gripper does not change.
        The gripper channel is treated as an absolute "open fraction" value (open=1, closed=0).
        """
        if action.ndim != 2 or action.shape[-1] != 7:
            raise ValueError(f"Expected action shape (T, 7), got {action.shape}")

        delta_pos = action[:, :3]
        delta_rot = action[:, 3:6]
        move_is_noop = np.all(np.abs(delta_pos) <= pos_eps_m, axis=1) & np.all(
            np.abs(delta_rot) <= rot_eps_rad, axis=1
        )

        gripper_next = action[:, 6]
        gripper_prev = np.concatenate(
            [np.asarray([prev_gripper_open], dtype=np.float32), gripper_next[:-1]],
            axis=0,
        )
        gripper_changed = np.abs(gripper_next - gripper_prev) > gripper_change_eps

        is_noop = move_is_noop & (~gripper_changed)
        return ~is_noop

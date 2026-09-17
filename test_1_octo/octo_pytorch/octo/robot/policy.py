from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from octo.model.octo_model_pt import OctoModelPt
from octo.utils.train_utils_pt import _np2pt, _to_device


class OctoTorchPolicy:
    def __init__(
        self,
        model: OctoModelPt,
        device: torch.device,
        dataset_key: Optional[str] = None,
        fp16: bool = False,
        seed: int = 0,
    ):
        self.model = model.to(device)
        if not fp16:
            # Keep full FP32 weights when running without autocast.
            # (Some checkpoints / environments may load weights in half/bfloat16.)
            self.model = self.model.to(dtype=torch.float32)
        self.model.eval()
        self.device = device
        self.dataset_key = dataset_key
        self.fp16 = fp16
        self.generator = torch.Generator(device=device).manual_seed(seed)

        self._obs_example = self.model.example_batch["observation"]
        self._task_example = self.model.example_batch["task"]

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str,
        device: str = "cuda",
        dataset_key: Optional[str] = None,
        fp16: bool = False,
        seed: int = 0,
    ) -> "OctoTorchPolicy":
        torch_device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        load_dict = OctoModelPt.load_pretrained(Path(checkpoint_dir))
        model = load_dict["octo_model"]
        return cls(model=model, device=torch_device, dataset_key=dataset_key, fp16=fp16, seed=seed)

    def expected_image_sizes(self) -> Dict[str, tuple]:
        sizes = {}
        for key, value in self._obs_example.items():
            if not isinstance(value, torch.Tensor):
                continue
            if key.startswith("image_") and value.ndim == 5:
                _, _, _, h, w = value.shape
                sizes[key.replace("image_", "")] = (int(h), int(w))
        return sizes

    def model_action_horizon(self) -> int:
        return int(getattr(self.model.module.heads["action"], "action_horizon", 1))

    def expected_goal_image_sizes(self) -> Dict[str, tuple]:
        sizes = {}
        for key, value in self._task_example.items():
            if not isinstance(value, torch.Tensor):
                continue
            if key.startswith("image_") and value.ndim == 4:
                _, _, h, w = value.shape
                sizes[key] = (int(h), int(w))
        return sizes

    def create_task(
        self,
        text: Optional[str] = None,
        goal_images: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict:
        goals = None
        if goal_images:
            goals = {k: np.stack([v], axis=0) for k, v in goal_images.items()}
        texts = [text] if text is not None else None
        return self.model.create_tasks(goals=goals, texts=texts, device=self.device)

    def _fill_observation_like_example(self, obs: Dict[str, np.ndarray]) -> Dict:
        filled = {}
        pad_mask_dict = {}
        expected_pad_mask_keys = None
        try:
            ex_pad_mask = self._obs_example.get("pad_mask_dict")
            if isinstance(ex_pad_mask, dict):
                expected_pad_mask_keys = set(ex_pad_mask.keys())
        except Exception:
            expected_pad_mask_keys = None
        time_mask = None
        time_dim = None

        if "timestep_pad_mask" in obs:
            time_mask = np.asarray(obs["timestep_pad_mask"]).astype(np.bool_)
            time_dim = int(time_mask.shape[0])

        for key, ex_val in self._obs_example.items():
            if key == "pad_mask_dict":
                continue
            if key == "timestep_pad_mask":
                continue

            if isinstance(ex_val, torch.Tensor):
                if key in obs:
                    arr = np.asarray(obs[key])
                    filled[key] = arr
                    if expected_pad_mask_keys is None or key in expected_pad_mask_keys:
                        pad_mask_dict[key] = np.ones((arr.shape[0],), dtype=np.bool_)
                    if time_dim is None:
                        time_dim = int(arr.shape[0])
                else:
                    if time_dim is None:
                        time_dim = int(ex_val.shape[1])
                    ex_dtype = np.uint8 if ex_val.dtype == torch.uint8 else np.float32
                    if key.startswith("image_") and ex_val.ndim == 5:
                        _, _, _, h, w = ex_val.shape
                        filled[key] = np.zeros((time_dim, h, w, 3), dtype=ex_dtype)
                    else:
                        filled[key] = np.zeros((time_dim, *tuple(ex_val.shape[2:])), dtype=ex_dtype)
                    if expected_pad_mask_keys is None or key in expected_pad_mask_keys:
                        pad_mask_dict[key] = np.zeros((time_dim,), dtype=np.bool_)

        if time_mask is None:
            if time_dim is None:
                raise ValueError("Unable to infer timestep dimension for observation")
            time_mask = np.ones((time_dim,), dtype=np.bool_)
        filled["timestep_pad_mask"] = time_mask
        if expected_pad_mask_keys is not None:
            # Ensure we match checkpoint's expected pad_mask_dict keys exactly.
            pad_mask_dict = {k: pad_mask_dict.get(k, np.zeros((time_dim,), dtype=np.bool_)) for k in expected_pad_mask_keys}
        filled["pad_mask_dict"] = pad_mask_dict
        return filled

    def prepare_observation(self, obs: Dict[str, np.ndarray]) -> Dict:
        filled = self._fill_observation_like_example(obs)
        obs_pt = _np2pt({k: v for k, v in filled.items() if k != "pad_mask_dict"}, dtype=None)
        pad_mask_pt = {
            k: torch.tensor(v[None], dtype=torch.bool, device=self.device)
            for k, v in filled["pad_mask_dict"].items()
        }
        obs_pt = {k: v[None].to(self.device) if isinstance(v, torch.Tensor) else v for k, v in obs_pt.items()}
        obs_pt["pad_mask_dict"] = pad_mask_pt
        obs_pt = _to_device(obs_pt, self.device)
        return obs_pt

    def sample_action_chunk(self, obs: Dict[str, np.ndarray], task: Dict) -> np.ndarray:
        model_obs = self.prepare_observation(obs)
        stats = None
        if self.dataset_key is not None:
            if self.dataset_key not in self.model.dataset_statistics:
                raise KeyError(
                    f"Unknown dataset_key '{self.dataset_key}'. Available keys: {list(self.model.dataset_statistics.keys())}"
                )
            stats = self.model.dataset_statistics[self.dataset_key]["action"]

        with torch.inference_mode():
            if self.fp16 and self.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    action = self.model.sample_actions(
                        model_obs,
                        task,
                        unnormalization_statistics=stats,
                        generator=self.generator,
                    )
            else:
                action = self.model.sample_actions(
                    model_obs,
                    task,
                    unnormalization_statistics=stats,
                    generator=self.generator,
                )
        return action[0].detach().cpu().numpy()

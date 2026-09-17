from __future__ import annotations
from collections import deque
import logging
from typing import Any, Dict, Optional, Sequence, Tuple
import gymnasium as gym
import numpy as np
import torch

# filepath: /home/ktmt/khoanhd/Octo_Lab/octo-pytorch-infer/octo/utils/gym_wrappers_pt.py


import torch.nn.functional as F


class OctoObsPreprocessWrapper(gym.ObservationWrapper):
    """Best-effort Octo-style observation preprocessing.

    Supports legacy/raw observation dicts like:
      - obs['image']: flat (3*im*im,) or CHW (3,im,im) float in [0,1]
      - obs['state']: vector where the last element is gripper open

    Produces/overwrites:
      - obs['image_primary']: HWC uint8 in [0,255]
      - obs['proprio']: optionally padded to 8 dims: [state[:6], 0, state[-1]]

    If the incoming observation already contains keys like 'image_primary', this wrapper
    will leave them as-is unless a conversion is explicitly possible/needed.
    """

    def __init__(
        self,
        env: gym.Env,
        im_size: Optional[int] = None,
        pad_proprio_to_8: bool = False,
        drop_raw_keys: bool = True,
    ):
        super().__init__(env)
        self.im_size = int(im_size) if im_size is not None else None
        self.pad_proprio_to_8 = bool(pad_proprio_to_8)
        self.drop_raw_keys = bool(drop_raw_keys)

        if isinstance(self.observation_space, gym.spaces.Dict):
            spaces = dict(self.observation_space.spaces)
            if "image_primary" not in spaces and self.im_size is not None:
                spaces["image_primary"] = gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(self.im_size, self.im_size, 3),
                    dtype=np.uint8,
                )
            if "proprio" in spaces and self.pad_proprio_to_8:
                # Only adjust if it looks like 7D proprio.
                sp = spaces["proprio"]
                if isinstance(sp, gym.spaces.Box) and tuple(getattr(sp, "shape", ())) == (7,):
                    spaces["proprio"] = gym.spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(8,),
                        dtype=np.float32,
                    )
            self.observation_space = gym.spaces.Dict(spaces)

    @staticmethod
    def _to_hwc_uint8_from_chw_or_flat(image: np.ndarray, im_size: int) -> np.ndarray:
        arr = np.asarray(image)
        if arr.ndim == 1:
            expected = 3 * im_size * im_size
            if arr.size != expected:
                raise ValueError(f"Expected flat image of size {expected}, got {arr.size}")
            arr = arr.reshape(3, im_size, im_size)
        if arr.ndim == 3 and arr.shape[0] == 3:
            arr = arr.transpose(1, 2, 0)

        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Expected image as HWC/CHW with 3 channels, got {arr.shape}")

        if arr.dtype != np.uint8:
            arr_f = arr.astype(np.float32, copy=False)
            if float(np.nanmax(arr_f)) <= 1.0 + 1e-6:
                arr_f = arr_f * 255.0
            arr = np.clip(np.round(arr_f), 0, 255).astype(np.uint8)
        return arr

    def observation(self, observation: Dict[str, Any]):
        obs = dict(observation)

        # Image conversion: obs['image'] -> obs['image_primary']
        if "image_primary" not in obs and "image" in obs:
            image = np.asarray(obs["image"])
            im_size = self.im_size
            if im_size is None and image.ndim == 1:
                # Infer square size from flat vector.
                size = int(round(np.sqrt(float(image.size) / 3.0)))
                if 3 * size * size == int(image.size):
                    im_size = size
            if im_size is None:
                raise ValueError(
                    "Cannot convert obs['image'] without im_size; pass im_size to OctoObsPreprocessWrapper."
                )
            obs["image_primary"] = self._to_hwc_uint8_from_chw_or_flat(image, im_size)

        # Proprio conversion/padding.
        if "proprio" not in obs and "state" in obs:
            state = np.asarray(obs["state"]).astype(np.float32, copy=False)
            if self.pad_proprio_to_8:
                if state.shape[0] < 7:
                    raise ValueError(f"Expected state with at least 7 dims, got {state.shape}")
                obs["proprio"] = np.concatenate(
                    [state[:6], np.asarray([0.0], dtype=np.float32), state[-1:]], axis=0
                )
            else:
                obs["proprio"] = state
        elif "proprio" in obs and self.pad_proprio_to_8:
            proprio = np.asarray(obs["proprio"]).astype(np.float32, copy=False)
            if proprio.shape[0] == 7:
                obs["proprio"] = np.concatenate(
                    [proprio[:6], np.asarray([0.0], dtype=np.float32), proprio[-1:]], axis=0
                )

        if self.drop_raw_keys:
            obs.pop("image", None)
            obs.pop("state", None)

        return obs


def stack_and_pad(history: deque, num_obs: int) -> Dict[str, np.ndarray]:
    """
    Stack a deque of dict observations into a single dict with leading time dim.
    Adds `timestep_pad_mask` (1=real, 0=padding).
    """
    horizon = len(history)
    full_obs = {k: np.stack([dic[k] for dic in history]) for k in history[0]}
    pad_length = horizon - min(num_obs, horizon)
    timestep_pad_mask = np.ones(horizon, dtype=np.float32)
    timestep_pad_mask[:pad_length] = 0.0
    full_obs["timestep_pad_mask"] = timestep_pad_mask
    return full_obs


def space_stack(space: gym.Space, repeat: int) -> gym.Space:
    """Create a new Gym space that repeats the original space `repeat` times."""
    if isinstance(space, gym.spaces.Box):
        low = np.repeat(space.low[None], repeat, axis=0)
        high = np.repeat(space.high[None], repeat, axis=0)
        return gym.spaces.Box(low=low, high=high, dtype=space.dtype)
    if isinstance(space, gym.spaces.Discrete):
        return gym.spaces.MultiDiscrete([space.n] * repeat)
    if isinstance(space, gym.spaces.Dict):
        return gym.spaces.Dict({k: space_stack(v, repeat) for k, v in space.spaces.items()})
    raise ValueError(f"Space {space} is not supported by Octo Gym wrappers.")


def listdict2dictlist(LD: Sequence[Dict[str, Any]]) -> Dict[str, list]:
    if len(LD) == 0:
        return {}
    return {k: [dic[k] for dic in LD] for k in LD[0]}


def add_octo_env_wrappers(
    env: gym.Env,
    action_proprio_metadata: dict,
    horizon: int,
    exec_horizon: int,
    resize_size: Optional[Dict[str, Tuple[int, int]]] = None,
    use_temp_ensembling: bool = True,
):
    """
    Adds wrappers for proprio normalization (PyTorch/NumPy), image resizing (PyTorch),
    history stacking, and action execution (RHC or temporal ensembling).
    """
    env = NormalizeProprio(env, action_proprio_metadata)
    env = ResizeImageWrapperPT(env, resize_size)
    env = HistoryWrapper(env, horizon)
    env = TemporalEnsembleWrapper(env, exec_horizon) if use_temp_ensembling else RHCWrapper(env, exec_horizon)
    return env


class HistoryWrapper(gym.Wrapper):
    """Stacks the last `horizon` observations and pads early timesteps with a mask."""

    def __init__(self, env: gym.Env, horizon: int):
        super().__init__(env)
        self.horizon = int(horizon)
        self.history = deque(maxlen=self.horizon)
        self.num_obs = 0
        self.observation_space = space_stack(self.env.observation_space, self.horizon)

    def step(self, action):
        obs, reward, done, trunc, info = self.env.step(action)
        self.num_obs += 1
        self.history.append(obs)
        assert len(self.history) == self.horizon
        full_obs = stack_and_pad(self.history, self.num_obs)
        return full_obs, reward, done, trunc, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.num_obs = 1
        self.history.clear()
        self.history.extend([obs] * self.horizon)
        full_obs = stack_and_pad(self.history, self.num_obs)
        return full_obs, info


class RHCWrapper(gym.Wrapper):
    """Receding horizon control: policy outputs a sequence, execute first `exec_horizon` actions."""

    def __init__(self, env: gym.Env, exec_horizon: int):
        super().__init__(env)
        self.exec_horizon = int(exec_horizon)

    def step(self, actions: np.ndarray):
        if self.exec_horizon == 1 and getattr(actions, "ndim", 0) == 1:
            actions = actions[None]
        assert len(actions) >= self.exec_horizon

        rewards, observations, infos = [], [], []
        done = trunc = False

        for i in range(self.exec_horizon):
            obs, reward, done, trunc, info = self.env.step(actions[i])
            observations.append(obs)
            rewards.append(reward)
            infos.append(info)
            if done or trunc:
                break

        out_info = listdict2dictlist(infos)
        out_info["rewards"] = rewards
        out_info["observations"] = observations
        return obs, float(np.sum(rewards)), done, trunc, out_info


class TemporalEnsembleWrapper(gym.Wrapper):
    """
    Temporal ensembling (as in https://arxiv.org/abs/2304.13705).
    Keeps last `pred_horizon` predicted chunks and averages the current-timestep action.
    """

    def __init__(self, env: gym.Env, pred_horizon: int, exp_weight: float = 0.0):
        super().__init__(env)
        self.pred_horizon = int(pred_horizon)
        self.exp_weight = float(exp_weight)
        self.act_history = deque(maxlen=self.pred_horizon)
        self.action_space = space_stack(self.env.action_space, self.pred_horizon)

    def step(self, actions: np.ndarray):
        assert len(actions) >= self.pred_horizon
             
        
        self.act_history.append(actions[: self.pred_horizon])
        num_actions = len(self.act_history)

        curr_act_preds = np.stack(
            [pred_actions[i] for (i, pred_actions) in zip(range(num_actions - 1, -1, -1), self.act_history)]
        )

        weights = np.exp(-self.exp_weight * np.arange(num_actions, dtype=np.float32))
        weights = weights / (weights.sum() + 1e-8)
        action = np.sum(weights[:, None] * curr_act_preds, axis=0)

        return self.env.step(action)

    def reset(self, **kwargs):
        self.act_history = deque(maxlen=self.pred_horizon)
        return self.env.reset(**kwargs)


class ResizeImageWrapperPT(gym.ObservationWrapper):
    """
    Resize image observations using PyTorch only (no TensorFlow/JAX).

    Notes:
      - TF "lanczos3" is approximated with PyTorch bicubic + antialias.
      - "Average augmentation" is implemented as a center crop with (avg_scale, avg_ratio),
        then resized to target size.
    """

    def __init__(
        self,
        env: gym.Env,
        resize_size: Optional[Dict[str, Tuple[int, int]]] = None,
        augmented_keys: Sequence[str] = ("image_primary",),
        avg_scale: float = 0.9,
        avg_ratio: float = 1.0,
        device: Optional[str] = None,
    ):
        super().__init__(env)
        if not isinstance(self.observation_space, gym.spaces.Dict):
            raise AssertionError("Only Dict observation spaces are supported.")

        self.resize_size = resize_size
        self.augmented_keys = tuple(augmented_keys)
        self.avg_scale = float(avg_scale)
        self.avg_ratio = float(avg_ratio)
        self.device = torch.device(device) if device is not None else torch.device("cpu")

        if resize_size is None:
            self.keys_to_resize: Dict[str, Tuple[int, int]] = {}
        else:
            self.keys_to_resize = {f"image_{i}": tuple(resize_size[i]) for i in resize_size.keys()}

        logging.info(f"Resizing images: {self.keys_to_resize}")

        spaces = dict(self.observation_space.spaces)
        for k, size in self.keys_to_resize.items():
            spaces[k] = gym.spaces.Box(low=0, high=255, shape=size + (3,), dtype=np.uint8)
        self.observation_space = gym.spaces.Dict(spaces)

    @staticmethod
    def _hwc_uint8_to_nchw_float(x: np.ndarray) -> torch.Tensor:
        # x: HWC uint8
        t = torch.from_numpy(x).to(dtype=torch.float32)  # HWC
        t = t.permute(2, 0, 1).unsqueeze(0)  # 1CHW
        return t

    @staticmethod
    def _nchw_float_to_hwc_uint8(t: torch.Tensor) -> np.ndarray:
        # t: 1CHW float in [0,255]
        t = t.squeeze(0).permute(1, 2, 0)  # HWC
        t = torch.clamp(torch.round(t), 0, 255).to(dtype=torch.uint8)
        return t.cpu().numpy()

    def _center_crop_for_avg_aug(self, t: torch.Tensor) -> torch.Tensor:
        # t: 1CHW
        _, _, h, w = t.shape
        # Match TF code:
        # new_height = sqrt(avg_scale / avg_ratio)
        # new_width  = sqrt(avg_scale * avg_ratio)
        new_h_frac = float(np.clip(np.sqrt(self.avg_scale / self.avg_ratio), 0.0, 1.0))
        new_w_frac = float(np.clip(np.sqrt(self.avg_scale * self.avg_ratio), 0.0, 1.0))
        crop_h = max(1, int(round(h * new_h_frac)))
        crop_w = max(1, int(round(w * new_w_frac)))
        top = max(0, (h - crop_h) // 2)
        left = max(0, (w - crop_w) // 2)
        return t[:, :, top : top + crop_h, left : left + crop_w]

    def _resize(self, t: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
        # bicubic + antialias approximates lanczos reasonably for inference resizing
        return F.interpolate(t, size=size_hw, mode="bicubic", align_corners=False, antialias=True)

    def observation(self, observation: Dict[str, Any]):
        for k, size in self.keys_to_resize.items():
            img = observation[k]
            if not (isinstance(img, np.ndarray) and img.dtype == np.uint8 and img.ndim == 3 and img.shape[2] == 3):
                raise ValueError(f"Expected {k} as HWC uint8 with 3 channels, got {type(img)} {getattr(img, 'shape', None)}")

            t = self._hwc_uint8_to_nchw_float(img).to(self.device)

            # initial resize to target size (approximating TF lanczos3 resize)
            t = self._resize(t, size)

            # average augmentation (center crop + resize)
            if k in self.augmented_keys:
                t = self._center_crop_for_avg_aug(t)
                t = self._resize(t, size)

            observation[k] = self._nchw_float_to_hwc_uint8(t)
        return observation


class NormalizeProprio(gym.ObservationWrapper):
    """Normalize proprio using provided metadata (mean/std/mask). Pure NumPy (no JAX)."""

    def __init__(self, env: gym.Env, action_proprio_metadata: dict):
        self.action_proprio_metadata = _to_numpy_tree(action_proprio_metadata)
        super().__init__(env)

    @staticmethod
    def normalize(data: np.ndarray, metadata: dict) -> np.ndarray:
        mean = np.asarray(metadata["mean"])
        std = np.asarray(metadata["std"])
        mask = np.asarray(metadata.get("mask", np.ones_like(mean, dtype=bool))).astype(bool)
        data = np.asarray(data)
        return np.where(mask, (data - mean) / (std + 1e-8), data)

    def observation(self, obs: Dict[str, Any]):
        if "proprio" in self.action_proprio_metadata:
            obs["proprio"] = self.normalize(obs["proprio"], self.action_proprio_metadata["proprio"])
        else:
            if "proprio" in obs:
                raise AssertionError("Cannot normalize proprio without metadata.")
        return obs


def _to_numpy_tree(x: Any) -> Any:
    """Recursively convert nested dict/list structures to NumPy arrays where appropriate."""
    if isinstance(x, dict):
        return {k: _to_numpy_tree(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        # lists in metadata are typically numeric arrays
        try:
            return np.array(x)
        except Exception:
            return type(x)(_to_numpy_tree(v) for v in x)
    return x
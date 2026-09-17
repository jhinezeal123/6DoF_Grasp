from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from octo.model.components.diffusion import cosine_beta_schedule
from octo.utils.train_utils_pt import _np2pt, _to_device


IMAGE_ALIASES = {
    "primary": "image_primary",
    "wrist": "image_wrist",
}


def parse_csv_paths(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    parts = [p.strip() for p in value.split(",") if p.strip()]
    return parts or None


def _canonical_image_key(key: str) -> str:
    return IMAGE_ALIASES.get(key, key)


def load_rgb_uint8(path: str, size_wh: Optional[Tuple[int, int]] = None) -> np.ndarray:
    from PIL import Image

    img = Image.open(path).convert("RGB")
    if size_wh is not None:
        img = img.resize(size_wh, Image.LANCZOS)
    arr = np.asarray(img, dtype=np.uint8).copy()
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected RGB image at {path}, got shape {arr.shape}")
    return arr


def _load_npz_mapping(path: Optional[str]) -> Dict[str, np.ndarray]:
    if path is None:
        return {}
    with np.load(path, allow_pickle=False) as data:
        return {str(k): np.asarray(v) for k, v in data.items()}


def _expected_image_size_from_jax(example_array: np.ndarray) -> Tuple[int, int]:
    if example_array.ndim == 5:
        _, _, h, w, _ = example_array.shape
    elif example_array.ndim == 4:
        _, h, w, _ = example_array.shape
    else:
        raise ValueError(f"Unsupported JAX image example shape: {example_array.shape}")
    return int(w), int(h)


def _strip_optional_batch_dim(arr: np.ndarray, expected_ndim_without_batch: int) -> np.ndarray:
    if arr.ndim == expected_ndim_without_batch + 1 and arr.shape[0] == 1:
        return arr[0]
    return arr


def build_jax_observation(
    example_observation: Mapping[str, np.ndarray],
    window: int,
    observation_overrides: Optional[Mapping[str, np.ndarray]] = None,
    image_sequences: Optional[Mapping[str, Sequence[str]]] = None,
) -> Dict[str, np.ndarray]:
    observation_overrides = dict(observation_overrides or {})
    image_sequences = {_canonical_image_key(k): v for k, v in (image_sequences or {}).items()}

    obs: Dict[str, np.ndarray] = {}
    pad_mask_dict: Dict[str, np.ndarray] = {}

    for key, example_value in example_observation.items():
        if key == "pad_mask_dict":
            continue
        if key == "timestep_pad_mask":
            obs[key] = np.ones((1, window), dtype=np.bool_)
            continue

        if key in observation_overrides:
            arr = np.asarray(observation_overrides[key])
            arr = _strip_optional_batch_dim(arr, example_value.ndim - 1)
            if arr.shape[0] != window:
                raise ValueError(
                    f"Override for observation key '{key}' must have first dimension == window ({window}), got {arr.shape}"
                )
            obs[key] = arr[None]
            pad_mask_dict[key] = np.ones((1, window), dtype=np.bool_)
            continue

        if key.startswith("image_") and key in image_sequences:
            paths = list(image_sequences[key])
            if len(paths) == 1:
                paths = paths * window
            if len(paths) != window:
                raise ValueError(f"Observation key '{key}' expects {window} image(s), got {len(paths)}")
            size_wh = _expected_image_size_from_jax(np.asarray(example_value))
            frames = [load_rgb_uint8(p, size_wh=size_wh) for p in paths]
            obs[key] = np.stack(frames, axis=0)[None]
            pad_mask_dict[key] = np.ones((1, window), dtype=np.bool_)
            continue

        target_shape = (1, window, *tuple(example_value.shape[2:]))
        obs[key] = np.zeros(target_shape, dtype=np.asarray(example_value).dtype)
        if key == "timestep":
            pad_mask_dict[key] = np.ones((1, window), dtype=np.bool_)
        else:
            pad_mask_dict[key] = np.zeros((1, window), dtype=np.bool_)

    obs["pad_mask_dict"] = pad_mask_dict
    obs.setdefault("timestep_pad_mask", np.ones((1, window), dtype=np.bool_))
    return obs


def build_goal_overrides(
    example_task: Mapping[str, np.ndarray],
    goal_overrides: Optional[Mapping[str, np.ndarray]] = None,
    goal_images: Optional[Mapping[str, str]] = None,
) -> Dict[str, np.ndarray]:
    goal_overrides = dict(goal_overrides or {})
    goal_images = {_canonical_image_key(k): v for k, v in (goal_images or {}).items()}
    goals: Dict[str, np.ndarray] = {}

    for key, arr in goal_overrides.items():
        if key not in example_task:
            raise KeyError(f"Task key '{key}' not found in example task. Available keys: {list(example_task.keys())}")
        ex = np.asarray(example_task[key])
        arr_np = _strip_optional_batch_dim(np.asarray(arr), ex.ndim - 1)
        goals[key] = arr_np

    for key, path in goal_images.items():
        if key not in example_task:
            raise KeyError(f"Task image key '{key}' not found in example task. Available keys: {list(example_task.keys())}")
        size_wh = _expected_image_size_from_jax(np.asarray(example_task[key]))
        goals[key] = load_rgb_uint8(path, size_wh=size_wh)

    return goals


def jax_obs_to_torch(jax_observation: Mapping[str, np.ndarray], device: torch.device) -> Dict:
    obs_pt = _np2pt({k: v for k, v in jax_observation.items() if k != "pad_mask_dict"}, device=None)
    pad_mask_pt = {
        k: torch.as_tensor(v, dtype=torch.bool, device=device)
        for k, v in jax_observation.get("pad_mask_dict", {}).items()
    }
    obs_pt["pad_mask_dict"] = pad_mask_pt
    return _to_device(obs_pt, device)


def ensure_torch_task_on_device(task: Dict, device: torch.device) -> Dict:
    out = {}
    for key, value in task.items():
        if key == "pad_mask_dict":
            out[key] = {
                k: (v if isinstance(v, torch.Tensor) else torch.as_tensor(v)).to(device=device)
                for k, v in value.items()
            }
        elif isinstance(value, torch.Tensor):
            out[key] = value.to(device=device)
        else:
            out[key] = value
    return out


def token_group_to_numpy(group) -> Tuple[np.ndarray, np.ndarray]:
    tokens = getattr(group, "tokens")
    mask = getattr(group, "mask")
    if isinstance(tokens, torch.Tensor):
        tokens_np = tokens.detach().cpu().numpy()
    else:
        tokens_np = np.asarray(tokens)
    if isinstance(mask, torch.Tensor):
        mask_np = mask.detach().cpu().numpy()
    else:
        mask_np = np.asarray(mask)
    return tokens_np, mask_np


def compare_arrays(a: np.ndarray, b: np.ndarray) -> Dict[str, object]:
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return {
            "shape_a": list(a.shape),
            "shape_b": list(b.shape),
            "match": False,
            "reason": "shape_mismatch",
        }
    diff = a.astype(np.float64) - b.astype(np.float64)
    abs_diff = np.abs(diff)
    denom = float(np.linalg.norm(a.ravel()) * np.linalg.norm(b.ravel()))
    cosine = None if denom == 0.0 else float(np.dot(a.ravel(), b.ravel()) / denom)
    return {
        "shape": list(a.shape),
        "match": True,
        "mean_abs": float(abs_diff.mean()),
        "max_abs": float(abs_diff.max()),
        "rmse": float(np.sqrt(np.mean(np.square(diff)))),
        "cosine": cosine,
    }


def compare_bool_arrays(a: np.ndarray, b: np.ndarray) -> Dict[str, object]:
    a = np.asarray(a).astype(bool)
    b = np.asarray(b).astype(bool)
    if a.shape != b.shape:
        return {
            "shape_a": list(a.shape),
            "shape_b": list(b.shape),
            "match": False,
            "reason": "shape_mismatch",
        }
    equal = a == b
    return {
        "shape": list(a.shape),
        "match": True,
        "fraction_equal": float(equal.mean()),
        "all_equal": bool(equal.all()),
    }


def make_shared_diffusion_noise(
    seed: int,
    diffusion_steps: int,
    shape: Tuple[int, ...],
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    base = rng.standard_normal(size=shape, dtype=np.float32)
    per_step = rng.standard_normal(size=(diffusion_steps, *shape), dtype=np.float32)
    return base, per_step


def _action_mask_np(
    batch_size: int,
    window_size: int,
    action_horizon: int,
    action_dim: int,
    embodiment_action_dim: Optional[int],
) -> np.ndarray:
    mask = np.ones((batch_size, window_size, action_horizon, action_dim), dtype=bool)
    if embodiment_action_dim is not None:
        mask[..., embodiment_action_dim:] = False
    return mask.reshape(batch_size, window_size, action_horizon * action_dim)


def run_shared_noise_diffusion_torch(
    action_head,
    transformer_outputs: Dict,
    seed: int,
    embodiment_action_dim: Optional[int] = None,
) -> np.ndarray:
    if not hasattr(action_head, "diffusion_steps"):
        raise TypeError("Shared-noise diffusion compare currently supports diffusion-based action heads only.")

    device = next(iter(transformer_outputs.values())).tokens.device
    batch_size, window_size = transformer_outputs[action_head.readout_key].tokens.shape[:2]
    flat_dim = action_head.action_horizon * action_head.action_dim
    base_noise, per_step_noise = make_shared_diffusion_noise(
        seed=seed,
        diffusion_steps=int(action_head.diffusion_steps),
        shape=(batch_size, window_size, flat_dim),
    )

    current_x = torch.as_tensor(base_noise, device=device, dtype=torch.float32)
    step_noises = torch.as_tensor(per_step_noise, device=device, dtype=torch.float32)
    flat_action_mask = torch.as_tensor(
        _action_mask_np(
            batch_size=batch_size,
            window_size=window_size,
            action_horizon=int(action_head.action_horizon),
            action_dim=int(action_head.action_dim),
            embodiment_action_dim=embodiment_action_dim,
        ),
        device=device,
        dtype=torch.bool,
    )

    for idx, time in enumerate(range(int(action_head.diffusion_steps) - 1, -1, -1)):
        input_time = torch.full((*current_x.shape[:-1], 1), float(time), device=device, dtype=torch.float32)
        eps_pred = action_head(transformer_outputs, time=input_time, noisy_actions=current_x, train=False)

        alpha_t = action_head.alphas[time].to(dtype=torch.float32)
        alpha_hat_t = action_head.alpha_hats[time].to(dtype=torch.float32)
        beta_t = action_head.betas[time].to(dtype=torch.float32)
        denom = torch.sqrt(torch.clamp(1.0 - alpha_hat_t, min=1e-12))
        alpha_1 = 1.0 / torch.sqrt(torch.clamp(alpha_t, min=1e-12))
        alpha_2 = (1.0 - alpha_t) / denom
        current_x = alpha_1 * (current_x - alpha_2 * eps_pred)

        z = step_noises[idx]
        if time > 0:
            current_x = current_x + torch.sqrt(torch.clamp(beta_t, min=1e-20)) * z
        current_x = torch.clip(current_x, -action_head.max_action, action_head.max_action)
        current_x = torch.where(
            flat_action_mask,
            current_x,
            torch.sqrt(torch.clamp(1.0 - alpha_hat_t, min=0.0)) * z,
        )

    actions = current_x.reshape(batch_size, window_size, action_head.action_horizon, action_head.action_dim)
    return actions[:, -1].detach().cpu().numpy()


def _jax_action_head_forward(model, transformer_outputs: Dict, time, noisy_actions):
    return model.module.apply(
        {"params": model.params},
        transformer_outputs,
        time=time,
        noisy_actions=noisy_actions,
        train=False,
        method=lambda module, transformer_outputs, time, noisy_actions, train=False: module.heads["action"](
            transformer_outputs,
            time=time,
            noisy_actions=noisy_actions,
            train=train,
        ),
    )


def run_shared_noise_diffusion_jax(
    model,
    transformer_outputs: Dict,
    diffusion_steps: int,
    action_horizon: int,
    action_dim: int,
    max_action: float,
    seed: int,
    embodiment_action_dim: Optional[int] = None,
    readout_key: str = "readout_action",
) -> np.ndarray:
    import jax.numpy as jnp

    batch_size, window_size = np.asarray(transformer_outputs[readout_key].tokens).shape[:2]
    flat_dim = action_horizon * action_dim
    base_noise, per_step_noise = make_shared_diffusion_noise(
        seed=seed,
        diffusion_steps=diffusion_steps,
        shape=(batch_size, window_size, flat_dim),
    )
    betas = np.asarray(cosine_beta_schedule(diffusion_steps), dtype=np.float32)
    alphas = 1.0 - betas
    alpha_hats = np.cumprod(alphas, axis=0)
    flat_action_mask = _action_mask_np(
        batch_size=batch_size,
        window_size=window_size,
        action_horizon=action_horizon,
        action_dim=action_dim,
        embodiment_action_dim=embodiment_action_dim,
    )

    current_x = jnp.asarray(base_noise, dtype=jnp.float32)
    step_noises = jnp.asarray(per_step_noise, dtype=jnp.float32)
    flat_action_mask = jnp.asarray(flat_action_mask)

    for idx, time in enumerate(range(diffusion_steps - 1, -1, -1)):
        input_time = jnp.full((*current_x.shape[:-1], 1), float(time), dtype=jnp.float32)
        eps_pred = _jax_action_head_forward(model, transformer_outputs, input_time, current_x)

        alpha_t = jnp.asarray(alphas[time], dtype=jnp.float32)
        alpha_hat_t = jnp.asarray(alpha_hats[time], dtype=jnp.float32)
        beta_t = jnp.asarray(betas[time], dtype=jnp.float32)
        denom = jnp.sqrt(jnp.clip(1.0 - alpha_hat_t, a_min=1e-12))
        alpha_1 = 1.0 / jnp.sqrt(jnp.clip(alpha_t, a_min=1e-12))
        alpha_2 = (1.0 - alpha_t) / denom
        current_x = alpha_1 * (current_x - alpha_2 * eps_pred)

        z = step_noises[idx]
        current_x = current_x + (time > 0) * (jnp.sqrt(jnp.clip(beta_t, a_min=1e-20)) * z)
        current_x = jnp.clip(current_x, -max_action, max_action)
        current_x = jnp.where(
            flat_action_mask,
            current_x,
            jnp.sqrt(jnp.clip(1.0 - alpha_hat_t, a_min=0.0)) * z,
        )

    actions = np.asarray(current_x).reshape(batch_size, window_size, action_horizon, action_dim)
    return actions[:, -1]


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def format_metric_row(name: str, metrics: Mapping[str, object]) -> str:
    if not metrics.get("match", False):
        return f"{name:<24} shape mismatch"
    parts = []
    if "mean_abs" in metrics:
        parts.append(f"mean_abs={metrics['mean_abs']:.6g}")
    if "max_abs" in metrics:
        parts.append(f"max_abs={metrics['max_abs']:.6g}")
    if "rmse" in metrics:
        parts.append(f"rmse={metrics['rmse']:.6g}")
    if "fraction_equal" in metrics:
        parts.append(f"equal={metrics['fraction_equal']:.3f}")
    return f"{name:<24} " + ", ".join(parts)


def unnormalize_action_numpy(action: np.ndarray, stats: Optional[Mapping[str, np.ndarray]]) -> np.ndarray:
    if stats is None:
        return np.asarray(action)
    action = np.asarray(action).astype(np.float32)
    if "mean" in stats and "std" in stats:
        mean = np.asarray(stats["mean"], dtype=np.float32)
        std = np.asarray(stats["std"], dtype=np.float32)
        mask = stats.get("mask", None)
        if mask is None:
            mask = np.isfinite(mean) & np.isfinite(std)
        else:
            mask = np.asarray(mask).astype(bool)
        mean = np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
        std = np.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0)
        action = action[..., : len(mask)]
        return np.where(mask, action * std + mean, action)
    if "p01" in stats and "p99" in stats:
        p01 = np.asarray(stats["p01"], dtype=np.float32)
        p99 = np.asarray(stats["p99"], dtype=np.float32)
        mask = stats.get("mask", np.ones_like(p01, dtype=bool))
        mask = np.asarray(mask).astype(bool)
        action = action[..., : len(mask)]
        return np.where(mask, (action + 1.0) * (p99 - p01) / 2.0 + p01, action)
    return action

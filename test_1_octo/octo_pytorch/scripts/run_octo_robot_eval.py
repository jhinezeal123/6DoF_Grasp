#!/usr/bin/env python3
"""Run a torch Octo checkpoint on either MyArm M750 or a mock MyArm backend.

This script follows the real-robot eval structure used by Octo's WidowX example,
but replaces the robot backend with a Gym-compatible MyArm wrapper and a modular
camera stack so the same pipeline can run on Jetson with torch-only inference.

Examples
--------
Real robot:
  python scripts/run_octo_robot_eval.py \
    --config scripts/robot_configs/myarm_m750_primary_wrist_720p_mjpg.json \
    --text "pick up the coke can"

Safe mock test with live cameras + 3D TCP visualization:
  python scripts/run_octo_robot_eval.py \
    --config scripts/robot_configs/mock_myarm_primary_wrist_720p_mjpg.json \
    --text "pick up the coke can"
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TRANSFORMERS_NO_JAX", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from octo.robot.cameras import MultiCameraRig
from octo.robot.config import RobotPipelineConfig
from octo.robot.debug_logger import RobotDebugLogger
from octo.robot.env import MyArmM750GymEnv
from octo.robot.mock_robot import MockMyArmM750Robot
from octo.robot.myarm_m750 import MyArmM750Robot
from octo.robot.policy import OctoTorchPolicy
from octo.robot.utils import center_crop_resize, summarize_pose
from octo.utils.gym_wrappers_pt import (
    HistoryWrapper,
    OctoObsPreprocessWrapper,
    ResizeImageWrapperPT,
    RHCWrapper,
    TemporalEnsembleWrapper,
)


def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="JSON config for robot + cameras + rollout")
    ap.add_argument("--checkpoint_dir", default=None, help="Override checkpoint_dir from config")
    ap.add_argument("--dataset_key", default=None, help="Override dataset_key from config")
    ap.add_argument("--text", default=None, help="Language instruction")
    ap.add_argument("--goal_primary", default=None, help="Path to primary goal image")
    ap.add_argument("--goal_wrist", default=None, help="Path to wrist goal image")
    ap.add_argument("--max_steps", type=int, default=None, help="Override rollout max_steps")
    ap.add_argument("--show_preview", action="store_true", help="Show current camera frame")
    ap.add_argument(
        "--fp32",
        action="store_true",
        help="Force full FP32 inference (disables rollout.fp16 autocast).",
    )
    ap.add_argument(
        "--octo_preprocess_observation",
        action="store_true",
        help="Use Octo-style observation preprocessing (e.g., pad proprio to match training).",
    )
    ap.add_argument(
        "--octo_resize_images",
        action="store_true",
        help="Use Octo-style image resizing + average crop/resize (closer to Octo JAX eval).",
    )
    return ap.parse_args()


def _load_rgb(path: str, target_hw):
    from PIL import Image

    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)
    return center_crop_resize(arr, target_hw)


def _make_goal_dict(args, policy: OctoTorchPolicy) -> Optional[Dict[str, np.ndarray]]:
    goal_sizes = policy.expected_goal_image_sizes()
    goals = {}
    if args.goal_primary is not None:
        if "image_primary" not in goal_sizes:
            raise KeyError("Loaded checkpoint does not accept goal image_primary")
        goals["image_primary"] = _load_rgb(args.goal_primary, goal_sizes["image_primary"])
    if args.goal_wrist is not None:
        if "image_wrist" not in goal_sizes:
            raise KeyError("Loaded checkpoint does not accept goal image_wrist")
        goals["image_wrist"] = _load_rgb(args.goal_wrist, goal_sizes["image_wrist"])
    return goals or None


def _build_robot(cfg: RobotPipelineConfig):
    backend = cfg.robot.backend.lower()
    if backend in {"mock", "mock_myarm", "mock_myarm_m750"}:
        return MockMyArmM750Robot(cfg.robot, visualize=cfg.rollout.visualize_mock_3d)
    if backend in {"myarm", "myarm_m750"}:
        return MyArmM750Robot(cfg.robot)
    raise ValueError(f"Unsupported robot backend: {cfg.robot.backend}")


def _wait_until_robot_idle(robot, timeout_s: float = 10.0) -> None:
    arm = getattr(robot, "_arm", None)
    if arm is None:
        return
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        try:
            moving = arm.is_moving()
        except Exception:
            break
        if moving in (0, False):
            break
        time.sleep(0.05)


def _resize_keep_aspect(image_rgb: np.ndarray, target_h: int) -> np.ndarray:
    import cv2

    h, w = image_rgb.shape[:2]
    if h <= 0 or w <= 0 or h == target_h:
        return image_rgb
    scale = float(target_h) / float(h)
    new_w = max(1, int(round(w * scale)))
    return cv2.resize(image_rgb, (new_w, int(target_h)), interpolation=cv2.INTER_AREA)


def _annotate_preview_frame(image_rgb: np.ndarray, *, label: str, meta: Optional[dict], show_label: bool, show_timestamp: bool) -> np.ndarray:
    import cv2

    frame = image_rgb.copy()
    y = 24
    if show_label:
        cv2.putText(frame, label.upper(), (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        y += 24
    if show_timestamp and meta is not None:
        age_s = float(meta.get("age_s", np.nan))
        cv2.putText(frame, f"age={age_s:.3f}s", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
    return frame


def _make_preview_image(
    cameras: MultiCameraRig,
    rollout_cfg,
    *,
    step: int,
    infer_dt: float,
    robot_debug: Dict[str, object],
) -> Optional[np.ndarray]:
    import cv2

    try:
        preview_obs, preview_meta = cameras.snapshot_with_metadata(target_sizes=None)
    except Exception:
        return None

    frame_items = []
    for name in ("primary", "wrist"):
        key = f"image_{name}"
        if key in preview_obs:
            frame = _annotate_preview_frame(
                preview_obs[key],
                label=name,
                meta=preview_meta.get(name),
                show_label=bool(getattr(rollout_cfg, "preview_show_labels", True)),
                show_timestamp=bool(getattr(rollout_cfg, "preview_show_timestamps", True)),
            )
            frame_items.append((name, frame))
    if not frame_items:
        for key, frame in preview_obs.items():
            name = key.replace("image_", "")
            frame = _annotate_preview_frame(
                frame,
                label=name,
                meta=preview_meta.get(name),
                show_label=bool(getattr(rollout_cfg, "preview_show_labels", True)),
                show_timestamp=bool(getattr(rollout_cfg, "preview_show_timestamps", True)),
            )
            frame_items.append((name, frame))

    if not frame_items:
        return None

    layout = str(getattr(rollout_cfg, "preview_layout", "side_by_side"))
    if layout == "auto":
        layout = "side_by_side" if len(frame_items) > 1 else "single"
    if layout == "single":
        frame_items = [frame_items[0]]

    target_h = int(getattr(rollout_cfg, "preview_max_height", 360))
    target_h = max(120, target_h)
    resized_frames = [_resize_keep_aspect(frame, target_h) for _, frame in frame_items]

    if len(resized_frames) == 1:
        canvas = resized_frames[0]
    else:
        separator = np.zeros((target_h, 8, 3), dtype=np.uint8)
        canvas = resized_frames[0]
        for frame in resized_frames[1:]:
            canvas = np.concatenate([canvas, separator, frame], axis=1)

    footer_h = 34
    footer = np.zeros((footer_h, canvas.shape[1], 3), dtype=np.uint8)
    infer_hz = 0.0 if infer_dt <= 0 else 1.0 / infer_dt
    status = (
        f"step={step:03d} infer={infer_dt*1000.0:.1f}ms {infer_hz:.2f}Hz "
        f"resync={bool(robot_debug.get('resync_applied', False))} "
        f"mode={robot_debug.get('integration_mode', '')} "
        f"pivot={robot_debug.get('pivot_source', '')}"
    )
    cv2.putText(footer, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate([canvas, footer], axis=0)


def main():
    args = _parse_args()
    cfg = RobotPipelineConfig.from_json(args.config)
    if args.checkpoint_dir is not None:
        cfg.rollout.checkpoint_dir = args.checkpoint_dir
    if args.dataset_key is not None:
        cfg.rollout.dataset_key = args.dataset_key
    if args.max_steps is not None:
        cfg.rollout.max_steps = int(args.max_steps)
    if args.fp32:
        cfg.rollout.fp16 = False
    if args.octo_preprocess_observation:
        cfg.rollout.octo_preprocess_observation = True
    if args.octo_resize_images:
        cfg.rollout.octo_resize_images = True

    if not cfg.rollout.checkpoint_dir:
        raise ValueError("checkpoint_dir must be set either in config or via --checkpoint_dir")
    if args.text is None and args.goal_primary is None and args.goal_wrist is None:
        raise ValueError("Provide at least --text or a goal image")

    save_dir = Path(cfg.rollout.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    policy = OctoTorchPolicy.from_checkpoint(
        checkpoint_dir=cfg.rollout.checkpoint_dir,
        device=cfg.rollout.device,
        dataset_key=cfg.rollout.dataset_key,
        fp16=cfg.rollout.fp16,
        seed=cfg.rollout.seed,
    )

    if (not cfg.rollout.fp16) and policy.device.type == "cuda":
        try:
            import torch

            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        except Exception:
            pass

    model_image_sizes = policy.expected_image_sizes()
    include_proprio = "proprio" in policy.model.example_batch["observation"]
    disable_proprio = bool(getattr(cfg.rollout, "disable_proprio", False))
    if disable_proprio:
        include_proprio = False

    expected_proprio_dim = None
    if include_proprio:
        try:
            expected_proprio_dim = int(policy.model.example_batch["observation"]["proprio"].shape[-1])
        except Exception:
            expected_proprio_dim = None

    pad_proprio_to_8 = bool(
        cfg.rollout.octo_preprocess_observation
        and include_proprio
        and expected_proprio_dim == 8
    )
    model_horizon = policy.model_action_horizon()
    exec_horizon = max(1, min(int(cfg.rollout.exec_horizon), int(model_horizon)))

    cameras = None
    robot = None
    env = None
    preview_available = False
    try:
        cameras = MultiCameraRig(cfg.cameras)
        cameras.start()
        available = set(cameras.available_modalities())
        active_model_image_sizes = {
            name: hw for name, hw in model_image_sizes.items() if name in available
        }
        if not active_model_image_sizes:
            raise RuntimeError(
                f"No enabled cameras match the checkpoint observation modalities. "
                f"Model expects {list(model_image_sizes.keys())}, cameras provide {list(available)}"
            )

        if cfg.rollout.octo_resize_images:
            env_image_sizes = {
                name: (int(cfg.cameras[name].height), int(cfg.cameras[name].width))
                for name in active_model_image_sizes.keys()
            }
            camera_target_sizes = None
        else:
            env_image_sizes = dict(active_model_image_sizes)
            camera_target_sizes = dict(active_model_image_sizes)

        robot = _build_robot(cfg)
        robot.connect()
        # Ap cau hinh thiet bi (fresh_mode, movement_type, end_type,
        # gripper_enabled, world/tool reference) NGAY sau khi ket noi.
        # Ban goc chi goi ham nay trong khoi finally, tuc la ca rollout chay voi
        # cau hinh thiet bi tuy tien, va reset_to_home() o env.reset() ben duoi
        # la chuyen dong dau tien da xay ra TRUOC khi cau hinh.
        robot.configure_for_policy()

        requested_control_period_s = float(getattr(cfg.robot, "control_period_s", 0.0))
        if requested_control_period_s <= 0.0:
            requested_control_period_s = 0.0 if cfg.robot.blocking_move else 1.0 / max(cfg.rollout.control_rate_hz, 1e-6)

        env = MyArmM750GymEnv(
            robot=robot,
            cameras=cameras,
            robot_config=cfg.robot,
            image_sizes=env_image_sizes,
            camera_target_sizes=camera_target_sizes,
            include_proprio=include_proprio,
            pad_proprio_to_8=pad_proprio_to_8,
            control_period_s=requested_control_period_s,
        )

        if cfg.rollout.octo_preprocess_observation:
            primary_size = model_image_sizes.get("primary")
            im_size = None
            if primary_size is not None and int(primary_size[0]) == int(primary_size[1]):
                im_size = int(primary_size[0])
            env = OctoObsPreprocessWrapper(
                env,
                im_size=im_size,
                pad_proprio_to_8=pad_proprio_to_8,
            )

        if cfg.rollout.octo_resize_images:
            env = ResizeImageWrapperPT(
                env,
                resize_size=active_model_image_sizes,
                avg_scale=float(getattr(cfg.rollout, "octo_resize_avg_scale", 0.9)),
                avg_ratio=float(getattr(cfg.rollout, "octo_resize_avg_ratio", 1.0)),
            )
        env = HistoryWrapper(env, int(cfg.rollout.window_size))
        if cfg.rollout.use_temporal_ensemble:
            env = TemporalEnsembleWrapper(env, pred_horizon=exec_horizon)
        else:
            env = RHCWrapper(env, exec_horizon=exec_horizon)

        goals = _make_goal_dict(args, policy)
        task = policy.create_task(text=args.text, goal_images=goals)

        print("Checkpoint:", cfg.rollout.checkpoint_dir)
        print("Device:", policy.device)
        print("Robot backend:", cfg.robot.backend)
        try:
            import torch

            param_dtype = next(policy.model.parameters()).dtype
            print("Model param dtype:", param_dtype)
            if policy.device.type == "cuda":
                print("TF32 matmul allowed:", bool(torch.backends.cuda.matmul.allow_tf32))
                print("TF32 cudnn allowed:", bool(torch.backends.cudnn.allow_tf32))
        except Exception:
            pass
        print("Model observation modalities:", list(model_image_sizes.keys()))
        print("Active cameras for model:", list(active_model_image_sizes.keys()))
        print("All enabled cameras:", list(available))
        print("Window size:", cfg.rollout.window_size)
        print("Blocking move:", bool(cfg.robot.blocking_move))
        print("Control period [s]:", float(requested_control_period_s))
        print("SE3 integration mode:", cfg.robot.se3_integration_mode)
        print("Hardware measured enabled:", bool(cfg.robot.hardware_measured_enabled))
        print("Pivot source:", cfg.robot.pivot_source)
        print("Compose base:", cfg.robot.compose_base)
        print("Periodic resync enabled:", bool(cfg.robot.periodic_resync_enabled))
        print("Periodic resync every n steps:", int(cfg.robot.periodic_resync_every_n_steps))
        print("Proprio orientation mode:", cfg.robot.proprio_orientation_mode)
        print("Octo preprocess observation:", bool(cfg.rollout.octo_preprocess_observation))
        print("Disable proprio:", bool(disable_proprio))
        print("Octo resize images:", bool(getattr(cfg.rollout, "octo_resize_images", False)))
        print("Preview layout:", cfg.rollout.preview_layout)
        if include_proprio:
            print("Expected proprio dim:", expected_proprio_dim)
            print("Pad proprio to 8:", pad_proprio_to_8)
        print("Model action horizon:", model_horizon)
        print("Execution horizon:", exec_horizon)

        obs, _ = env.reset()

        preview_enabled = bool(cfg.rollout.show_camera_preview or args.show_preview)
        if preview_enabled:
            try:
                import cv2  # noqa: F401

                preview_available = True
            except Exception:
                preview_available = False

        debug_logger = RobotDebugLogger() if bool(getattr(cfg.rollout, "enable_debug_logging", False)) else None

        rollout_t0 = time.time()
        for step in range(int(cfg.rollout.max_steps)):
            infer_t0 = time.time()
            action_chunk = policy.sample_action_chunk(obs, task)
            infer_dt = time.time() - infer_t0
            if action_chunk.ndim == 1:
                action_chunk = action_chunk[None]

            obs, reward, done, trunc, info = env.step(action_chunk)
            robot_debug = robot.get_debug_snapshot() if hasattr(robot, "get_debug_snapshot") else {}
            camera_meta = env.unwrapped.get_last_camera_meta() if hasattr(env.unwrapped, "get_last_camera_meta") else {}
            if debug_logger is not None:
                debug_logger.log_step(
                    step=step,
                    infer_ms=infer_dt * 1000.0,
                    robot_debug=robot_debug,
                    camera_meta=camera_meta,
                )
            tcp = summarize_pose(robot.get_tcp_pose_m_rad())
            trans_err = float(robot_debug.get("translation_error_m", 0.0))
            rot_err = float(robot_debug.get("rotation_error_rad", 0.0))
            print(
                f"step={step:03d} infer={infer_dt*1000:.1f}ms "
                f"tcp=({tcp['x_m']:.3f}, {tcp['y_m']:.3f}, {tcp['z_m']:.3f}) "
                f"rpy=({tcp['rx_rad']:.3f}, {tcp['ry_rad']:.3f}, {tcp['rz_rad']:.3f}) "
                f"gripper={robot.get_gripper_open_fraction():.2f} "
                f"err=({trans_err:.4f}m, {rot_err:.4f}rad) "
                f"mode={robot_debug.get('integration_mode', '')} "
                f"pivot={robot_debug.get('pivot_source', '')} "
                f"compose={robot_debug.get('compose_base', '')} "
                f"resync={bool(robot_debug.get('resync_applied', False))}"
            )

            if preview_available:
                import cv2

                preview_img = _make_preview_image(
                    cameras,
                    cfg.rollout,
                    step=step,
                    infer_dt=infer_dt,
                    robot_debug=robot_debug,
                )
                if preview_img is not None:
                    cv2.imshow("octo_robot_preview", cv2.cvtColor(preview_img, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1)

            if done or trunc:
                if cfg.rollout.terminate_on_truncation:
                    break

        total_dt = time.time() - rollout_t0
        print(f"Rollout finished in {total_dt:.2f}s")

        if hasattr(robot, "save_plot"):
            robot.save_plot(cfg.rollout.save_mock_plot_path)
        if debug_logger is not None:
            debug_csv_path = cfg.rollout.save_debug_csv_path or str(save_dir / "robot_debug_trace.csv")
            debug_plot_path = cfg.rollout.save_debug_plot_path or str(save_dir / "robot_debug_trace.png")
            debug_logger.save_csv(debug_csv_path)
            debug_logger.save_plot(debug_plot_path)
            print(f"Saved debug CSV: {debug_csv_path}")
            print(f"Saved debug plot: {debug_plot_path}")
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt: stopping rollout")
    finally:
        if robot is not None:
            try:
                print("Returning robot to home pose...")
                try:
                    robot.configure_for_policy()
                except Exception:
                    pass
                robot.reset_to_home()
                _wait_until_robot_idle(robot, timeout_s=max(10.0, float(cfg.robot.wait_timeout_s)))
                print("Robot reset_to_home() done")
            except Exception as e:
                print(f"WARNING: failed to reset_to_home(): {e}")

        if env is not None:
            try:
                env.close()
            except Exception:
                pass

        if cameras is not None:
            try:
                cameras.stop()
            except Exception:
                pass

        if robot is not None:
            try:
                robot.disconnect()
            except Exception:
                pass


if __name__ == "__main__":
    main()

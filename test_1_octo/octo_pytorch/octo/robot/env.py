from __future__ import annotations

import time
from typing import Dict, Optional

import gymnasium as gym
import numpy as np

from octo.robot.cameras import MultiCameraRig
from octo.robot.config import RobotConfig
from octo.robot.interfaces import RobotInterface


class MyArmM750GymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        robot: RobotInterface,
        cameras: MultiCameraRig,
        robot_config: RobotConfig,
        image_sizes: Dict[str, tuple],
        camera_target_sizes: Optional[Dict[str, tuple]] = None,
        include_proprio: bool = False,
        pad_proprio_to_8: bool = False,
        control_period_s: float = 0.0,
    ):
        super().__init__()
        self.robot = robot
        self.cameras = cameras
        self.robot_config = robot_config
        self.image_sizes = image_sizes
        self.camera_target_sizes = camera_target_sizes
        self.include_proprio = include_proprio
        self.pad_proprio_to_8 = bool(pad_proprio_to_8)
        self.control_period_s = float(control_period_s)
        self._last_step_t = None
        self._last_camera_meta: Dict[str, dict] = {}
        self._gripper_state = 1.0 if robot_config.open_gripper_on_reset else 0.0
        self._gripper_change_count = 0

        obs_spaces = {}
        for name, hw in image_sizes.items():
            h, w = int(hw[0]), int(hw[1])
            obs_spaces[f"image_{name}"] = gym.spaces.Box(
                low=0,
                high=255,
                shape=(h, w, 3),
                dtype=np.uint8,
            )
        if include_proprio:
            proprio_dim = 8 if self.pad_proprio_to_8 else 7
            obs_spaces["proprio"] = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(proprio_dim,),
                dtype=np.float32,
            )
        self.observation_space = gym.spaces.Dict(obs_spaces)
        self.action_space = gym.spaces.Box(
            low=np.array(
                [
                    -robot_config.max_step_translation_m,
                    -robot_config.max_step_translation_m,
                    -robot_config.max_step_translation_m,
                    -robot_config.max_step_rotation_rad,
                    -robot_config.max_step_rotation_rad,
                    -robot_config.max_step_rotation_rad,
                    0.0,
                ],
                dtype=np.float32,
            ),
            high=np.array(
                [
                    robot_config.max_step_translation_m,
                    robot_config.max_step_translation_m,
                    robot_config.max_step_translation_m,
                    robot_config.max_step_rotation_rad,
                    robot_config.max_step_rotation_rad,
                    robot_config.max_step_rotation_rad,
                    1.0,
                ],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

    def _get_obs(self) -> Dict[str, np.ndarray]:
        obs, meta = self.cameras.snapshot_with_metadata(target_sizes=self.camera_target_sizes)
        self._last_camera_meta = meta
        if self.include_proprio:
            proprio = self.robot.get_proprio().astype(np.float32)
            if self.pad_proprio_to_8:
                if proprio.shape[0] < 7:
                    raise ValueError(f"Expected proprio with at least 7 dims, got {proprio.shape}")
                proprio = np.concatenate([proprio[:6], np.asarray([0.0], dtype=np.float32), proprio[-1:]], axis=0)
            obs["proprio"] = proprio
        return obs

    def get_last_camera_meta(self) -> Dict[str, dict]:
        return dict(self._last_camera_meta)

    def _apply_sticky_gripper(self, desired_open: float) -> float:
        desired_open = 1.0 if desired_open >= self.robot_config.gripper_threshold else 0.0
        if desired_open != self._gripper_state:
            self._gripper_change_count += 1
        else:
            self._gripper_change_count = 0
        if self._gripper_change_count >= max(1, self.robot_config.gripper_sticky_steps):
            self._gripper_state = desired_open
            self._gripper_change_count = 0
        return self._gripper_state

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape[0] < 6:
            raise ValueError(f"Expected at least 6 action dims, got {action.shape}")
        delta_pose = action[:6]
        desired_gripper = float(action[6]) if action.shape[0] >= 7 else self._gripper_state
        gripper_cmd = self._apply_sticky_gripper(desired_gripper)

        if self.control_period_s > 0 and self._last_step_t is not None:
            elapsed = time.time() - self._last_step_t
            remaining = self.control_period_s - elapsed
            if remaining > 0:
                time.sleep(remaining)
        self._last_step_t = time.time()

        self.robot.apply_action(delta_pose, gripper_cmd)
        obs = self._get_obs()
        return obs, 0.0, False, False, {}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.robot.configure_for_policy()
        self.robot.reset_to_home()
        self._gripper_state = 1.0 if self.robot_config.open_gripper_on_reset else 0.0
        self._gripper_change_count = 0
        self._last_step_t = None
        obs = self._get_obs()
        return obs, {}

    def close(self):
        try:
            self.cameras.stop()
        finally:
            self.robot.disconnect()

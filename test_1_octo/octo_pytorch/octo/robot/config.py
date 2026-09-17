from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


SE3_INTEGRATION_MODES = {"bridge_exact", "measured_compose", "target_only"}
PIVOT_SOURCES = {"hardware_measured", "target_transform", "measured_or_target"}
COMPOSE_BASE_SOURCES = {"previous_target", "hardware_measured"}
TCP_POSE_SOURCES = {"internal", "hardware"}
PROPRIO_ORIENTATION_MODES = {"absolute", "relative_to_neutral"}
PREVIEW_LAYOUTS = {"auto", "single", "side_by_side"}


def _ensure_len(values: List[float], n: int, name: str) -> List[float]:
    if len(values) != n:
        raise ValueError(f"{name} must have length {n}, got {len(values)}")
    return values


@dataclass
class WorkspaceBounds:
    translation_min_m: List[float]
    translation_max_m: List[float]
    rotation_min_rad: List[float]
    rotation_max_rad: List[float]

    def __post_init__(self) -> None:
        _ensure_len(self.translation_min_m, 3, "translation_min_m")
        _ensure_len(self.translation_max_m, 3, "translation_max_m")
        _ensure_len(self.rotation_min_rad, 3, "rotation_min_rad")
        _ensure_len(self.rotation_max_rad, 3, "rotation_max_rad")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkspaceBounds":
        return cls(
            translation_min_m=list(data["translation_min_m"]),
            translation_max_m=list(data["translation_max_m"]),
            rotation_min_rad=list(data["rotation_min_rad"]),
            rotation_max_rad=list(data["rotation_max_rad"]),
        )


@dataclass
class CameraConfig:
    name: str
    enabled: bool = True
    index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 60
    fourcc: str = "MJPG"
    api_preference: int = 0
    warmup_frames: int = 10
    buffer_size: int = 1
    convert_rgb: bool = True
    crop_top: int = 0
    crop_bottom: int = 0
    crop_left: int = 0
    crop_right: int = 0
    flip_horizontal: bool = False
    flip_vertical: bool = False
    rotation_deg: int = 0
    stale_timeout_s: float = 1.0
    allow_stale_frames: bool = True

    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any]) -> "CameraConfig":
        return cls(
            name=name,
            enabled=bool(data.get("enabled", True)),
            index=int(data.get("index", 0)),
            width=int(data.get("width", 1280)),
            height=int(data.get("height", 720)),
            fps=int(data.get("fps", 60)),
            fourcc=str(data.get("fourcc", "MJPG")),
            api_preference=int(data.get("api_preference", 0)),
            warmup_frames=int(data.get("warmup_frames", 10)),
            buffer_size=int(data.get("buffer_size", 1)),
            convert_rgb=bool(data.get("convert_rgb", True)),
            crop_top=int(data.get("crop_top", 0)),
            crop_bottom=int(data.get("crop_bottom", 0)),
            crop_left=int(data.get("crop_left", 0)),
            crop_right=int(data.get("crop_right", 0)),
            flip_horizontal=bool(data.get("flip_horizontal", False)),
            flip_vertical=bool(data.get("flip_vertical", False)),
            rotation_deg=int(data.get("rotation_deg", 0)),
            stale_timeout_s=float(data.get("stale_timeout_s", 1.0)),
            allow_stale_frames=bool(data.get("allow_stale_frames", True)),
        )


@dataclass
class RobotConfig:
    backend: str = "myarm_m750"
    port: str = "/dev/ttyACM0"
    baudrate: Optional[int] = 1000000
    power_on: bool = True
    fresh_mode: int = 1
    reference_frame: int = 0
    movement_type: int = 1
    end_type: int = 1
    move_speed: int = 10
    coord_mode: int = 0
    blocking_move: bool = True
    control_period_s: float = 0.0
    post_command_sleep_s: float = 0.0
    hardware_read_settle_s: float = 0.03
    reset_readback_sync: bool = True
    home_pose_m_rad: List[float] = field(default_factory=lambda: [0.30, 0.0, 0.1, 1.5, 0.0, 1.5])
    neutral_orientation_rpy_rad: Optional[List[float]] = None
    proprio_orientation_mode: str = "relative_to_neutral"
    open_gripper_on_reset: bool = True
    max_step_translation_m: float = 0.1
    max_step_rotation_rad: float = 0.15
    gripper_threshold: float = 0.5
    gripper_sticky_steps: int = 1
    gripper_speed: int = 50
    gripper_open_value: int = 100
    gripper_close_value: int = 0
    gripper_use_value_api: bool = False
    reset_sleep_s: float = 1.0
    wait_for_motion: bool = False
    wait_timeout_s: float = 5.0
    tcp_pose_source: str = "internal"

    # v0_2_3 control semantics.
    se3_integration_mode: str = "bridge_exact"
    hardware_measured_enabled: bool = True
    pivot_source: str = "hardware_measured"
    compose_base: str = "previous_target"
    clamp_rotation_in_pose: bool = False
    periodic_resync_enabled: bool = False
    periodic_resync_every_n_steps: int = 0
    periodic_resync_position_threshold_m: float = 0.03
    periodic_resync_rotation_threshold_rad: float = 0.35
    debug_print_every_n_steps: int = 1

    workspace: WorkspaceBounds = field(
        default_factory=lambda: WorkspaceBounds(
            translation_min_m=[0.15, -0.30, 0.05],
            translation_max_m=[0.55, 0.30, 0.30],
            rotation_min_rad=[-1.5, -1.5, -1.5],
            rotation_max_rad=[1.5, 1.5, 1.5],
        )
    )

    def __post_init__(self) -> None:
        _ensure_len(self.home_pose_m_rad, 6, "home_pose_m_rad")
        if self.neutral_orientation_rpy_rad is not None:
            _ensure_len(self.neutral_orientation_rpy_rad, 3, "neutral_orientation_rpy_rad")
        if self.tcp_pose_source not in TCP_POSE_SOURCES:
            raise ValueError(
                f"tcp_pose_source must be one of {sorted(TCP_POSE_SOURCES)}, got {self.tcp_pose_source!r}"
            )
        if self.proprio_orientation_mode not in PROPRIO_ORIENTATION_MODES:
            raise ValueError(
                "proprio_orientation_mode must be 'absolute' or 'relative_to_neutral', "
                f"got {self.proprio_orientation_mode!r}"
            )
        if self.se3_integration_mode not in SE3_INTEGRATION_MODES:
            raise ValueError(
                f"se3_integration_mode must be one of {sorted(SE3_INTEGRATION_MODES)}, "
                f"got {self.se3_integration_mode!r}"
            )
        if self.pivot_source not in PIVOT_SOURCES:
            raise ValueError(
                f"pivot_source must be one of {sorted(PIVOT_SOURCES)}, got {self.pivot_source!r}"
            )
        if self.compose_base not in COMPOSE_BASE_SOURCES:
            raise ValueError(
                f"compose_base must be one of {sorted(COMPOSE_BASE_SOURCES)}, got {self.compose_base!r}"
            )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RobotConfig":
        workspace = WorkspaceBounds.from_dict(data["workspace"])
        kwargs = dict(data)
        kwargs["workspace"] = workspace
        kwargs["home_pose_m_rad"] = list(kwargs.get("home_pose_m_rad", [0.30, 0.0, 0.1, 1.5, 0.0, 1.5]))
        if kwargs.get("neutral_orientation_rpy_rad") is not None:
            kwargs["neutral_orientation_rpy_rad"] = list(kwargs["neutral_orientation_rpy_rad"])

        # Backward-compatibility mapping from v0_2_0-v0_2_2 config keys.
        if "se3_integration_mode" not in kwargs:
            if bool(kwargs.get("use_bridge_style_action_integration", True)):
                kwargs["se3_integration_mode"] = "bridge_exact"
            else:
                kwargs["se3_integration_mode"] = "target_only"
        if "hardware_measured_enabled" not in kwargs:
            kwargs["hardware_measured_enabled"] = bool(
                kwargs.get("tcp_pose_source") == "hardware"
                or kwargs.get("resync_use_hardware_pose_for_pivot", False)
            )
        if "pivot_source" not in kwargs:
            kwargs["pivot_source"] = (
                "hardware_measured"
                if bool(kwargs.get("resync_use_hardware_pose_for_pivot", False))
                else "target_transform"
            )
        if "compose_base" not in kwargs:
            kwargs["compose_base"] = "previous_target"
        if "periodic_resync_enabled" not in kwargs:
            kwargs["periodic_resync_enabled"] = int(kwargs.get("resync_every_n_steps", 0)) > 0
        if "periodic_resync_every_n_steps" not in kwargs:
            kwargs["periodic_resync_every_n_steps"] = int(kwargs.get("resync_every_n_steps", 0))
        if "periodic_resync_position_threshold_m" not in kwargs:
            kwargs["periodic_resync_position_threshold_m"] = float(
                kwargs.get("resync_threshold_translation_m", 0.03)
            )
        if "periodic_resync_rotation_threshold_rad" not in kwargs:
            kwargs["periodic_resync_rotation_threshold_rad"] = float(
                kwargs.get("resync_threshold_rotation_rad", 0.35)
            )
        # Prefer transform-space integration with late pose conversion in bridge_exact.
        if "clamp_rotation_in_pose" not in kwargs and kwargs["se3_integration_mode"] == "bridge_exact":
            kwargs["clamp_rotation_in_pose"] = False

        for legacy_key in (
            "use_bridge_style_action_integration",
            "resync_every_n_steps",
            "resync_threshold_translation_m",
            "resync_threshold_rotation_rad",
            "resync_use_hardware_pose_for_pivot",
        ):
            kwargs.pop(legacy_key, None)
        return cls(**kwargs)


@dataclass
class RolloutConfig:
    checkpoint_dir: str = ""
    dataset_key: Optional[str] = "bridge_dataset"
    window_size: int = 2
    exec_horizon: int = 4
    use_temporal_ensemble: bool = True
    control_rate_hz: float = 5.0
    max_steps: int = 120
    device: str = "cuda"
    fp16: bool = True
    seed: int = 0
    save_dir: str = "./artifacts/robot_eval"
    show_camera_preview: bool = False
    preview_layout: str = "side_by_side"
    preview_show_labels: bool = True
    preview_show_timestamps: bool = True
    preview_max_height: int = 360
    visualize_mock_3d: bool = False
    save_mock_plot_path: Optional[str] = None
    terminate_on_truncation: bool = True
    octo_preprocess_observation: bool = False
    disable_proprio: bool = False
    octo_resize_images: bool = False
    octo_resize_avg_scale: float = 0.9
    octo_resize_avg_ratio: float = 1.0
    enable_debug_logging: bool = True
    save_debug_csv_path: Optional[str] = None
    save_debug_plot_path: Optional[str] = None

    def __post_init__(self) -> None:
        if self.preview_layout not in PREVIEW_LAYOUTS:
            raise ValueError(
                f"preview_layout must be one of {sorted(PREVIEW_LAYOUTS)}, got {self.preview_layout!r}"
            )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RolloutConfig":
        kwargs = dict(data)
        if "preview_layout" not in kwargs:
            kwargs["preview_layout"] = "side_by_side"
        if "preview_show_labels" not in kwargs:
            kwargs["preview_show_labels"] = True
        if "preview_show_timestamps" not in kwargs:
            kwargs["preview_show_timestamps"] = True
        return cls(**kwargs)


@dataclass
class RobotPipelineConfig:
    robot: RobotConfig
    cameras: Dict[str, CameraConfig]
    rollout: RolloutConfig

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RobotPipelineConfig":
        robot = RobotConfig.from_dict(data["robot"])
        cameras = {
            name: CameraConfig.from_dict(name, cam_cfg)
            for name, cam_cfg in data.get("cameras", {}).items()
        }
        rollout = RolloutConfig.from_dict(data.get("rollout", {}))
        return cls(robot=robot, cameras=cameras, rollout=rollout)

    @classmethod
    def from_json(cls, path: str) -> "RobotPipelineConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "robot": {
                **self.robot.__dict__,
                "workspace": self.robot.workspace.__dict__,
            },
            "cameras": {name: cfg.__dict__ for name, cfg in self.cameras.items()},
            "rollout": self.rollout.__dict__,
        }

    def save_json(self, path: str) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

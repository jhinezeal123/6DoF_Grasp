from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np


class RobotDebugLogger:
    def __init__(self):
        self.records: List[Dict[str, object]] = []

    @staticmethod
    def _flatten_transform(prefix: str, transform: Optional[np.ndarray]) -> Dict[str, float]:
        if transform is None:
            return {f"{prefix}_{r}{c}": np.nan for r in range(4) for c in range(4)}
        tf = np.asarray(transform, dtype=np.float64).reshape(4, 4)
        return {f"{prefix}_{r}{c}": float(tf[r, c]) for r in range(4) for c in range(4)}

    @staticmethod
    def _pose_columns(prefix: str, pose: Optional[Iterable[float]]) -> Dict[str, float]:
        names = ["x_m", "y_m", "z_m", "rx_rad", "ry_rad", "rz_rad"]
        if pose is None:
            return {f"{prefix}_{name}": np.nan for name in names}
        arr = np.asarray(list(pose), dtype=np.float64).reshape(6)
        return {f"{prefix}_{name}": float(arr[i]) for i, name in enumerate(names)}

    @staticmethod
    def _xyz_columns(prefix: str, xyz: Optional[Iterable[float]]) -> Dict[str, float]:
        names = ["x_m", "y_m", "z_m"]
        if xyz is None:
            return {f"{prefix}_{name}": np.nan for name in names}
        arr = np.asarray(list(xyz), dtype=np.float64).reshape(3)
        return {f"{prefix}_{name}": float(arr[i]) for i, name in enumerate(names)}

    def log_step(
        self,
        *,
        step: int,
        infer_ms: float,
        robot_debug: Dict[str, object],
        camera_meta: Optional[Dict[str, dict]] = None,
    ) -> None:
        target_after = robot_debug.get("target_pose_after_step_m_rad", robot_debug.get("target_pose_m_rad"))
        target_after_rel = robot_debug.get("target_pose_after_step_rel_m_rad", robot_debug.get("target_pose_rel_m_rad"))
        measured_after = robot_debug.get("measured_pose_after_step_m_rad", robot_debug.get("measured_pose_m_rad"))
        measured_after_rel = robot_debug.get("measured_pose_after_step_rel_m_rad", robot_debug.get("measured_pose_rel_m_rad"))

        record: Dict[str, object] = {
            "step": int(step),
            "infer_ms": float(infer_ms),
            "translation_error_m": float(robot_debug.get("translation_error_m", np.nan)),
            "rotation_error_rad": float(robot_debug.get("rotation_error_rad", np.nan)),
            "translation_error_norm_m": float(robot_debug.get("translation_error_norm_m", robot_debug.get("translation_error_m", np.nan))),
            "rotation_error_norm_rad": float(robot_debug.get("rotation_error_norm_rad", robot_debug.get("rotation_error_rad", np.nan))),
            "gripper_open": float(robot_debug.get("gripper_open", np.nan)),
            "resync_applied": bool(robot_debug.get("resync_applied", False)),
            "resync_reason": robot_debug.get("resync_reason", ""),
            "integration_mode": robot_debug.get("integration_mode", ""),
            "pivot_source": robot_debug.get("pivot_source", ""),
            "compose_base": robot_debug.get("compose_base", ""),
        }
        record.update(self._pose_columns("target_before_abs", robot_debug.get("target_pose_before_step_m_rad")))
        record.update(self._pose_columns("target_after_abs", target_after))
        record.update(self._pose_columns("target_before_rel", robot_debug.get("target_pose_before_step_rel_m_rad")))
        record.update(self._pose_columns("target_after_rel", target_after_rel))
        record.update(self._pose_columns("measured_before_abs", robot_debug.get("measured_pose_before_step_m_rad")))
        record.update(self._pose_columns("measured_after_abs", measured_after))
        record.update(self._pose_columns("measured_before_rel", robot_debug.get("measured_pose_before_step_rel_m_rad")))
        record.update(self._pose_columns("measured_after_rel", measured_after_rel))
        # Backward-compatible aliases expected by existing plots.
        record.update(self._pose_columns("target_abs", target_after))
        record.update(self._pose_columns("target_rel", target_after_rel))
        record.update(self._pose_columns("measured_abs", measured_after))
        record.update(self._pose_columns("measured_rel", measured_after_rel))
        record.update(self._pose_columns("hardware_abs", robot_debug.get("hardware_pose_m_rad")))
        record.update(self._xyz_columns("pivot_position", robot_debug.get("pivot_position_m")))
        record.update(self._flatten_transform("target_tf_before", robot_debug.get("target_transform_before_step")))
        record.update(self._flatten_transform("target_tf_after", robot_debug.get("target_transform_after_step", robot_debug.get("target_transform"))))
        record.update(self._flatten_transform("target_tf", robot_debug.get("target_transform")))
        record.update(self._flatten_transform("compose_base_tf", robot_debug.get("compose_base_transform")))
        if camera_meta:
            for name, meta in camera_meta.items():
                record[f"camera_{name}_timestamp"] = float(meta.get("timestamp", np.nan))
                record[f"camera_{name}_age_s"] = float(meta.get("age_s", np.nan))
        self.records.append(record)

    def save_csv(self, path: str) -> None:
        if not self.records:
            return
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        keys = []
        for rec in self.records:
            for key in rec.keys():
                if key not in keys:
                    keys.append(key)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.records)

    def save_plot(self, path: str) -> None:
        if not self.records:
            return
        try:
            import matplotlib.pyplot as plt
        except Exception:
            return
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        steps = np.asarray([rec["step"] for rec in self.records], dtype=np.float64)
        fig = plt.figure(figsize=(12, 9))

        ax1 = fig.add_subplot(311)
        for axis, name in zip(range(3), ["x", "y", "z"]):
            ax1.plot(steps, [rec[f"target_after_abs_{name}_m"] for rec in self.records], label=f"target {name}")
            ax1.plot(steps, [rec[f"measured_after_abs_{name}_m"] for rec in self.records], linestyle="--", label=f"measured {name}")
        ax1.set_ylabel("translation [m]")
        ax1.set_title("Target vs measured TCP pose")
        ax1.legend(ncol=3, fontsize=8)
        ax1.grid(True)

        ax2 = fig.add_subplot(312)
        for axis_name in ["rx", "ry", "rz"]:
            ax2.plot(steps, [rec[f"target_after_rel_{axis_name}_rad"] for rec in self.records], label=f"target {axis_name}")
            ax2.plot(steps, [rec[f"measured_after_rel_{axis_name}_rad"] for rec in self.records], linestyle="--", label=f"measured {axis_name}")
        ax2.set_ylabel("relative orientation [rad]")
        ax2.set_title("Bridge-like relative orientation vs neutral")
        ax2.legend(ncol=3, fontsize=8)
        ax2.grid(True)

        ax3 = fig.add_subplot(313)
        ax3.plot(steps, [rec["translation_error_m"] for rec in self.records], label="translation error")
        ax3.plot(steps, [rec["rotation_error_rad"] for rec in self.records], label="rotation error")
        for name in {k[len("camera_"):-len("_age_s")] for rec in self.records for k in rec.keys() if k.startswith("camera_") and k.endswith("_age_s")}:
            ax3.plot(steps, [rec.get(f"camera_{name}_age_s", np.nan) for rec in self.records], label=f"camera {name} age")
        ax3.set_xlabel("step")
        ax3.set_ylabel("error")
        ax3.set_title("Tracking error and camera freshness")
        ax3.legend(ncol=3, fontsize=8)
        ax3.grid(True)

        fig.tight_layout()
        fig.savefig(out_path, dpi=160)
        plt.close(fig)

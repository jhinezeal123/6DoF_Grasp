from __future__ import annotations

from typing import Optional

import numpy as np

from octo.robot.config import WorkspaceBounds
from octo.robot.utils import rpy_to_rotation_matrix


class Trajectory3DVisualizer:
    def __init__(self, workspace: WorkspaceBounds, interactive: bool = False):
        try:
            import matplotlib.pyplot as plt  # noqa
            from mpl_toolkits.mplot3d import Axes3D  # noqa
        except Exception as e:
            raise ImportError(
                "matplotlib is required for mock 3D visualization."
            ) from e
        self._plt = plt
        self.workspace = workspace
        self.interactive = interactive
        self.fig = plt.figure(figsize=(7, 6))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.path = []
        self._setup_axes()
        if self.interactive:
            plt.ion()
            plt.show(block=False)

    def _setup_axes(self):
        ax = self.ax
        ws = self.workspace
        ax.set_title("Mock MyArm TCP trajectory (base frame)")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        ax.set_xlim(ws.translation_min_m[0], ws.translation_max_m[0])
        ax.set_ylim(ws.translation_min_m[1], ws.translation_max_m[1])
        ax.set_zlim(ws.translation_min_m[2], ws.translation_max_m[2])
        self._draw_workspace_box()
        self._draw_base_axes(np.zeros(3, dtype=np.float64), 0.05)

    def _draw_workspace_box(self):
        ws = self.workspace
        x0, y0, z0 = ws.translation_min_m
        x1, y1, z1 = ws.translation_max_m
        corners = np.array(
            [
                [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
            ],
            dtype=np.float64,
        )
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        for i, j in edges:
            xs = [corners[i, 0], corners[j, 0]]
            ys = [corners[i, 1], corners[j, 1]]
            zs = [corners[i, 2], corners[j, 2]]
            self.ax.plot(xs, ys, zs, linestyle="--", linewidth=0.8)

    def _draw_base_axes(self, origin: np.ndarray, scale: float):
        self.ax.quiver(origin[0], origin[1], origin[2], scale, 0, 0, arrow_length_ratio=0.2)
        self.ax.quiver(origin[0], origin[1], origin[2], 0, scale, 0, arrow_length_ratio=0.2)
        self.ax.quiver(origin[0], origin[1], origin[2], 0, 0, scale, arrow_length_ratio=0.2)

    def update(self, pose_m_rad: np.ndarray, home_pose_m_rad: Optional[np.ndarray] = None) -> None:
        pose = np.asarray(pose_m_rad, dtype=np.float64)
        self.path.append(pose[:3].copy())
        self.ax.cla()
        self._setup_axes()
        if home_pose_m_rad is not None:
            self._draw_pose_axes(np.asarray(home_pose_m_rad, dtype=np.float64), axis_scale=0.03, label="home")
        if len(self.path) > 1:
            pts = np.stack(self.path, axis=0)
            self.ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], linewidth=2.0)
        self._draw_pose_axes(pose, axis_scale=0.04, label="tcp")
        if self.interactive:
            self._plt.pause(0.001)

    def _draw_pose_axes(self, pose_m_rad: np.ndarray, axis_scale: float, label: str):
        origin = pose_m_rad[:3]
        rot = rpy_to_rotation_matrix(*pose_m_rad[3:6])
        axes = rot @ np.eye(3)
        for i in range(3):
            direction = axes[:, i] * axis_scale
            self.ax.quiver(
                origin[0], origin[1], origin[2],
                direction[0], direction[1], direction[2],
                arrow_length_ratio=0.2,
            )
        self.ax.text(origin[0], origin[1], origin[2], label)

    def save(self, path: str) -> None:
        self.fig.tight_layout()
        self.fig.savefig(path, dpi=150)

"""Small Unix-socket client for a separate pipeline_grasppose worker process.

This module implements the pinned worker JSON protocol without importing the
pipeline package or its JetPack/TensorRT dependencies into the 6DoF process.
"""

from __future__ import annotations

import json
import os
import socket
import tempfile
from dataclasses import dataclass
from typing import Any

import numpy as np


_MAX_RESPONSE_BYTES = 1024 * 1024


class WorkerError(RuntimeError):
    """The inference worker could not return a valid grasp estimate."""


@dataclass(frozen=True)
class _GraspPose:
    score: float
    width_m: float
    translation_m: tuple[float, float, float]
    rotation: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True)
class _EstimateResult:
    grasps: tuple[_GraspPose, ...]
    depth_m: float | None
    detection_count: int
    mask_pixels: int
    grasp_count: int
    request_id: str | None
    snapshot_available: bool
    latency_ms: float | None


def _camera_values(camera_K: Any) -> list[float] | None:
    if camera_K is None:
        return None
    values = np.asarray(camera_K, dtype=np.float64)
    if values.size == 4:
        return values.reshape(4).tolist()
    matrix = values.reshape(3, 3)
    return [
        float(matrix[0, 0]),
        float(matrix[1, 1]),
        float(matrix[0, 2]),
        float(matrix[1, 2]),
    ]


def _image_path(image: Any) -> tuple[str, str | None]:
    if isinstance(image, (str, os.PathLike)):
        return os.path.abspath(os.fspath(image)), None

    from PIL import Image

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError("input image must have shape (H, W, 3+)")
    with tempfile.NamedTemporaryFile(prefix="m750-grasp-", suffix=".png", delete=False) as temp:
        path = temp.name
    try:
        Image.fromarray(array[:, :, :3].astype(np.uint8)).save(path, format="PNG")
    except Exception:
        os.unlink(path)
        raise
    return path, path


def _parse_estimate(response: dict[str, Any]) -> _EstimateResult:
    try:
        grasps = []
        for item in response["grasps"]:
            translation = tuple(float(value) for value in item["translation_m"])
            rotation = tuple(
                tuple(float(value) for value in row) for row in item["rotation"]
            )
            if len(translation) != 3 or len(rotation) != 3 or any(len(row) != 3 for row in rotation):
                raise ValueError("invalid grasp pose shape")
            grasps.append(_GraspPose(
                score=float(item["score"]),
                width_m=float(item["width_m"]),
                translation_m=translation,
                rotation=rotation,
            ))
        depth = response.get("depth_m")
        latency = response.get("server_ms")
        return _EstimateResult(
            grasps=tuple(grasps),
            depth_m=None if depth is None else float(depth),
            detection_count=int(response.get("detection_count", 0)),
            mask_pixels=int(response.get("mask_pixels", 0)),
            grasp_count=int(response.get("grasp_count", len(grasps))),
            request_id=response.get("run_id"),
            snapshot_available=bool(response.get("snapshot_available", False)),
            latency_ms=None if latency is None else float(latency),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise WorkerError("invalid inference worker response") from exc


class WorkerGraspEstimator:
    """GraspEstimator-shaped client for the local pipeline worker.

    The worker and this process share a host, so an image path or temporary PNG
    is passed over a local Unix socket. Closing this client never stops the
    shared worker.
    """

    def __init__(self, socket_path: str | os.PathLike[str] | None = None, timeout: float = 300.0):
        if not hasattr(socket, "AF_UNIX"):
            raise WorkerError("Unix sockets are unavailable on this host")
        configured = socket_path or os.environ.get("GRASP_WORKER_SOCKET")
        if not configured:
            raise ValueError("set GRASP_WORKER_SOCKET or pass socket_path")
        self.socket_path = os.path.abspath(os.fspath(configured))
        self.timeout = float(timeout)
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.settimeout(self.timeout)
                client.connect(self.socket_path)
                client.sendall((json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8"))
                with client.makefile("rb") as stream:
                    raw = stream.readline(_MAX_RESPONSE_BYTES + 1)
        except FileNotFoundError as exc:
            raise WorkerError("pipeline worker socket not found: " + self.socket_path) from exc
        except socket.timeout as exc:
            raise WorkerError("timed out waiting for pipeline worker") from exc
        except OSError as exc:
            raise WorkerError("cannot connect to pipeline worker: " + str(exc)) from exc
        if not raw or len(raw) > _MAX_RESPONSE_BYTES or not raw.endswith(b"\n"):
            raise WorkerError("invalid or incomplete pipeline worker response")
        try:
            response = json.loads(raw)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise WorkerError("invalid JSON from pipeline worker") from exc
        if not isinstance(response, dict):
            raise WorkerError("invalid pipeline worker response")
        if not response.get("ok"):
            raise WorkerError(str(response.get("error") or "pipeline worker failed"))
        return response

    def load(self) -> None:
        """Check that the separately managed pipeline worker is ready."""

        self._request({"op": "status"})

    def estimate(
        self,
        image: Any,
        prompt_id: str,
        camera_K: Any = None,
        fov_x: float | None = None,
        fov_y: float | None = None,
        camera_K_size: Any = None,
        max_width: float = 0.080,
        top: int = 1,
        T_cam_volume: Any = None,
    ) -> _EstimateResult:
        path, cleanup = _image_path(image)
        try:
            response = self._request({
                "op": "infer",
                "image": path,
                "prompt_id": str(prompt_id),
                "camera_k": _camera_values(camera_K),
                "camera_k_size": (
                    None if camera_K_size is None
                    else np.asarray(camera_K_size).reshape(2).tolist()
                ),
                "fov_x": fov_x,
                "fov_y": fov_y,
                "render": False,
                "output_dir": None,
                "top": int(top),
                "max_width": float(max_width),
                "T_cam_volume": (
                    None if T_cam_volume is None
                    else np.asarray(T_cam_volume, dtype=np.float64).reshape(4, 4).tolist()
                ),
            })
        finally:
            if cleanup is not None:
                try:
                    os.unlink(cleanup)
                except FileNotFoundError:
                    pass
        return _parse_estimate(response)

    def close(self) -> None:
        """The pipeline worker has its own lifecycle."""


__all__ = ["WorkerError", "WorkerGraspEstimator"]

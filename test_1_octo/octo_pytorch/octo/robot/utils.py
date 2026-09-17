from __future__ import annotations

import math
from typing import Dict, Iterable, Tuple

import numpy as np

from octo.robot.config import WorkspaceBounds


def mm_to_m(value):
    return np.asarray(value, dtype=np.float64) / 1000.0


def m_to_mm(value):
    return np.asarray(value, dtype=np.float64) * 1000.0


def deg_to_rad(value):
    return np.deg2rad(np.asarray(value, dtype=np.float64))


def rad_to_deg(value):
    return np.rad2deg(np.asarray(value, dtype=np.float64))


def wrap_to_pi(value: Iterable[float] | float) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    return (arr + np.pi) % (2.0 * np.pi) - np.pi


def pose_mm_deg_to_m_rad(coords: Iterable[float]) -> np.ndarray:
    coords = np.asarray(list(coords), dtype=np.float64)
    if coords.shape[0] != 6:
        raise ValueError(f"Expected 6D pose, got shape {coords.shape}")
    xyz_m = mm_to_m(coords[:3])
    rpy_rad = deg_to_rad(coords[3:6])
    return np.concatenate([xyz_m, rpy_rad], axis=0)


def pose_m_rad_to_mm_deg(coords: Iterable[float]) -> np.ndarray:
    coords = np.asarray(list(coords), dtype=np.float64)
    if coords.shape[0] != 6:
        raise ValueError(f"Expected 6D pose, got shape {coords.shape}")
    xyz_mm = m_to_mm(coords[:3])
    rpy_deg = rad_to_deg(coords[3:6])
    return np.concatenate([xyz_mm, rpy_deg], axis=0)


def clamp_pose_to_workspace(pose_m_rad: np.ndarray, workspace: WorkspaceBounds) -> np.ndarray:
    pose = np.asarray(pose_m_rad, dtype=np.float64).copy()
    pose[:3] = np.clip(
        pose[:3],
        np.asarray(workspace.translation_min_m, dtype=np.float64),
        np.asarray(workspace.translation_max_m, dtype=np.float64),
    )
    pose[3:6] = np.clip(
        pose[3:6],
        np.asarray(workspace.rotation_min_rad, dtype=np.float64),
        np.asarray(workspace.rotation_max_rad, dtype=np.float64),
    )
    return pose


def clamp_transform_translation_to_workspace(transform: np.ndarray, workspace: WorkspaceBounds) -> np.ndarray:
    tf = np.asarray(transform, dtype=np.float64).copy()
    if tf.shape != (4, 4):
        raise ValueError(f"Expected 4x4 transform, got {tf.shape}")
    tf[:3, 3] = np.clip(
        tf[:3, 3],
        np.asarray(workspace.translation_min_m, dtype=np.float64),
        np.asarray(workspace.translation_max_m, dtype=np.float64),
    )
    return tf


def clip_delta_action(delta_pose: np.ndarray, max_translation_m: float, max_rotation_rad: float) -> np.ndarray:
    delta_pose = np.asarray(delta_pose, dtype=np.float64).copy()
    delta_pose[:3] = np.clip(delta_pose[:3], -max_translation_m, max_translation_m)
    delta_pose[3:6] = np.clip(delta_pose[3:6], -max_rotation_rad, max_rotation_rad)
    return delta_pose


def rpy_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def rotation_matrix_to_rpy(rotation: np.ndarray) -> np.ndarray:
    r = np.asarray(rotation, dtype=np.float64)
    if r.shape != (3, 3):
        raise ValueError(f"Expected 3x3 rotation matrix, got {r.shape}")

    sy = math.sqrt(r[0, 0] * r[0, 0] + r[1, 0] * r[1, 0])
    singular = sy < 1e-6

    if not singular:
        roll = math.atan2(r[2, 1], r[2, 2])
        pitch = math.atan2(-r[2, 0], sy)
        yaw = math.atan2(r[1, 0], r[0, 0])
    else:
        roll = math.atan2(-r[1, 2], r[1, 1])
        pitch = math.atan2(-r[2, 0], sy)
        yaw = 0.0
    return np.asarray([roll, pitch, yaw], dtype=np.float64)


def select_rpy_near_reference(rpy_rad: Iterable[float], reference_rpy_rad: Iterable[float]) -> np.ndarray:
    """Choose an equivalent ZYX (Rz@Ry@Rx) Euler representation close to a reference.

    With ZYX Euler angles, multiple (roll, pitch, yaw) triplets can represent the
    same 3D rotation. Some robot firmwares (and also matrix->Euler conversions)
    may jump between equivalent branches (notably around pitch≈±pi), which makes
    downstream clamping on raw Euler values look like it "snaps" to bounds.

    This helper picks between two common equivalent representations and returns
    the one closest (in wrapped angle distance) to the provided reference.
    """

    rpy = np.asarray(list(rpy_rad), dtype=np.float64).reshape(3)
    ref = np.asarray(list(reference_rpy_rad), dtype=np.float64).reshape(3)

    cand_a = wrap_to_pi(rpy)
    # Equivalent representation for R = Rz(y) @ Ry(p) @ Rx(r):
    # (r, p, y) and (r+pi, pi-p, y+pi) produce the same rotation.
    cand_b = wrap_to_pi(np.asarray([rpy[0] + np.pi, np.pi - rpy[1], rpy[2] + np.pi], dtype=np.float64))

    def dist2(candidate: np.ndarray) -> float:
        d = wrap_to_pi(candidate - ref)
        return float(np.dot(d, d))

    return cand_a if dist2(cand_a) <= dist2(cand_b) else cand_b


def make_transform(rotation: np.ndarray, translation: Iterable[float]) -> np.ndarray:
    rot = np.asarray(rotation, dtype=np.float64)
    xyz = np.asarray(translation, dtype=np.float64).reshape(3)
    if rot.shape != (3, 3):
        raise ValueError(f"Expected 3x3 rotation matrix, got {rot.shape}")
    tf = np.eye(4, dtype=np.float64)
    tf[:3, :3] = rot
    tf[:3, 3] = xyz
    return tf


def invert_transform(transform: np.ndarray) -> np.ndarray:
    tf = np.asarray(transform, dtype=np.float64)
    if tf.shape != (4, 4):
        raise ValueError(f"Expected 4x4 transform, got {tf.shape}")
    rot = tf[:3, :3]
    xyz = tf[:3, 3]
    rot_t = rot.T
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rot_t
    out[:3, 3] = -rot_t @ xyz
    return out


def pose_to_transform(pose_m_rad: Iterable[float]) -> np.ndarray:
    pose = np.asarray(list(pose_m_rad), dtype=np.float64)
    if pose.shape[0] != 6:
        raise ValueError(f"Expected 6D pose, got shape {pose.shape}")
    return make_transform(rpy_to_rotation_matrix(*pose[3:6]), pose[:3])


def transform_to_pose(transform: np.ndarray) -> np.ndarray:
    tf = np.asarray(transform, dtype=np.float64)
    if tf.shape != (4, 4):
        raise ValueError(f"Expected 4x4 transform, got {tf.shape}")
    xyz = tf[:3, 3]
    rpy = rotation_matrix_to_rpy(tf[:3, :3])
    return np.concatenate([xyz, wrap_to_pi(rpy)], axis=0)


def get_neutral_orientation_rpy(neutral_orientation_rpy_rad, home_pose_m_rad) -> np.ndarray:
    if neutral_orientation_rpy_rad is not None:
        return wrap_to_pi(np.asarray(neutral_orientation_rpy_rad, dtype=np.float64).reshape(3))
    home = np.asarray(home_pose_m_rad, dtype=np.float64).reshape(6)
    return wrap_to_pi(home[3:6])


def absolute_pose_to_relative_neutral_pose(
    pose_m_rad: Iterable[float],
    neutral_orientation_rpy_rad: Iterable[float],
) -> np.ndarray:
    pose = np.asarray(list(pose_m_rad), dtype=np.float64).reshape(6)
    neutral_rpy = np.asarray(list(neutral_orientation_rpy_rad), dtype=np.float64).reshape(3)
    rot_abs = rpy_to_rotation_matrix(*pose[3:6])
    rot_neutral = rpy_to_rotation_matrix(*neutral_rpy)
    rot_rel = rot_abs @ rot_neutral.T
    rel_rpy = wrap_to_pi(rotation_matrix_to_rpy(rot_rel))
    return np.concatenate([pose[:3], rel_rpy], axis=0)


def relative_neutral_pose_to_absolute_pose(
    pose_xyz_rel_rpy: Iterable[float],
    neutral_orientation_rpy_rad: Iterable[float],
) -> np.ndarray:
    pose = np.asarray(list(pose_xyz_rel_rpy), dtype=np.float64).reshape(6)
    neutral_rpy = np.asarray(list(neutral_orientation_rpy_rad), dtype=np.float64).reshape(3)
    rot_rel = rpy_to_rotation_matrix(*pose[3:6])
    rot_neutral = rpy_to_rotation_matrix(*neutral_rpy)
    rot_abs = rot_rel @ rot_neutral
    abs_rpy = wrap_to_pi(rotation_matrix_to_rpy(rot_abs))
    return np.concatenate([pose[:3], abs_rpy], axis=0)


def rotation_angle_between_matrices(rot_a: np.ndarray, rot_b: np.ndarray) -> float:
    rel = np.asarray(rot_a, dtype=np.float64) @ np.asarray(rot_b, dtype=np.float64).T
    trace = float(np.trace(rel))
    cos_theta = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return float(math.acos(cos_theta))


def pose_error_metrics(target_pose_m_rad: Iterable[float], measured_pose_m_rad: Iterable[float]) -> Dict[str, float]:
    target = np.asarray(list(target_pose_m_rad), dtype=np.float64).reshape(6)
    measured = np.asarray(list(measured_pose_m_rad), dtype=np.float64).reshape(6)
    translation_err = float(np.linalg.norm(target[:3] - measured[:3]))
    rotation_err = rotation_angle_between_matrices(
        rpy_to_rotation_matrix(*target[3:6]),
        rpy_to_rotation_matrix(*measured[3:6]),
    )
    return {
        "translation_m": translation_err,
        "rotation_rad": rotation_err,
    }


def action_to_bridge_delta_transform(delta_pose_m_rad: Iterable[float], current_eef_position_m: Iterable[float]) -> np.ndarray:
    """Bridge-style delta pose integration.

    The delta translation is expressed in base/world axes. The delta rotation is applied
    around the *current* end-effector position while keeping world/base rotation axes,
    matching `action2transform_local` in Bridge Data Robot.
    """

    action = np.asarray(list(delta_pose_m_rad), dtype=np.float64)
    if action.shape[0] != 6:
        raise ValueError(f"Expected 6D delta pose, got shape {action.shape}")
    pivot_xyz = np.asarray(list(current_eef_position_m), dtype=np.float64).reshape(3)

    teef = make_transform(np.eye(3, dtype=np.float64), pivot_xyz)
    local = make_transform(rpy_to_rotation_matrix(*action[3:6]), action[:3])
    return teef @ local @ invert_transform(teef)


def make_zero_image(height: int, width: int) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def center_crop_resize(image_rgb: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    import cv2

    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"Expected HWC RGB image, got {image_rgb.shape}")

    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    src_h, src_w = image_rgb.shape[:2]
    src_aspect = src_w / max(src_h, 1)
    dst_aspect = target_w / max(target_h, 1)

    if src_aspect > dst_aspect:
        new_w = int(round(src_h * dst_aspect))
        x0 = max((src_w - new_w) // 2, 0)
        cropped = image_rgb[:, x0 : x0 + new_w]
    else:
        new_h = int(round(src_w / dst_aspect))
        y0 = max((src_h - new_h) // 2, 0)
        cropped = image_rgb[y0 : y0 + new_h, :]

    interp = getattr(cv2, "INTER_LANCZOS4", cv2.INTER_LINEAR)
    resized = cv2.resize(cropped, (target_w, target_h), interpolation=interp)
    return resized.astype(np.uint8, copy=False)


def ensure_hw3_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected HWC uint8 image, got {arr.shape}")
    return arr


def apply_image_transform(
    image_rgb: np.ndarray,
    *,
    crop_top: int = 0,
    crop_bottom: int = 0,
    crop_left: int = 0,
    crop_right: int = 0,
    flip_horizontal: bool = False,
    flip_vertical: bool = False,
    rotation_deg: int = 0,
) -> np.ndarray:
    arr = ensure_hw3_uint8(image_rgb)
    h, w = arr.shape[:2]

    top = max(int(crop_top), 0)
    bottom = max(int(crop_bottom), 0)
    left = max(int(crop_left), 0)
    right = max(int(crop_right), 0)
    y1 = min(top, h)
    y2 = max(y1, h - bottom)
    x1 = min(left, w)
    x2 = max(x1, w - right)
    arr = arr[y1:y2, x1:x2]

    if flip_horizontal:
        arr = np.ascontiguousarray(arr[:, ::-1])
    if flip_vertical:
        arr = np.ascontiguousarray(arr[::-1, :])

    rot = int(rotation_deg) % 360
    if rot not in (0, 90, 180, 270):
        raise ValueError(f"rotation_deg must be one of 0, 90, 180, 270; got {rotation_deg}")
    if rot == 90:
        arr = np.ascontiguousarray(np.rot90(arr, k=1))
    elif rot == 180:
        arr = np.ascontiguousarray(np.rot90(arr, k=2))
    elif rot == 270:
        arr = np.ascontiguousarray(np.rot90(arr, k=3))
    return ensure_hw3_uint8(arr)


def summarize_pose(pose_m_rad: np.ndarray) -> Dict[str, float]:
    pose = np.asarray(pose_m_rad, dtype=np.float64)
    return {
        "x_m": float(pose[0]),
        "y_m": float(pose[1]),
        "z_m": float(pose[2]),
        "rx_rad": float(pose[3]),
        "ry_rad": float(pose[4]),
        "rz_rad": float(pose[5]),
    }

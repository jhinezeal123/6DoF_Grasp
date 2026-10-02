"""Camera and grasp geometry for simulation validation.

Pure NumPy/SciPy math with no MuJoCo import, so the transforms and the candidate
guards can be exercised without a renderer or an EGL platform library.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from ..perception.types import GraspCandidate
from .scenario import FOVY, HEIGHT, MAX_OPEN, MJ_TO_CV, VOLUME_SIZE_M, WIDTH
from .types import ValidationFailure


def tf(r,p):
    m=np.eye(4);m[:3,:3]=np.asarray(r).reshape(3,3);m[:3,3]=np.asarray(p).reshape(3);return m

def camera_optical_transform(position_world,rotation_world_mj):
    return tf(np.asarray(rotation_world_mj).reshape(3,3)@MJ_TO_CV,position_world)

def candidate_transform(c):
    try:p=np.asarray(c.position_m,dtype=float).reshape(3);q=np.asarray(c.quaternion_xyzw,dtype=float).reshape(4);w=float(c.width_m);s=float(c.score)
    except (AttributeError,TypeError,ValueError) as e:raise ValidationFailure("invalid grasp candidate") from e
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(q)) or not np.isfinite(s):raise ValidationFailure("non-finite grasp values")
    if not np.isfinite(w) or not 0.<w<=MAX_OPEN:raise ValidationFailure(f"grasp width {w!r} outside (0, 0.069]")
    n=np.linalg.norm(q)
    if n<1e-8:raise ValidationFailure("zero-norm grasp quaternion")
    r=Rotation.from_quat(q/n).as_matrix()
    if not np.allclose(r.T@r,np.eye(3),atol=1e-6) or abs(np.linalg.det(r)-1)>1e-6:raise ValidationFailure("invalid grasp rotation")
    return tf(r,p)

def camera_k(width=WIDTH,height=HEIGHT,fovy=FOVY):
    if width<=0 or height<=0 or not 0<fovy<180:raise ValueError("invalid camera")
    f=height/(2*np.tan(np.radians(fovy)/2))
    return np.array([[f,0,width/2],[0,f,height/2],[0,0,1.]])

def project_camera(p,k):
    p=np.asarray(p,dtype=float).reshape(3);k=np.asarray(k).reshape(3,3)
    if p[2]<=0:raise ValueError("point behind camera")
    return np.array([k[0,0]*p[0]/p[2]+k[0,2],k[1,1]*p[1]/p[2]+k[1,2]])

def cube_grasp_error_mm(T_base_grasp,cube_position_m,cube_quaternion_wxyz,side_m):
    """Where the predicted grasp point sits relative to the cube, in millimetres.

    The point is the translation of ``T_base_grasp``; it is moved into the cube's
    own axes with the cube orientation (MuJoCo ``wxyz`` free-joint quaternion), so
    a rotated cube reads exactly like an axis-aligned one, and compared with the
    half-extent ``side_m/2`` to get a per-axis overshoot ``d`` (negative where the
    axis is inside the cube).

    ``surface_mm`` is the distance to the nearest point of the cube surface: the
    L2 norm of the positive overshoots, i.e. 0 on a face and 7.5 for a point
    7.5 mm off one, and larger than ``depth_mm`` off a corner. Strictly inside
    the cube every axis is negative and there is no outward surface point to
    measure to, so the same norm is taken unclamped; it reads half the space
    diagonal at the centre and is discontinuous with the outside value at the
    boundary, so read ``surface_mm`` together with the sign of ``depth_mm``.

    ``depth_mm`` is the largest per-axis overshoot instead: positive outside,
    where it is the distance past the nearest face *plane* (equal to
    ``surface_mm`` on a face normal, smaller off a corner), and negative inside,
    where it is minus the distance to the closest face. A correct surface grasp
    lands near 0 on both.
    """
    p=np.asarray(T_base_grasp,dtype=float).reshape(4,4)[:3,3]-np.asarray(cube_position_m,dtype=float).reshape(3)
    r=Rotation.from_quat(np.asarray(cube_quaternion_wxyz,dtype=float).reshape(4)[[1,2,3,0]]).as_matrix()
    d=np.abs(r.T@p)-side_m/2.
    surface=np.linalg.norm(np.maximum(d,0.) if (d>=0).any() else d)
    return {"surface_mm":float(surface*1000),"depth_mm":float(d.max()*1000)}

def rotation_error_deg(a,b):
    r=np.asarray(a).reshape(3,3).T@np.asarray(b).reshape(3,3)
    return float(np.degrees(np.arccos(np.clip((np.trace(r)-1)/2,-1,1))))

def gravity_aligned_volume(T_base_camera,centre_base,size=VOLUME_SIZE_M):
    """Volume axes aligned with the robot base, i.e. Z pointing up along gravity.

    VGN was trained on gravity-aligned volumes; its own simulator builds the
    volume with ``Transform(Rotation.identity(), ...)`` and gravity along -Z.
    """
    T=np.asarray(T_base_camera,dtype=float).reshape(4,4)
    rotation=T[:3,:3].T
    volume=np.eye(4);volume[:3,:3]=rotation
    volume[:3,3]=rotation@(np.asarray(centre_base,dtype=float).reshape(3)-size/2.-T[:3,3])
    return volume

def oracle_candidate(position,width=.025):
    r=np.array([[0.,1.,0.],[1.,0.,0.],[0.,0.,-1.]])
    q=Rotation.from_matrix(r).as_quat()
    return GraspCandidate(1.,width,tuple(np.asarray(position,dtype=float)),tuple(q),{"source":"simulation_ground_truth"})


__all__ = [
    "camera_k",
    "camera_optical_transform",
    "candidate_transform",
    "cube_grasp_error_mm",
    "gravity_aligned_volume",
    "oracle_candidate",
    "project_camera",
    "rotation_error_deg",
    "tf",
]

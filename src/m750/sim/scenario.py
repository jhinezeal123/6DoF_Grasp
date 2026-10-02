"""The fixed MuJoCo validation scenario.

Every value here is a simulation assumption about a synthetic cell, not a
measurement of the real one. The MuJoCo adapter builds the world from these, and
the application layer reports them, so they live outside both.

The scene XML itself stays in ``m750/model`` and is not part of this package.
MuJoCo resolves ``meshdir`` relative to the directory of the *main* XML file and
then prefixes the included file's directory, so a scene that includes
``myarm_m750_mujoco.xml`` from anywhere else fails to find the meshes:

    Error opening file
    'src/m750/sim/assets/meshes/mujoco/src/m750/model/shoulder_link.obj'

The scene is a 32-line override of the robot model and has to sit beside it.
"""

from __future__ import annotations

import numpy as np

from ..spec import JOINT_NAMES, model_dir

ACTUATORS=("pos_j1","pos_j2","pos_j3","pos_j4","pos_j5","pos_j6")
CAMERA="wrist_cam"
CUBE=np.array([.30,.10,.1245])
# scene_grasp_validation.xml validation_cube_geom is a box with half-extents
# 0.0125, so the edge is 25 mm (18.75 g at that geom's 1200 kg/m^3 density).
CUBE_SIDE_M=.025
OFFSETS=((0.,0.),(.02,0.),(-.02,0.),(0.,.02),(0.,-.02))
LIGHTS=(.8,1.2)
CAMERA_Q_DEG=np.array([-37.66,-64.89,41.76,37.34,88.57,129.87])
FOVY=42.2
WIDTH,HEIGHT=1280,720
MAX_OPEN=.069
LIFT_HEIGHT_M=.065
SEED=20260930
SCENE=model_dir()/"scene_grasp_validation.xml"
MJ_TO_CV=np.diag([1.,-1.,-1.])
# Must match the pipeline's TSDF_SIZE_M; this repo cannot import grasppose
# because the two live in different virtualenvs. The bridge reports the value it
# actually used and the harness fails on a mismatch, so this cannot drift
# silently.
VOLUME_SIZE_M=.30

__all__ = [
    "ACTUATORS",
    "CAMERA",
    "CAMERA_Q_DEG",
    "CUBE",
    "CUBE_SIDE_M",
    "FOVY",
    "HEIGHT",
    "JOINT_NAMES",
    "LIFT_HEIGHT_M",
    "LIGHTS",
    "MAX_OPEN",
    "MJ_TO_CV",
    "OFFSETS",
    "SCENE",
    "SEED",
    "VOLUME_SIZE_M",
    "WIDTH",
]

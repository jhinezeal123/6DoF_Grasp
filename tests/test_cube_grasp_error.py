import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from m750.sim.geometry import cube_grasp_error_mm

# scene_grasp_validation.xml: box half-extents 0.0125 -> 25 mm cube.
SIDE=.025;HALF=SIDE/2;CUBE=np.array([.30,.10,.1245]);IDENTITY=[1.,0.,0.,0.]

def pose(offset):
    t=np.eye(4);t[:3,3]=CUBE+np.asarray(offset,dtype=float);return t

def test_point_on_a_face_is_zero():
    e=cube_grasp_error_mm(pose([HALF,0,0]),CUBE,IDENTITY,SIDE)
    assert e["surface_mm"]==pytest.approx(0.,abs=1e-9)
    assert e["depth_mm"]==pytest.approx(0.,abs=1e-9)

def test_point_7_5_mm_outside_a_face():
    e=cube_grasp_error_mm(pose([HALF+.0075,0,0]),CUBE,IDENTITY,SIDE)
    assert e["surface_mm"]==pytest.approx(7.5)
    assert e["depth_mm"]==pytest.approx(7.5)

def test_centre_is_half_the_space_diagonal_away():
    e=cube_grasp_error_mm(pose([0,0,0]),CUBE,IDENTITY,SIDE)
    assert e["surface_mm"]==pytest.approx(np.sqrt(3)*SIDE/2*1000.)
    assert e["depth_mm"]==pytest.approx(-HALF*1000.)

def test_off_corner_surface_is_euclidean_not_per_axis():
    # 3 mm past one face and 4 mm past another: 5 mm to the nearest surface point,
    # but the largest per-axis overshoot is only 4 mm.
    e=cube_grasp_error_mm(pose([HALF+.003,HALF+.004,0]),CUBE,IDENTITY,SIDE)
    assert e["surface_mm"]==pytest.approx(5.)
    assert e["depth_mm"]==pytest.approx(4.)

def test_identity_quaternion_matches_the_axis_aligned_path():
    aligned=cube_grasp_error_mm(pose([HALF+.0075,.004,-.003]),CUBE,IDENTITY,SIDE)
    rotated=cube_grasp_error_mm(pose([HALF+.0075,.004,-.003]),CUBE,Rotation.identity().as_quat()[[3,0,1,2]],SIDE)
    assert rotated==aligned

@pytest.mark.parametrize("q",[(1.,0.,0.,0.),Rotation.from_euler("zyx",[37.,-52.,11.]).as_quat()[[3,0,1,2]]])
def test_rotating_cube_and_point_together_changes_nothing(q):
    # Same geometry seen from a rotated cube frame: R@offset must read like offset.
    offset=np.array([HALF+.0075,.004,-.003])
    r=Rotation.from_quat(np.asarray(q)[[1,2,3,0]]).as_matrix()
    e=cube_grasp_error_mm(pose(r@offset),CUBE,list(q),SIDE)
    assert e["surface_mm"]==pytest.approx(7.5)
    assert e["depth_mm"]==pytest.approx(7.5)

def test_depth_sign_flips_inside_the_cube():
    outside=cube_grasp_error_mm(pose([HALF+.002,0,0]),CUBE,IDENTITY,SIDE)
    inside=cube_grasp_error_mm(pose([HALF-.002,0,0]),CUBE,IDENTITY,SIDE)
    assert outside["depth_mm"]>0>inside["depth_mm"]
    assert outside["depth_mm"]==pytest.approx(2.)
    assert inside["depth_mm"]==pytest.approx(-2.)

def test_inside_point_reads_the_unclamped_overlap_norm():
    # Documented wart: strictly inside there is no outward surface point, so
    # surface_mm is the unclamped norm, like the centre case above.
    e=cube_grasp_error_mm(pose([HALF-.002,0,0]),CUBE,IDENTITY,SIDE)
    # 2 mm under the +x face and 12.5 mm from each of the other two.
    assert e["surface_mm"]==pytest.approx(np.linalg.norm([2.,HALF*1000,HALF*1000]))
    assert e["surface_mm"]>0.

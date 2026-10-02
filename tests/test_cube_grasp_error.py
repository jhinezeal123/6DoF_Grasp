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

def test_centre_is_one_half_edge_from_the_nearest_face():
    # The nearest surface point from the centre is a face, 12.5 mm away -- not a
    # corner, which is the farthest.
    e=cube_grasp_error_mm(pose([0,0,0]),CUBE,IDENTITY,SIDE)
    assert e["surface_mm"]==pytest.approx(HALF*1000.)
    assert e["depth_mm"]==pytest.approx(-HALF*1000.)

def test_off_corner_surface_is_euclidean_not_per_axis():
    # 3 mm past one face and 4 mm past another: 5 mm to the nearest surface point.
    e=cube_grasp_error_mm(pose([HALF+.003,HALF+.004,0]),CUBE,IDENTITY,SIDE)
    assert e["surface_mm"]==pytest.approx(5.)
    assert e["depth_mm"]==pytest.approx(5.)

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

def test_signed_distance_is_continuous_across_a_face():
    # One signed value is the point of this function: 2 mm either side of a face
    # reads the same magnitude with opposite sign, and nothing jumps at zero.
    out=cube_grasp_error_mm(pose([HALF+.002,0,0]),CUBE,IDENTITY,SIDE)
    on =cube_grasp_error_mm(pose([HALF,0,0]),CUBE,IDENTITY,SIDE)
    ins=cube_grasp_error_mm(pose([HALF-.002,0,0]),CUBE,IDENTITY,SIDE)
    assert [e["depth_mm"] for e in (out,on,ins)]==pytest.approx([2.,0.,-2.],abs=1e-9)
    assert [e["surface_mm"] for e in (out,on,ins)]==pytest.approx([2.,0.,2.],abs=1e-9)
    # And well inside, the nearest face is what is measured, not the far side.
    deep=cube_grasp_error_mm(pose([0,0,0]),CUBE,IDENTITY,SIDE)
    assert deep["depth_mm"]==pytest.approx(-HALF*1000.)

# The real failure mode this metric was added to expose. A recorded e2e run put
# the grasp point at these offsets from the cube centre, in whole TSDF voxels
# (0.30 m / 40), i.e. on the voxel corners and never on a face -- the cube's
# faces fall between lattice sites. Values are the signed distances that run
# reported, recomputed here straight from the offsets.
VOXEL=.30/40
@pytest.mark.parametrize("voxels,expected_mm",[
    ((1,0,4),17.5),((-1,0,6),32.5),((0,0,5),25.0),((-2,-1,7),40.08),
])
def test_observed_lattice_points_reproduce_the_recorded_distances(voxels,expected_mm):
    e=cube_grasp_error_mm(pose(np.asarray(voxels,float)*VOXEL),CUBE,IDENTITY,SIDE)
    assert e["depth_mm"]==pytest.approx(expected_mm,abs=.01)
    assert e["surface_mm"]==pytest.approx(expected_mm,abs=.01)
    # Every one of them is outside the cube: the lattice cannot land on a face.
    assert e["depth_mm"]>0.

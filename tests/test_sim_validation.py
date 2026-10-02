import subprocess
import sys
import numpy as np
import pytest
from m750.perception.types import GraspCandidate
from m750.sim import ValidationFailure,camera_k,camera_optical_transform,candidate_transform,oracle_candidate,project_camera
from m750.sim.adapters.mujoco import ValidationWorld

def test_camera_axes_convert_to_right_handed_cv_optical():
    t=camera_optical_transform([1,2,3],np.eye(3))
    assert np.allclose(t[:3,3],[1,2,3])
    assert np.allclose(t[:3,:3],np.diag([1,-1,-1]))
    assert np.linalg.det(t[:3,:3])==pytest.approx(1.)

def test_intrinsics_projection_round_trip():
    k=camera_k(1280,720,42.2);p=np.array([.02,-.01,.3]);uv=project_camera(p,k)
    recovered=np.array([(uv[0]-k[0,2])*p[2]/k[0,0],(uv[1]-k[1,2])*p[2]/k[1,1],p[2]])
    assert np.linalg.norm(uv-project_camera(recovered,k))<2.
    assert k[0,0]==pytest.approx(k[1,1])

def test_oracle_pose_is_valid_and_proper():
    c=oracle_candidate([.3,.1,.1045]);t=candidate_transform(c)
    assert np.all(np.isfinite(t));assert np.linalg.det(t[:3,:3])==pytest.approx(1.)
    assert c.width_m==pytest.approx(.025)

@pytest.mark.parametrize("width",[0.,-.001,.0691,.080])
def test_oversized_or_empty_candidate_stops_before_motion(width):
    c=GraspCandidate(1.,width,(0.,0.,.3),(0.,0.,0.,1.),{})
    with pytest.raises(ValidationFailure,match="grasp width"):
        candidate_transform(c)

def test_zero_quaternion_is_rejected():
    c=GraspCandidate(1.,.025,(0.,0.,.3),(0.,0.,0.,0.),{})
    with pytest.raises(ValidationFailure,match="zero-norm"):
        candidate_transform(c)

def test_module_import_does_not_load_hardware_backend():
    # The GL backend is inherited rather than forced to EGL here: what this test
    # proves is that importing the MuJoCo world drags in neither the robot driver
    # nor ROS, and forcing EGL made it fail on every machine without libEGL.
    code=("import sys; import m750.sim.adapters.mujoco; import m750.sim.application; "
          "import m750.sim.cli; "
          "assert 'm750.robot.adapters.pymycobot' not in sys.modules; "
          "assert 'm750.ros' not in sys.modules")
    result=subprocess.run([sys.executable,"-c",code],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


@pytest.mark.parametrize("failure",["worker","width","ik","collision"])
def test_pre_motion_guard_failures_never_execute(monkeypatch,tmp_path,failure):
    from types import SimpleNamespace
    from pathlib import Path

    import m750.sim.application as runner

    class FakeWorld:
        last=None
        def __init__(self,brightness,video_path=None):
            FakeWorld.last=self
            self.qaddr=np.arange(6)
            self.cube_q=6
            self.data=SimpleNamespace(qpos=np.array([0.,0.,0.,0.,0.,0.,.3,.1,.1245,1.,0.,0.,0.]))
            self.K=np.eye(3)
            self.executed=False
        def reset(self,offset):pass
        def verify_fk(self):return {"max_position_error_mm":0.,"max_rotation_error_deg":0.}
        def verify_projection(self):return 0.
        def render(self):return np.zeros((8,8,3),dtype=np.uint8)
        def save_image(self,path,rgb):Path(path).parent.mkdir(parents=True,exist_ok=True);Path(path).write_bytes(b"image")
        def camera_pose(self):return np.eye(4)
        def start_video(self):pass
        def record(self):pass
        def make_plan(self,candidate):
            if failure=="width":
                runner.candidate_transform(candidate)
            message="IK found no solution" if failure=="ik" else "swept path contact: table"
            raise runner.ValidationFailure(message)
        def execute(self,plan):
            self.executed=True
            raise AssertionError("execution must not occur after a failed guard")
        def close_video(self):pass
        def close(self):pass

    monkeypatch.setattr(runner,"ValidationWorld",FakeWorld)
    candidate=SimpleNamespace(score=.9,width_m=.080 if failure=="width" else .025,
                              position_m=(0.,0.,.3),quaternion_xyzw=(0.,0.,0.,1.),metadata={})
    def fake_infer(*args,**kwargs):
        if failure=="worker":raise RuntimeError("worker offline")
        raw=SimpleNamespace(detection_count=1,mask_pixels=64)
        return SimpleNamespace(grasps=(candidate,),raw=raw)
    monkeypatch.setattr(runner,"infer",fake_infer)

    result=runner.one_case(1,(0.,0.),.8,"e2e",object(),tmp_path)
    assert result["success"] is False
    assert FakeWorld.last.executed is False
    if failure=="worker":
        assert result["error"]=="worker offline"
    else:
        attempt=result["perception"]["planning_attempts"][0]
        assert attempt["result"]=="rejected"
        if failure=="width":
            assert "grasp width" in attempt["error"]


def test_rendered_cube_center_reprojects_within_two_pixels():
    world=ValidationWorld()
    try:
        world.reset()
        assert world.verify_projection()<=2.
        initial=world.telemetry[-1]
        world.step(10,False)
        current=world.telemetry[-1]
        assert current["time_s"]>initial["time_s"]
        assert current["cube_height_m"]==pytest.approx(world.data.xpos[world.cube_body,2])
        assert len(current["q_rad"])==6
        assert isinstance(current["contacts"],list)
    finally:
        world.close()

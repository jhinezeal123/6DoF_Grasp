"""MuJoCo implementation of the validation world.

This module never constructs a physical robot driver: it reaches only for the
MuJoCo model, the kinematic solver and the camera geometry.
"""

from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL","egl")
from pathlib import Path
import mujoco, numpy as np, pinocchio as pin
from scipy.spatial.transform import Rotation
from ...ik import IKSolver
from ...kinematics import ArmKinematics
from ...perception.types import GraspCandidate
from ...spec import TOOL0_OFFSET_M, TOOL0_ROTATION
from ..geometry import (camera_optical_transform, candidate_transform,
    project_camera, rotation_error_deg, tf)
from ..scenario import (ACTUATORS, CAMERA, CAMERA_Q_DEG, CUBE, FOVY, HEIGHT,
    JOINT_NAMES, LIFT_HEIGHT_M, MAX_OPEN, SCENE, SEED, WIDTH)
from ..types import MotionPlan, ValidationFailure

# spec.py keeps these as plain tuples so a minimal install can import it; the
# offsets themselves are shared with robot/adapters/mujoco.py.
TOOL0_OFFSET=np.array(TOOL0_OFFSET_M,float);TOOL0_ROT=np.array(TOOL0_ROTATION,float)


def _id(m,kind,name):
    i=mujoco.mj_name2id(m,kind,name)
    if i<0:raise ValidationFailure(f"MuJoCo scene missing {name}")
    return int(i)

class ValidationWorld:
    """Dynamic MuJoCo episode controlled through mj_step and position actuators."""
    def __init__(self,brightness=1.,video_path=None):
        if brightness<=0:raise ValueError("brightness must be positive")
        self.model=mujoco.MjModel.from_xml_path(str(SCENE));self.data=mujoco.MjData(self.model);self.model.opt.timestep=.002
        self.cam=_id(self.model,mujoco.mjtObj.mjOBJ_CAMERA,CAMERA);self.model.cam_fovy[self.cam]=FOVY
        self.cube_body=_id(self.model,mujoco.mjtObj.mjOBJ_BODY,"validation_cube")
        self.cube_geom=_id(self.model,mujoco.mjtObj.mjOBJ_GEOM,"validation_cube_geom")
        self.ped_body=_id(self.model,mujoco.mjtObj.mjOBJ_BODY,"validation_pedestal")
        self.ped_geom=_id(self.model,mujoco.mjtObj.mjOBJ_GEOM,"validation_pedestal_geom")
        self.table_geom=_id(self.model,mujoco.mjtObj.mjOBJ_GEOM,"validation_table_top")
        self.gripper_body=_id(self.model,mujoco.mjtObj.mjOBJ_BODY,"gripper_base_link")
        self.left_geom=_id(self.model,mujoco.mjtObj.mjOBJ_GEOM,"left_finger_col")
        self.right_geom=_id(self.model,mujoco.mjtObj.mjOBJ_GEOM,"right_finger_col")
        self.joints=tuple(_id(self.model,mujoco.mjtObj.mjOBJ_JOINT,n) for n in JOINT_NAMES)
        self.qaddr=np.array([self.model.jnt_qposadr[j] for j in self.joints],int)
        self.daddr=np.array([self.model.jnt_dofadr[j] for j in self.joints],int)
        self.acts=tuple(_id(self.model,mujoco.mjtObj.mjOBJ_ACTUATOR,n) for n in ACTUATORS)
        self.gj=_id(self.model,mujoco.mjtObj.mjOBJ_JOINT,"left_gripper_joint");self.rgj=_id(self.model,mujoco.mjtObj.mjOBJ_JOINT,"right_gripper_joint")
        self.gq=int(self.model.jnt_qposadr[self.gj]);self.rgq=int(self.model.jnt_qposadr[self.rgj])
        self.ga=_id(self.model,mujoco.mjtObj.mjOBJ_ACTUATOR,"pos_left_gripper")
        cj=_id(self.model,mujoco.mjtObj.mjOBJ_JOINT,"validation_cube_free");self.cube_q=int(self.model.jnt_qposadr[cj])
        light=_id(self.model,mujoco.mjtObj.mjOBJ_LIGHT,"validation_light")
        self.model.light_diffuse[light]=np.array([.7,.7,.7])*brightness;self.model.light_ambient[light]=np.array([.15,.15,.15])*brightness
        self.kin=ArmKinematics();self.ik=IKSolver(self.kin,n_restart=6)
        self.renderer=mujoco.Renderer(self.model,height=HEIGHT,width=WIDTH)
        self.video_path=Path(video_path) if video_path else None;self.video=None;self.frames=0;self.steps=0;self.telemetry=[]

    @property
    def K(self):
        f=HEIGHT/(2*np.tan(np.radians(self.model.cam_fovy[self.cam])/2))
        return np.array([[f,0,WIDTH/2],[0,f,HEIGHT/2],[0,0,1.]])

    def reset(self,offset=(0.,0.)):
        x,y=CUBE[:2]+np.asarray(offset);self.model.body_pos[self.ped_body]=[x,y,.082]
        mujoco.mj_resetData(self.model,self.data);self.steps=0;self.telemetry=[]
        self.data.qpos[self.qaddr]=np.radians(CAMERA_Q_DEG)
        self.data.qpos[self.gq]=.0345;self.data.qpos[self.rgq]=.0345
        self.data.qpos[self.cube_q:self.cube_q+7]=[x,y,CUBE[2],1,0,0,0]
        self.data.ctrl[list(self.acts)]=np.radians(CAMERA_Q_DEG);self.data.ctrl[self.ga]=.0345
        mujoco.mj_forward(self.model,self.data);self.step(5,False);self.trace_state(force=True)

    def camera_pose(self):
        return camera_optical_transform(self.data.cam_xpos[self.cam],self.data.cam_xmat[self.cam].reshape(3,3))

    def render(self):
        self.renderer.update_scene(self.data,camera=CAMERA);return self.renderer.render().copy()

    def render_depth(self):
        """Axial camera-Z depth in metres, in the same OpenCV camera frame as K."""
        self.renderer.enable_depth_rendering()
        try:
            self.renderer.update_scene(self.data,camera=CAMERA)
            return self.renderer.render().copy()
        finally:
            self.renderer.disable_depth_rendering()

    def save_image(self,path,rgb):
        import cv2
        Path(path).parent.mkdir(parents=True,exist_ok=True);cv2.imwrite(str(path),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))

    def start_video(self):
        if self.video_path is None or self.video is not None:return
        import cv2
        self.video_path.parent.mkdir(parents=True,exist_ok=True)
        self.video=cv2.VideoWriter(str(self.video_path),cv2.VideoWriter_fourcc(*"mp4v"),20.,(WIDTH,HEIGHT))
        if not self.video.isOpened():self.video.release();self.video=None;raise ValidationFailure("cannot create episode video")
        self.record()

    def record(self):
        if self.video is not None:
            import cv2
            self.video.write(cv2.cvtColor(self.render(),cv2.COLOR_RGB2BGR));self.frames+=1

    def close_video(self):
        if self.video is not None:self.video.release();self.video=None

    def close(self):
        self.close_video();self.renderer.close()

    def _after_physics_step(self,record_video=True):
        if self.steps%15==0:
            self.trace_state()
            if record_video and self.video is not None:self.record()

    def step(self,n,record=True):
        for _ in range(n):
            mujoco.mj_step(self.model,self.data);self.steps+=1
            self._after_physics_step(record)

    def tool0(self):
        flange=_id(self.model,mujoco.mjtObj.mjOBJ_BODY,"flange_link");r=self.data.xmat[flange].reshape(3,3);p=self.data.xpos[flange]
        rt=r@TOOL0_ROT
        return tf(rt,p+r@TOOL0_OFFSET)

    def tool_to_grasp(self):
        saved=self.data.qpos.copy();self.data.qpos[self.qaddr]=0.;self.data.qpos[self.gq]=self.data.qpos[self.rgq]=.0345;mujoco.mj_forward(self.model,self.data)
        l=self.data.geom_xpos[self.left_geom].copy();r=self.data.geom_xpos[self.right_geom].copy();base=self.data.xpos[self.gripper_body]
        gy=l-r;gy/=np.linalg.norm(gy);gz=(l+r)/2-base;gz-=np.dot(gz,gy)*gy;gz/=np.linalg.norm(gz)
        gx=np.cross(gy,gz);gx/=np.linalg.norm(gx);bg=tf(np.column_stack((gx,gy,gz)),(l+r)/2)
        result=np.linalg.inv(self.tool0())@bg;self.data.qpos[:]=saved;mujoco.mj_forward(self.model,self.data);return result

    def _limits(self):
        urdf=np.radians(np.asarray(self.kin.limits_deg).T);mj=np.asarray([self.model.jnt_range[j] for j in self.joints])
        return np.column_stack((np.maximum(urdf[:,0],mj[:,0]),np.minimum(urdf[:,1],mj[:,1])))

    def verify_fk(self,n=20,seed=SEED):
        saved=self.data.qpos.copy();rng=np.random.default_rng(seed);lim=self._limits();ep=eo=0.
        for i in range(n):
            q=np.zeros(6) if i==0 else rng.uniform(lim[:,0],lim[:,1])
            self.data.qpos[self.qaddr]=q;mujoco.mj_forward(self.model,self.data);tmj=self.tool0();tp=self.kin.fk_tool0(np.degrees(q))
            ep=max(ep,float(np.linalg.norm(tmj[:3,3]-tp.translation)*1000));eo=max(eo,rotation_error_deg(tmj[:3,:3],tp.rotation))
        self.data.qpos[:]=saved;mujoco.mj_forward(self.model,self.data)
        if ep>2 or eo>2:raise ValidationFailure(f"FK mismatch {ep:.3f} mm, {eo:.3f} deg")
        return {"max_position_error_mm":ep,"max_rotation_error_deg":eo}

    def verify_projection(self):
        # Compare the pinhole projection of the known cube center against the
        # rendered cube silhouette centroid from MuJoCo segmentation.
        self.renderer.enable_segmentation_rendering()
        try:
            self.renderer.update_scene(self.data,camera=CAMERA)
            segmentation=self.renderer.render()
        finally:
            self.renderer.disable_segmentation_rendering()
        mask=(segmentation[:,:,0]==self.cube_geom)&(
            segmentation[:,:,1]==int(mujoco.mjtObj.mjOBJ_GEOM))
        ys,xs=np.nonzero(mask)
        if len(xs)==0:raise ValidationFailure("cube is not visible in projection check")
        rendered=np.array([xs.mean(),ys.mean()])
        position=self.data.qpos[self.cube_q:self.cube_q+3]
        t=self.camera_pose()
        point_camera=t[:3,:3].T@(position-t[:3,3])
        projected=project_camera(point_camera,self.K)
        err=float(np.linalg.norm(rendered-projected))
        if err>2:raise ValidationFailure(f"render reprojection mismatch {err:.3f} px")
        return err

    def _solve(self,target,seed):
        q,ep,eo=self.ik.solve(pin.SE3(target[:3,:3],target[:3,3]),seed)
        if q is None:raise ValidationFailure("IK found no solution")
        q=np.asarray(q);lim=np.degrees(self._limits())
        if np.any(q<lim[:,0]) or np.any(q>lim[:,1]):raise ValidationFailure("IK violates joint limits")
        if ep>3 or eo>3:raise ValidationFailure(f"IK residual {ep:.2f} mm, {eo:.2f} deg")
        return q,(float(ep),float(eo))

    def _pairs(self):
        return [(int(self.data.contact[i].geom1),int(self.data.contact[i].geom2)) for i in range(self.data.ncon)]

    def _name(self,g):return mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_GEOM,g) or str(g)

    def contact_names(self):
        return sorted([list(sorted((self._name(a),self._name(b)))) for a,b in set(self._pairs())])

    def trace_state(self,force=False):
        if not force and self.steps%15:return
        sample={"time_s":float(self.steps*self.model.opt.timestep),
            "cube_height_m":float(self.data.xpos[self.cube_body,2]),
            "q_rad":self.data.qpos[self.qaddr].copy().tolist(),
            "contacts":self.contact_names()}
        if self.telemetry and self.telemetry[-1]["time_s"]==sample["time_s"]:
            self.telemetry[-1]=sample
        else:self.telemetry.append(sample)

    def _body_name(self,g):
        body=int(self.model.geom_bodyid[g])
        return mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_BODY,body) or str(body)

    def _excluded_proxy_contact(self,a,b):
        return {self._body_name(a),self._body_name(b)}=={"upper_arm_link","forearm_link"}

    def check_path(self,q0,q1):
        saved=self.data.qpos.copy();self.data.qpos[self.gq]=self.data.qpos[self.rgq]=.0345;bad=set();env={"validation_table_top","validation_pedestal_geom","validation_cube_geom"}
        lim=self._limits()
        for a in np.linspace(0,1,25):
            q=np.radians(q0+a*(q1-q0))
            if np.any(q<lim[:,0]) or np.any(q>lim[:,1]):bad.add(("joint_limit","joint_limit"));continue
            self.data.qpos[self.qaddr]=q;mujoco.mj_forward(self.model,self.data)
            for x,y in self._pairs():
                nx,ny=self._name(x),self._name(y)
                if nx in env or ny in env:
                    if {nx,ny}=={"validation_cube_geom","validation_pedestal_geom"}:continue
                    bad.add(tuple(sorted((nx,ny))))
                elif not self._excluded_proxy_contact(x,y):
                    bad.add(tuple(sorted((self._body_name(x),self._body_name(y)))))
        self.data.qpos[:]=saved;mujoco.mj_forward(self.model,self.data)
        if bad:raise ValidationFailure("swept path contact: "+", ".join(f"{x}/{y}" for x,y in sorted(bad)))

    def make_oracle_candidate(self,position_base,width=.025,jaw_axis="x"):
        # Geometry-only top grasp: approach vertically, align the jaw axis with
        # one of the cube's horizontal edges, and keep the 25 mm face width.
        if jaw_axis=="x":
            r=np.array([[0.,1.,0.],[1.,0.,0.],[0.,0.,-1.]])
        elif jaw_axis=="y":
            r=np.array([[-1.,0.,0.],[0.,1.,0.],[0.,0.,-1.]])
        else:
            raise ValueError("jaw_axis must be x or y")
        tbg=tf(r,position_base)
        tcg=np.linalg.inv(self.camera_pose())@tbg
        quat=Rotation.from_matrix(tcg[:3,:3]).as_quat()
        return GraspCandidate(1.,width,tuple(tcg[:3,3]),tuple(quat),
                              {"source":"simulation_ground_truth","jaw_axis_world":jaw_axis})

    def _joint_waypoints(self,q0,q1,max_step_deg=8.):
        q0=np.asarray(q0,dtype=float);q1=np.asarray(q1,dtype=float)
        count=max(1,int(np.ceil(np.max(np.abs(q1-q0))/max_step_deg)))
        return [q0+(q1-q0)*(i/count) for i in range(1,count+1)]

    def make_plan(self,c):
        tcg=candidate_transform(c);tbg=self.camera_pose()@tcg;ttg=self.tool_to_grasp()
        inverse=np.linalg.inv(ttg);pre=tbg.copy()
        pre[:3,3]-=tbg[:3,:3]@np.array([0.,0.,.080])
        seed=np.degrees(self.data.qpos[self.qaddr]);errs=[]
        qpre,e=self._solve(pre@inverse,seed);errs.append(e)
        pre_path=self._joint_waypoints(seed,qpre)
        approach_path=[]
        count=8
        for i in range(1,count+1):
            pose=tbg.copy();pose[:3,3]=pre[:3,3]+(tbg[:3,3]-pre[:3,3])*(i/count)
            q,e=self._solve(pose@inverse,qpre if i==1 else approach_path[-1])
            approach_path.append(q);errs.append(e)
        lift_path=[]
        count=7
        for i in range(1,count+1):
            pose=tbg.copy();pose[2,3]+=LIFT_HEIGHT_M*(i/count)
            q,e=self._solve(pose@inverse,approach_path[-1] if i==1 else lift_path[-1])
            lift_path.append(q);errs.append(e)
        previous=seed
        for q in pre_path+approach_path+lift_path:
            self.check_path(previous,q);previous=q
        return MotionPlan(tuple(tuple(q) for q in pre_path),tuple(tuple(q) for q in approach_path),
            tuple(tuple(q) for q in lift_path),tuple(errs),tuple(map(tuple,tbg)),tuple(map(tuple,ttg)),MAX_OPEN)

    def _bad_contacts(self,allow_cube):
        bad=[];env={self.table_geom,self.ped_geom,self.cube_geom};fingers={self.left_geom,self.right_geom}
        for a,b in self._pairs():
            if a in env or b in env:
                e,o=(a,b) if a in env else (b,a)
                if e==self.cube_geom and allow_cube and o in fingers:continue
                if e in (self.cube_geom,self.ped_geom) and o in env:continue
                bad.append((self._name(e),self._name(o)))
            elif not self._excluded_proxy_contact(a,b):
                bad.append((self._body_name(a),self._body_name(b)))
        return bad

    def _checked(self,n,allow_cube):
        for _ in range(n):
            mujoco.mj_step(self.model,self.data);self.steps+=1;self._after_physics_step()
            bad=self._bad_contacts(allow_cube)
            if bad:raise ValidationFailure(f"unexpected contact {bad[:4]}")

    def _move(self,qdeg,label,allow_cube=False):
        target=np.radians(qdeg);self.data.ctrl[list(self.acts)]=target;stable=0
        for i in range(int(3/self.model.opt.timestep)):
            mujoco.mj_step(self.model,self.data);self.steps+=1;self._after_physics_step()
            bad=self._bad_contacts(allow_cube)
            if bad:raise ValidationFailure(f"{label} contact {bad[:4]}")
            e=float(np.max(np.abs(self.data.qpos[self.qaddr]-target)));v=float(np.max(np.abs(self.data.qvel[self.daddr])))
            stable=stable+1 if e<.012 and v<.04 else 0
            if stable>=25:return {"steps":i+1,"joint_error_rad":e}
        raise ValidationFailure(f"{label} actuator did not settle in 3 s")

    def execute(self,plan):
        self.start_video();self.data.ctrl[self.ga]=plan.opening_m/2;self._checked(250,False)
        qstart=np.degrees(self.data.qpos[self.qaddr]).tolist();moves={"pregrasp":[],"approach":[],"lift":[]}
        for q in plan.q_pre_path:moves["pregrasp"].append(self._move(q,"pregrasp"))
        cube_before=self.data.qpos[self.cube_q:self.cube_q+3].copy()
        for q in plan.q_approach_path:moves["approach"].append(self._move(q,"approach",True))
        delta=float(np.linalg.norm(self.data.qpos[self.cube_q:self.cube_q+3]-cube_before))
        if delta>.002:raise ValidationFailure(f"cube moved {delta*1000:.1f} mm before closing")
        self.data.ctrl[self.ga]=0.;self._checked(350,True)
        contacts=[(self._name(x),self._name(y)) for x,y in self._pairs()
                  if self.cube_geom in (x,y) and ({x,y}&{self.left_geom,self.right_geom})]
        if not contacts:raise ValidationFailure("no finger-to-cube contact")
        for q in plan.q_lift_path:moves["lift"].append(self._move(q,"lift",True))
        zl=float(self.data.xpos[self.cube_body,2])
        self._checked(int(1/self.model.opt.timestep),True);zh=float(self.data.xpos[self.cube_body,2]);dz=zh-CUBE[2]
        if dz<.050:raise ValidationFailure(f"cube lift {dz*1000:.1f} mm < 50 mm")
        if abs(zh-zl)>.005:raise ValidationFailure("cube fell during 1 second hold")
        self.trace_state(force=True);self.record();self.close_video()
        return {"success":True,"q_start_deg":qstart,"moves":moves,"contacts":contacts,
                "cube_motion_before_close_m":delta,"cube_z_initial_m":float(CUBE[2]),
                "cube_z_after_lift_m":zl,"cube_z_after_hold_m":zh,"cube_lift_m":dz,
                "hold_seconds":1.,"video_frames":self.frames,"simulation_trace":self.telemetry}


__all__ = ["ValidationWorld"]

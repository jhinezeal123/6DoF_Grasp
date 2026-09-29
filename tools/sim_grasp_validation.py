#!/usr/bin/env python3
"""Offline photo replay and simulation-only grasp validation."""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
os.environ.setdefault("MUJOCO_GL","egl")
import cv2,numpy as np
from m750.perception.adapters.grasppose import GRASPPOSE_COMMIT,GraspPosePerceptionAdapter
from m750.perception.adapters.grasppose_worker import WorkerGraspEstimator
from m750.perception.types import PerceptionRequest
from m750.sim_validation import CAMERA_Q_DEG,CUBE,LIGHTS,OFFSETS,WIDTH,HEIGHT,LIFT_HEIGHT_M,SEED,ValidationFailure,ValidationWorld,candidate_transform,rotation_error_deg

ROOT=Path(__file__).resolve().parents[1];PIPELINE=Path("/workspace/6DoF_Grasp/grasp_pipeline_repo")
OUTPUT=ROOT/".local_data"/"sim_grasp_validation"
PHOTO_K=np.array([[957.746642,0.,636.883856],[0.,948.820235,352.232764],[0.,0.,1.]])

def commit(path):
    try:return subprocess.check_output(["git","-C",str(path),"rev-parse","HEAD"],text=True,stderr=subprocess.DEVNULL).strip()
    except Exception:return None

def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):h.update(block)
    return h.hexdigest()

def json_default(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,Path):return str(x)
    raise TypeError(type(x).__name__)

def save_report(data,out):
    out.mkdir(parents=True,exist_ok=True);path=out/"report.json"
    path.write_text(json.dumps(data,indent=2,sort_keys=True,default=json_default)+"\n",encoding="utf-8")
    return path

def provider_for(socket):
    est=WorkerGraspEstimator(socket_path=socket) if socket else WorkerGraspEstimator()
    p=GraspPosePerceptionAdapter(est);p.open();return p

def infer(p,image,k,size):
    return p.infer(PerceptionRequest(image=image,prompt_id="cube",camera_matrix=k,
        camera_matrix_size=size,max_width_m=.069,top=5))

def choose(grasps):
    rejected=[]
    for g in sorted(grasps,key=lambda x:float(x.score),reverse=True):
        try:candidate_transform(g);return g,rejected
        except ValidationFailure as e:rejected.append(str(e))
    raise ValidationFailure("no valid worker grasp: "+"; ".join(rejected))

def candidate_dict(g):
    return {"score":float(g.score),"width_m":float(g.width_m),"position_camera_m":list(g.position_m),
            "quaternion_xyzw":list(g.quaternion_xyzw),"metadata":dict(g.metadata)}

def photo_run(photo,p):
    bgr=cv2.imread(str(photo),cv2.IMREAD_COLOR)
    if bgr is None:raise ValidationFailure(f"cannot read {photo}")
    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    trials=[]
    for i in range(3):
        t=time.perf_counter();r=infer(p,rgb,PHOTO_K,(1280,720));ms=(time.perf_counter()-t)*1000
        g,bad=choose(r.grasps);raw=r.raw
        trials.append({"trial":i+1,"elapsed_ms":ms,"detections":getattr(raw,"detection_count",None),
            "mask_pixels":getattr(raw,"mask_pixels",None),"candidate":candidate_dict(g),"rejected":bad})
    return {"path":str(photo),"sha256":digest(photo),"image_size":[rgb.shape[1],rgb.shape[0]],
        "camera_intrinsics":"estimated prior benchmark K at 1280x720","camera_K":PHOTO_K.tolist(),
        "camera_K_reference_size":[1280,720],"joint_state_rad":None,"base_frame_grasp_valid":False,
        "trials":trials,"pass":len(trials)==3}

def one_case(i,offset,light,mode,p,out):
    name=f"{mode}_{i:02d}"
    video=out/"videos"/(name+".mp4") if mode in ("oracle","e2e") else None
    w=ValidationWorld(light,video)
    context={"case":name,"mode":mode,"brightness":light,"offset_xy_m":list(offset),
             "video":str(video) if video else None}
    try:
        w.reset(offset);fk=w.verify_fk();reproj=w.verify_projection();rgb=w.render()
        image=out/"images"/(name+".png");w.save_image(image,rgb)
        camera=w.camera_pose();q=w.data.qpos[w.qaddr].copy()
        cube_at_capture={"position_m":w.data.qpos[w.cube_q:w.cube_q+3].copy(),
            "quaternion_wxyz":w.data.qpos[w.cube_q+3:w.cube_q+7].copy()}
        context.update({"image":str(image),"image_sha256":digest(image),"camera_K":w.K.tolist(),
            "camera_K_size":[WIDTH,HEIGHT],"T_base_camera_cv":camera.tolist(),"q_at_capture_rad":q.tolist(),
            "cube_state_at_capture":cube_at_capture,"fk_check":fk,"reprojection_error_px":reproj})
        if mode=="e2e":
            w.start_video();w.record()
        if mode=="oracle":
            pos=CUBE.copy();pos[:2]+=np.asarray(offset)
            feasible=[];rejected=[]
            for jaw_axis in ("x","y"):
                candidate=w.make_oracle_candidate(pos,jaw_axis=jaw_axis)
                try:
                    candidate_plan=w.make_plan(candidate)
                    path=candidate_plan.q_pre_path+candidate_plan.q_approach_path+candidate_plan.q_lift_path
                    qpath=np.vstack([q,np.asarray(path,dtype=float)])
                    cost=float(np.abs(np.diff(qpath,axis=0)).sum())
                    feasible.append((cost,jaw_axis,candidate,candidate_plan))
                except ValidationFailure as e:
                    rejected.append({"jaw_axis_world":jaw_axis,"error":str(e)})
            if not feasible:
                context["perception"]={"source":"simulation_ground_truth","rejected_candidates":rejected}
                raise ValidationFailure("no collision-free geometric oracle grasp: "+str(rejected))
            _,jaw_axis,g,plan=min(feasible,key=lambda row:row[0])
            perception={"source":"simulation_ground_truth","selected_jaw_axis_world":jaw_axis,
                "candidate":candidate_dict(g),"planned_candidates":[{"jaw_axis_world":axis,"joint_travel_deg":cost}
                for cost,axis,_,_ in feasible],"rejected_candidates":rejected}
        else:
            t=time.perf_counter();r=infer(p,rgb,w.K,(WIDTH,HEIGHT));elapsed=(time.perf_counter()-t)*1000
            candidates=sorted(r.grasps,key=lambda x:float(x.score),reverse=True)
            raw=r.raw
            perception={"elapsed_ms":elapsed,"detections":getattr(raw,"detection_count",None),
                "mask_pixels":getattr(raw,"mask_pixels",None),"candidate_count":len(candidates),
                "worker_candidates":[candidate_dict(c) for c in candidates],"planning_attempts":[]}
            context["perception"]=perception
            g=None;plan=None
            for rank,candidate in enumerate(candidates,1):
                attempt={"rank":rank,"score":float(candidate.score),"width_m":float(candidate.width_m)}
                try:
                    tbg=w.camera_pose()@candidate_transform(candidate)
                    cube_pos=np.asarray(cube_at_capture["position_m"],dtype=float)
                    orientation_x=np.array([[0.,1.,0.],[1.,0.,0.],[0.,0.,-1.]])
                    orientation_y=np.array([[-1.,0.,0.],[0.,1.,0.],[0.,0.,-1.]])
                    attempt["simulation_truth_error"]={
                        "position_mm":float(np.linalg.norm(tbg[:3,3]-cube_pos)*1000),
                        "rotation_to_nearest_top_grasp_deg":min(
                            rotation_error_deg(tbg[:3,:3],orientation_x),
                            rotation_error_deg(tbg[:3,:3],orientation_y))}
                    candidate_plan=w.make_plan(candidate)
                    attempt["result"]="ik_and_path_pass"
                    perception["planning_attempts"].append(attempt)
                    g=candidate;plan=candidate_plan
                    perception["selected_rank"]=rank
                    perception["selected_candidate"]=candidate_dict(candidate)
                    perception["selected_simulation_truth_error"]=attempt["simulation_truth_error"]
                    break
                except (ValidationFailure,ValueError) as e:
                    attempt.update({"result":"rejected","error":str(e)})
                    perception["planning_attempts"].append(attempt)
            if g is None:
                if not candidates:raise ValidationFailure("worker returned no grasps")
                raise ValidationFailure("no worker grasp passes IK and swept-path checks")
        context["perception"]=perception
        motion_plan={"q_pregrasp_path_deg":[list(map(float,q)) for q in plan.q_pre_path],
            "q_approach_path_deg":[list(map(float,q)) for q in plan.q_approach_path],
            "q_lift_path_deg":[list(map(float,q)) for q in plan.q_lift_path],
            "ik_errors":plan.ik_errors,"grasp_base":plan.grasp_base,"requested_lift_m":LIFT_HEIGHT_M,
            "tool_to_grasp":plan.tool_to_grasp,"opening_m":plan.opening_m}
        context["motion_plan"]=motion_plan
        result=w.execute(plan)
        result.update({"case":name,"mode":mode,"brightness":light,"offset_xy_m":list(offset),"image":str(image),
            "image_sha256":digest(image),"camera_K":w.K.tolist(),"camera_K_size":[WIDTH,HEIGHT],"T_base_camera_cv":camera.tolist(),
            "q_at_capture_rad":q.tolist(),"cube_state_at_capture":cube_at_capture,"fk_check":fk,"reprojection_error_px":reproj,
            "perception":perception,"motion_plan":motion_plan,"video":str(video) if video else None})
        return result
    except Exception as e:
        w.close_video()
        context.update({"success":False,"error_type":type(e).__name__,"error":str(e)})
        return context
    finally:
        w.close()

def ten_cases(mode,p,out):
    rows=[]
    for offset in OFFSETS:
        for light in LIGHTS:rows.append(one_case(len(rows)+1,offset,light,mode,p,out))
    return rows

def main():
    ap=argparse.ArgumentParser(description="MuJoCo only; this tool has no real-robot mode.")
    ap.add_argument("--mode",choices=("photo","oracle","e2e","all"),default="all")
    ap.add_argument("--photo",type=Path);ap.add_argument("--socket");ap.add_argument("--pipeline-repo",type=Path,default=PIPELINE)
    ap.add_argument("--output",type=Path,default=OUTPUT);args=ap.parse_args()
    if args.mode in ("photo","all") and args.photo is None:ap.error("--photo required for photo/all")
    if "m750.robot.adapters.pymycobot" in sys.modules or "m750.ros" in sys.modules:
        raise SystemExit("hardware/ROS module loaded; refusing simulation run")
    report={"schema_version":1,"mode":args.mode,"sixdof_commit":commit(ROOT),"pipeline_commit":commit(args.pipeline_repo),
        "worker_pin":GRASPPOSE_COMMIT,"worker_prompt_id":"cube","hardware_driver_imported":False,"real_robot_commands_sent":False,
        "seed":SEED,"seed_usage":"FK sample generation; physics episodes are deterministic",
        "scene_configuration":{"scene_xml":str(ValidationWorld.SCENE),"scene_xml_sha256":digest(ValidationWorld.SCENE),
            "robot_model_xml_sha256":digest(ValidationWorld.SCENE.parent/"myarm_m750_mujoco.xml"),
            "resolution_px":[WIDTH,HEIGHT],"camera_name":"wrist_cam","camera_fovy_deg":42.2,
            "camera_start_joints_deg":CAMERA_Q_DEG.tolist(),"cube_edge_m":.025,
            "cube_center_base_m":CUBE.tolist(),"cube_density_kg_m3":1200,"cube_mass_kg":.01875,
            "pedestal_radius_m":.004,"pedestal_height_m":.060,"table_top_z_m":.052,
            "table_half_size_xy_m":[.26,.26],"light_diffuse_rgb":[.7,.7,.7],
            "light_ambient_rgb":[.15,.15,.15],"max_gripper_opening_m":.069,
            "excluded_collision_proxy_bodies":[["upper_arm_link","forearm_link"]]},
        "assumptions":{"cube_edge_m":.025,"density_kg_m3":1200,"mass_kg":.01875,"pedestal_height_m":.060,
        "camera_fovy_deg":42.2,"sim_camera_K":"derived from rendered MuJoCo camera"},
        "results":{}}
    p=None
    try:
        if args.mode in ("photo","e2e","all"):p=provider_for(args.socket)
        if args.mode in ("photo","all"):report["results"]["photo"]=photo_run(args.photo,p)
        if args.mode in ("oracle","all"):report["results"]["oracle"]=ten_cases("oracle",None,args.output)
        if args.mode in ("e2e","all"):report["results"]["e2e"]=ten_cases("e2e",p,args.output)
    except Exception as e:report["fatal_error"]={"type":type(e).__name__,"message":str(e)}
    finally:
        if p is not None:
            try:p.close()
            except Exception:pass
    for key,val in list(report["results"].items()):
        if key in ("oracle","e2e"):
            report["results"][key+"_pass_count"]=sum(bool(x.get("success")) for x in val)
            report["results"][key+"_case_count"]=len(val)
            report["results"][key+"_passed"]=len(val)==10 and all(x.get("success") for x in val)
    path=save_report(report,args.output)
    print(json.dumps({"report":str(path),"results":report["results"],"fatal_error":report.get("fatal_error")},indent=2,default=json_default))
    return 1 if "fatal_error" in report or not all(report["results"].get(k+"_passed",True) for k in ("oracle","e2e") if k in report["results"]) or not report["results"].get("photo",{}).get("pass",True) else 0

if __name__=="__main__":raise SystemExit(main())

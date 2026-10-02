#!/usr/bin/env python3
"""Command line entry point for the simulation grasp validation harness.

MuJoCo only; this tool has no real-robot mode.
"""

from __future__ import annotations

import argparse,json,sys
from pathlib import Path
from ..perception.adapters.grasppose import GRASPPOSE_COMMIT
from .application import (BRIDGE, PIPELINE, SETTINGS, commit, digest, json_default,
    photo_run, provider_for, save_report, ten_cases)
from .scenario import CAMERA_Q_DEG, CUBE, HEIGHT, SCENE, SEED, VOLUME_SIZE_M, WIDTH

# src/m750/sim/cli.py -> repository root. Used only for the reported commit and
# for the default artifact directory.
ROOT=Path(__file__).resolve().parents[3]
OUTPUT=ROOT/".local_data"/"sim_grasp_validation"

def main():
    ap=argparse.ArgumentParser(description="MuJoCo only; this tool has no real-robot mode.")
    ap.add_argument("--mode",choices=("photo","oracle","e2e","all"),default="all")
    ap.add_argument("--photo",type=Path);ap.add_argument("--socket");ap.add_argument("--pipeline-repo",type=Path,default=PIPELINE)
    ap.add_argument("--volume",choices=("auto","gravity"),default="auto",
        help="auto keeps the pipeline's camera-aligned TSDF volume; gravity anchors a gravity-aligned volume at the known capture-time object centre")
    ap.add_argument("--depth-source",choices=("worker","sim"),default="worker",
        help="sim feeds simulator ground-truth depth through m750/sim/adapters/grasppose_bridge.py")
    ap.add_argument("--output",type=Path,default=OUTPUT);args=ap.parse_args()
    SETTINGS["volume"]=args.volume;SETTINGS["depth_source"]=args.depth_source;SETTINGS["pipeline"]=args.pipeline_repo
    if args.mode in ("photo","all") and args.photo is None:ap.error("--photo required for photo/all")
    if "m750.robot.adapters.pymycobot" in sys.modules or "m750.ros" in sys.modules:
        raise SystemExit("hardware/ROS module loaded; refusing simulation run")
    report={"schema_version":1,"mode":args.mode,"sixdof_commit":commit(ROOT),"pipeline_commit":commit(args.pipeline_repo),
        "worker_pin":GRASPPOSE_COMMIT,"worker_prompt_id":"cube","hardware_driver_imported":False,"real_robot_commands_sent":False,
        "seed":SEED,"seed_usage":"FK sample generation; physics episodes are deterministic",
        "validation_options":{"volume_frame":args.volume,"depth_source":args.depth_source,
            "volume_size_m":VOLUME_SIZE_M,
            "volume_anchor":("known capture-time object centre (harness ground truth)"
                if args.volume=="gravity" else "pipeline automatic camera-aligned pose"),
            "depth_bridge":str(BRIDGE) if args.depth_source=="sim" else None},
        "scene_configuration":{"scene_xml":str(SCENE),"scene_xml_sha256":digest(SCENE),
            "robot_model_xml_sha256":digest(SCENE.parent/"myarm_m750_mujoco.xml"),
            "resolution_px":[WIDTH,HEIGHT],"camera_name":"wrist_cam","camera_fovy_deg":42.2,
            "camera_local_position_m":[-.04650,0.,.02069],
            "camera_local_quaternion_wxyz":[0.,.7071068,-.7071068,0.],
            "camera_start_joints_deg":CAMERA_Q_DEG.tolist(),
            "cube_edge_m":.025,"cube_center_base_m":CUBE.tolist(),"cube_density_kg_m3":1200,
            "cube_mass_kg":.01875,"cube_visual_rgba":[.08,.22,.85,1.],
            "cube_friction":[1.2,.01,.001],
            "pedestal_center_base_m":[.30,.10,.082],"pedestal_radius_m":.004,
            "pedestal_height_m":.060,"pedestal_visual_rgba":[.35,.35,.38,1.],
            "pedestal_friction":[.8,.01,.001],
            "table_top_z_m":.052,"table_half_size_xy_m":[.26,.26],
            "table_visual_rgba":[.12,.12,.12,1.],"table_friction":[.8,.01,.001],
            "light_position_base_m":[.30,-.20,.75],"light_direction":[0.,0.,-1.],
            "light_directional":True,"light_diffuse_rgb":[.7,.7,.7],
            "light_ambient_rgb":[.15,.15,.15],"max_gripper_opening_m":.069,
            "excluded_collision_proxy_bodies":[["upper_arm_link","forearm_link"]]},
        "assumptions":{"cube_edge_m":.025,"density_kg_m3":1200,"mass_kg":.01875,
            "cube_visual_rgba":[.08,.22,.85,1.],"cube_friction":[1.2,.01,.001],
            "pedestal_height_m":.060,"camera_fovy_deg":42.2,
            "camera_local_pose":"MuJoCo model assumption; not real hand-eye calibration",
            "sim_camera_K":"derived from rendered MuJoCo camera",
            "photo_camera_K":"estimated prior benchmark K; no synchronized joint state"},
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

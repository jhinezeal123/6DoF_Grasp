"""P3 - Lap rap pipeline gap vat: PhysBrain + servo anh, chay trong khung m750.

    Source            CameraRobotSource(PrimaryCamera, Robot)     <- adapter co san
    Policy            GraspPointPolicy(model + ImageServo)
    guard             WorkspaceGuard
    Sink              RobotSink                                   <- adapter co san

Khong co harness rieng: Source va Sink la adapter CO SAN cua m750.pipeline, chi
Policy va guard la module moi. Doi model hay doi cach dieu khien chi can thay
mot module.

Yeu cau: pip install -e . o repo root (package m750).

Chay:
    python run_pipeline.py --dry-run          # khong ra lenh chuyen dong
    python run_pipeline.py                    # CHAY THAT
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # chi de import vla/ (goi tu thu muc experiment)

from m750.pipeline.adapters.dry_run import DryRunSink    # noqa: E402
from m750.pipeline.adapters.sdk_sink import RobotSink    # noqa: E402
from m750.pipeline.adapters.sdk_source import CameraRobotSource  # noqa: E402
from m750.pipeline.runner import Pipeline                # noqa: E402
from m750.pipeline.types import ActionSpec               # noqa: E402

from m750.ros.robot import Robot                         # noqa: E402
from m750.ros.bridge import get_bridge                   # noqa: E402

from vla.guard import WorkspaceGuard                      # noqa: E402
from vla.online_jacobian import OnlineJacobian            # noqa: E402
from vla.physbrain_model import PhysBrainClient           # noqa: E402
from vla.policy import GraspPointPolicy                   # noqa: E402
from vla.primary_camera import PrimaryCamera              # noqa: E402
from vla.servo import ImageServo                          # noqa: E402

CONFIG = HERE / "configs" / "grasp_myarm_m750.json"
JACOBIAN = HERE / "configs" / "servo_jacobian.json"

# Khop voi RobotSink: mode "tcp_pose" doi dung 7 gia tri [x,y,z,qx,qy,qz,qw].
SPEC = ActionSpec(
    mode="tcp_pose",
    axes=("x", "y", "z", "qx", "qy", "qz", "qw"),
    units=("m", "m", "m", "rad", "rad", "rad", "rad"),
    frame_id="base_link",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Pipeline gap vat PhysBrain + servo anh.")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--jacobian", type=Path, default=JACOBIAN)
    parser.add_argument("--instruction", default=None, help="de trong = lay tu config")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="di het pipeline nhung KHONG ra lenh chuyen dong")
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    instruction = args.instruction or config["instruction"]
    safety = config["safety"]

    # ---------------------------------------------------------------- kiem tra truoc
    print("=== KIEM TRA TRUOC KHI CHAY ===")
    model = PhysBrainClient(config["model"]["url"], config["model"]["timeout_s"])
    if not model.ready():
        print("LOI: llama-server khong tra loi o", config["model"]["url"])
        print("     chay: scripts/serve_llamacpp.sh start")
        return 1
    print("  model server : OK")

    jacobian_data = json.loads(args.jacobian.read_text())
    servo = ImageServo(
        OnlineJacobian(jacobian_data["J"]),
        max_step_xy_m=config["servo"]["max_step_xy_m"],
        max_step_z_m=config["servo"]["max_step_z_m"],
        align_px=config["servo"]["align_px"],
    )
    print("  J ban dau    : cond=%.1f  do nhay (px/mm) %s"
          % (servo.cond,
             "  ".join("%s=%.2f" % (a, b / 1000.0)
                       for a, b in zip(jacobian_data.get("axes", "xyz"), servo.px_per_m))))
    print("                 (se tu hoc lai trong luc chay; day chi la diem khoi dong)")

    robot = Robot()
    print("  cho feedback robot (toi da 25s)...")
    if not robot.wait_until_online(25.0):
        print("LOI: stack ROS khong phat feedback. Da chay run_web.sh chua?")
        return 1

    p0 = np.array(robot.tcp_pos, dtype=np.float64)
    q0 = np.array(robot.tcp_quat, dtype=np.float64)
    if not np.all(np.isfinite(p0)) or not np.all(np.isfinite(q0)):
        print("LOI: tcp_pos/tcp_quat la NaN. Kiem tra DDS (xem P2_KET_QUA.md §2 loi 1).")
        return 1
    print("  tcp_pos      : %s" % np.round(p0, 4))
    print("  tcp_quat     : %s" % np.round(q0, 4))

    if not args.dry_run:
        bridge = get_bridge()
        if not robot.is_armed:
            print("  dang disarmed; goi rearm()...")
            if not robot.rearm():
                print("LOI: rearm() that bai.")
                return 1
            time.sleep(1.5)
        if not robot.is_armed:
            print("LOI: van chua armed. safety_state =", robot.safety_state)
            return 1
        print("  safety_state : %s (armed)" % robot.safety_state)
        state, detail, _ = bridge.motion_state
        print("  motion_state : %s" % (state or "(trong)"))
    else:
        print("  CHE DO DRY-RUN: khong ra lenh chuyen dong nao")

    # ---------------------------------------------------------------- lap rap
    camera = PrimaryCamera(config["camera"]["index"],
                           config["camera"]["width"],
                           config["camera"]["height"])
    source = CameraRobotSource(
        camera, robot,
        camera_name=config["camera"]["name"],
        # PHAI lon hon do tre mot buoc (2 lan goi model ~30 s): runner kiem tra do
        # tuoi SAU predict(), nen observation da het han se lam buoc do vo.
        max_age_s=config["runner"]["max_age_s"],
    )
    policy = GraspPointPolicy(
        robot, model, servo, SPEC,
        camera_name=config["camera"]["name"],
        z_floor_m=safety["z_floor_m"],
        max_descend_m=safety["max_descend_m"],
        max_target_drift_m=safety["max_target_drift_m"],
    )
    guard = WorkspaceGuard(
        SPEC,
        origin=p0,
        reach_m=safety["reach_m"],
        z_floor_m=safety["z_floor_m"],
        z_ceiling_m=safety["z_ceiling_m"],
        max_jump_m=safety["max_jump_m"],
    )
    sink = DryRunSink(SPEC) if args.dry_run else RobotSink(robot, SPEC)

    pipeline = Pipeline(
        source=source, policy=policy, sink=sink, guard=guard,
        rate_hz=config["runner"]["rate_hz"],
        read_timeout_s=config["runner"]["read_timeout_s"],
    )

    max_steps = args.max_steps if args.max_steps is not None else safety["max_steps"]
    print("\n=== CHAY: %r  (toi da %d buoc) ===" % (instruction, max_steps))
    print("  moi buoc ~15 s vi phai hoi model; kien nhan.\n")

    steps = 0
    try:
        steps = pipeline.run(instruction, max_steps=max_steps)
        print("\n=== XONG: %d buoc ===" % steps)
    except KeyboardInterrupt:
        print("\n=== NGUOI DUNG DUNG (Ctrl-C) ===")
    except Exception as exc:  # noqa: BLE001 - bao loi that ra ngoai
        print("\n=== DUNG VI LOI: %s: %s ===" % (type(exc).__name__, exc))
    finally:
        if not args.dry_run:
            print("\n=== VE TU THE DAU ===")
            try:
                robot.set_tcp_pose(p0, q0)
                time.sleep(6.0)
                back = np.array(robot.tcp_pos, dtype=np.float64)
                print("  tcp sau khi ve = %s   lech %.1f mm"
                      % (np.round(back, 4), np.linalg.norm(back - p0) * 1000))
            except Exception as exc:  # noqa: BLE001
                print("  khong ve duoc tu the dau:", exc)
        camera.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

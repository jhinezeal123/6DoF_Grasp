"""Do khung hinh wrist_cam o DUNG pose dang chay tren server (tay da xe khoi HOME).

Chay:  $env:MUJOCO_GL="wgl"; python probe_pose.py
Thu ca phuong an ngang (khong pitch) lan pitch xuong, in toa do anh cua 5 vat + 2 ngon.
"""
import json
import os
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "wgl")
import mujoco  # noqa: E402

DS = r"D:\Documents\mujoco\kaggle_mujoco\test\dataset_src"
PROBE = r"D:\Documents\mujoco\_probe"
SCENE = os.path.join(DS, "robot_model", "scene_vla.xml")
OBJS = ["obj_cube", "obj_box", "obj_cyl", "obj_sphere", "obj_plate"]
W, H = 640, 480


def quat_pitch(q, deg):
    """Quay them quanh truc x cua chinh camera (pitch), tra ve quat moi."""
    a = np.radians(deg) / 2.0
    p = np.array([np.cos(a), np.sin(a), 0.0, 0.0])
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, q, p)
    return out


def main():
    M = mujoco.MjModel.from_xml_path(SCENE)
    D = mujoco.MjData(M)
    # qpos that tren server (tay da xe khoi HOME) neu co; khong thi dung HOME ly thuyet.
    # Luu y: dat qpos=0 se nem ca 5 free joint ve goc toa do the gioi -> dung bao gio.
    qp = os.path.join(PROBE, "server_qpos.json")
    if os.path.exists(qp):
        D.qpos[:] = json.load(open(qp))
        print("dung qpos that lay tu /api/state")
    else:
        D.qpos[:] = M.qpos0
        D.qpos[:8] = [0.0, -0.45, 0.9, 0.0, 0.45, 0.0, 0.017, 0.017]
        print("khong co server_qpos.json -> dung HOME ly thuyet "
              "(tren server that tay xe ~0.136 rad, khung hinh se khac)")
    mujoco.mj_forward(M, D)

    ci = mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
    gb = mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_BODY, "gripper_base_link")
    obid = {n: mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_BODY, n) for n in OBJS}
    th = np.tan(np.radians(M.cam_fovy[ci]) / 2)
    tw = th * W / H
    R_gb = D.xmat[gb].reshape(3, 3)
    print("pose: khop =", [round(v, 4) for v in D.qpos[:8]])
    print("goc gripper_base_link (world) =", np.round(D.xpos[gb], 4).tolist())
    print("z_body (world) =", np.round(R_gb[:, 2], 4).tolist())
    print("goc nhin so voi mat phang ngang = %.1f do (duong = huong len)"
          % np.degrees(np.arcsin(-R_gb[2, 2])))

    base_quat = np.array(M.cam_quat[ci], dtype=float)
    CAND = [("khong pitch", -0.045, 0.020, 0.0),
            ("pitch -8", -0.045, 0.020, -8.0),
            ("pitch -15", -0.045, 0.020, -15.0),
            ("pitch -22", -0.045, 0.020, -22.0)]
    r = mujoco.Renderer(M, height=H, width=W)
    for name, cx, cz, pitch in CAND:
        M.cam_pos[ci] = (cx, 0.0, cz)
        M.cam_quat[ci] = base_quat if pitch == 0 else quat_pitch(base_quat, pitch)
        mujoco.mj_forward(M, D)
        cm = D.cam_xmat[ci].reshape(3, 3)
        # goc nhin so voi phuong ngang: am = chuc xuong
        look = -cm[:, 2]
        print("\n%-12s pos=(%.3f,0,%.3f) pitch=%+.0f  nhin=%s  goc xuong=%.1f do"
              % (name, cx, cz, pitch, np.round(look, 3).tolist(),
                 np.degrees(np.arcsin(-look[2]))))
        nvis = 0
        for n in OBJS:
            d = D.xpos[obid[n]] - D.cam_xpos[ci]
            comp = np.array([d @ cm[:, 0], d @ cm[:, 1], -d @ cm[:, 2]])
            if comp[2] <= 1e-6:
                print("   %-11s SAU camera" % n)
                continue
            u, v = comp[0] / comp[2] / tw, comp[1] / comp[2] / th
            ins = abs(u) <= 1 and abs(v) <= 1
            nvis += ins
            print("   %-11s depth=%.3f u=%+.2f v=%+.2f %s"
                  % (n, comp[2], u, v, "OK" if ins else "ngoai"))
        print("   -> %d/5 vat trong khung" % nvis)
        r.update_scene(D, camera=ci)
        from PIL import Image
        Image.fromarray(r.render()).save(
            os.path.join(PROBE, "pose_%s.jpg" % name.replace(" ", "_").replace("-", "m")),
            quality=88)
    r.close()


if __name__ == "__main__":
    sys.exit(main())

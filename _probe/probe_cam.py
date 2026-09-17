"""Thu cac ung vien vi tri + huong cho wrist_cam, render ra JPEG de nhin.

Chay:  $env:MUJOCO_GL="wgl"; python probe_cam.py
Ghi ra: D:\Documents\mujoco\_probe\cam_<ten>.jpg  + in toa do anh cua dinh 2 ngon.
"""
import os
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "wgl")
import mujoco  # noqa: E402

DS = r"D:\Documents\mujoco\kaggle_mujoco\test\dataset_src"
OUT = r"D:\Documents\mujoco\_probe"
SCENE = os.path.join(DS, "robot_model", "scene_vla.xml")
HOME = [0.0, -0.45, 0.9, 0.0, 0.45, 0.0, 0.017, 0.017]

# (ten, pos, quat, ghi chu)
CAND = [
    ("A_hien_tai",     (0.055, 0, 0.035), (0, 0, 1, 0), "dang dung: +x, quat cu"),
    ("B_doix_035",     (-0.055, 0, 0.035), (0, 0.7071068, -0.7071068, 0), "-x, roll tu nhien"),
    ("C_doix_020",     (-0.055, 0, 0.020), (0, 0.7071068, -0.7071068, 0), "-x, thap hon"),
    ("D_doi_045_020",  (-0.045, 0, 0.020), (0, 0.7071068, -0.7071068, 0), "-x gan hon"),
    ("E_doix_quatcu",  (-0.055, 0, 0.035), (0, 0, 1, 0), "-x, quat cu"),
    ("F_cux_rollmoi",  (0.055, 0, 0.035), (0, 0.7071068, -0.7071068, 0), "+x, roll tu nhien"),
]


def main():
    M = mujoco.MjModel.from_xml_path(SCENE)
    D = mujoco.MjData(M)
    # qpos0 chu khong phai 0: 5 vat the la free joint, dat qpos=0 se nem chung ve
    # goc toa do the gioi (nam SAU camera) va moi ket luan ve khung hinh deu sai.
    D.qpos[:] = M.qpos0
    D.qpos[:8] = HOME
    mujoco.mj_forward(M, D)

    ci = mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
    gb = mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_BODY, "gripper_base_link")
    assert ci >= 0 and M.cam_bodyid[ci] == gb
    fovy = float(M.cam_fovy[ci])
    W, H = 640, 480
    th = np.tan(np.radians(fovy) / 2)
    tw = th * W / H
    print("fovy=%.1f deg -> tan(v/2)=%.4f  tan(h/2)=%.4f" % (fovy, th, tw))

    # diem dinh 2 ngon trong frame gripper_base_link (do tu AABB)
    tips = {"left(+y)": np.array([0.0, 0.025, 0.100]),
            "right(-y)": np.array([0.0, -0.025, 0.100])}
    objs = ["obj_cube", "obj_box", "obj_cyl", "obj_sphere", "obj_plate"]
    obid = {n: mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_BODY, n) for n in objs}
    R_gb = D.xmat[gb].reshape(3, 3)
    p_gb = D.xpos[gb]
    print("truc gripper_base_link (world): x=%s y=%s z=%s"
          % tuple(np.round(R_gb[:, i], 4).tolist() for i in range(3)))
    print("goc gripper_base_link (world) = %s" % np.round(p_gb, 4).tolist())

    def project(pw, cpos, cx, tw, th):
        d = pw - cpos
        comp = np.array([d @ cx[:, 0], d @ cx[:, 1], -d @ cx[:, 2]])
        if comp[2] <= 1e-6:
            return None
        return comp[0] / comp[2] / tw, comp[1] / comp[2] / th, comp[2]

    r = mujoco.Renderer(M, height=H, width=W)
    for name, pos, quat, note in CAND:
        M.cam_pos[ci] = pos
        M.cam_quat[ci] = quat
        mujoco.mj_forward(M, D)
        cx = D.cam_xmat[ci].reshape(3, 3)
        print("\n%-16s pos=%-22s quat=%s  (%s)" % (name, str(pos), str(quat), note))
        print("   anh-right=%s  anh-up=%s  nhin=%s"
              % (tuple(np.round(cx[:, 0], 4).tolist()),
                 tuple(np.round(cx[:, 1], 4).tolist()),
                 tuple(np.round(-cx[:, 2], 4).tolist())))
        print("   anh-up . z_world = %+.4f  (>0 la huong len)" % cx[2, 1])
        nvis = 0
        for n in objs:
            r_ = project(D.xpos[obid[n]], D.cam_xpos[ci], cx, tw, th)
            if r_ is None:
                print("   vat %-11s SAU camera" % n)
                continue
            u, v, dep = r_
            ins = abs(u) <= 1 and abs(v) <= 1
            nvis += ins
            print("   vat %-11s depth=%.3f  u=%+.2f v=%+.2f  %s"
                  % (n, dep, u, v, "TRONG KHUNG" if ins else "ngoai khung"))
        print("   -> %d/5 vat trong khung" % nvis)
        for tn, tb in tips.items():
            pw = p_gb + R_gb @ tb
            d = pw - D.cam_xpos[ci]
            comp = np.array([d @ cx[:, 0], d @ cx[:, 1], -d @ cx[:, 2]])
            if comp[2] <= 1e-6:
                print("   %-10s SAU camera" % tn)
                continue
            u, v = comp[0] / comp[2] / tw, comp[1] / comp[2] / th
            inside = abs(u) <= 1 and abs(v) <= 1
            print("   %-10s depth=%.3f  anh u=%+.2f v=%+.2f  %s"
                  % (tn, comp[2], u, v, "TRONG KHUNG" if inside else "ngoai khung"))
        r.update_scene(D, camera=ci)
        img = r.render()
        # luu bang chinh bo ma hoa cua server neu co, khong thi PPM tho
        try:
            from PIL import Image
            Image.fromarray(img).save(os.path.join(OUT, "cam_%s.jpg" % name), quality=88)
        except ImportError:
            print("   (khong co PIL, bo qua luu anh)")
    r.close()


if __name__ == "__main__":
    sys.exit(main())

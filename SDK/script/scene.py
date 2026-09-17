"""
script/scene.py - Cac ham tien ich quan ly va reset scene mo phong MuJoCo.

Gom 2 nhom:
  - reset_scene(): dua sim ve trang thai ban dau.
  - get_objects() / set_object_pose() / randomize_objects(): doc & dat vat the
    theo TEN (khong can biet body_jntadr / jnt_qposadr cua MuJoCo).
"""
from typing import Optional, Dict, Any, Sequence
import numpy as np
import mujoco


def table_z(mj_model: mujoco.MjModel, geom_name: str = "table_top") -> float:
    """
    Cao do mat lam viec (met) doc TRUC TIEP tu model, khong hardcode.

    Day la NGUON DUY NHAT cho chieu cao mat ban. Truoc day 0.045 bi viet cung o
    nhieu noi; khi doi chieu day foam trong scene_vla.xml thi cac cho do am tham
    sai (vat spawn ngap trong ban). Sua chieu cao -> sua XML, khong sua Python.

    Mat ban = geom center z + half size z (geom dang box).
    """
    gid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if gid < 0:
        raise ValueError(
            f"Khong tim thay geom '{geom_name}' trong scene de lay cao do mat ban. "
            f"Truyen table_z=... thu cong neu scene cua ban dat ten khac."
        )
    return float(mj_model.geom_pos[gid][2] + mj_model.geom_size[gid][2])


# Alias noi bo: randomize_objects co tham so cung ten 'table_z' nen khong goi
# truc tiep duoc ten ham ma khong bi che khuat.
_table_z_of = table_z


def _free_joint_addr(mj_model: mujoco.MjModel, name: str):
    """Tra (body_id, qpos_addr) cua mot body co free joint. None neu khong phai free joint."""
    bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        return None, None
    jid = mj_model.body_jntadr[bid]
    if jid < 0 or mj_model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
        return None, None
    return bid, mj_model.jnt_qposadr[jid]


def get_objects(mj_model: mujoco.MjModel, mj_data: mujoco.MjData) -> Dict[str, Dict[str, Any]]:
    """
    Tra ve vi tri + huong cua MOI vat the co free joint trong scene.

    Returns:
        {'obj_cube': {'pos': (3,), 'quat': (4,)}, ...}
    """
    out = {}
    for bid in range(mj_model.nbody):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if not name:
            continue
        _, addr = _free_joint_addr(mj_model, name)
        if addr is None:
            continue
        out[name] = {'pos': mj_data.qpos[addr:addr + 3].copy(),
                     'quat': mj_data.qpos[addr + 3:addr + 7].copy()}
    return out


def set_object_pose(mj_model: mujoco.MjModel, mj_data: mujoco.MjData,
                    name: str, pos: Sequence[float],
                    quat: Optional[Sequence[float]] = None) -> bool:
    """
    Dat vi tri (va huong) cho mot vat the theo TEN.

    Args:
        name: ten body, vd 'obj_cube'.
        pos: [x, y, z] met.
        quat: [w, x, y, z] neu muon doi huong; None = giu nguyen huong hien tai.

    Returns:
        True neu dat duoc, False neu khong tim thay vat the / khong phai free joint.
    """
    _, addr = _free_joint_addr(mj_model, name)
    if addr is None:
        return False
    mj_data.qpos[addr:addr + 3] = np.asarray(pos, dtype=np.float64)
    if quat is not None:
        mj_data.qpos[addr + 3:addr + 7] = np.asarray(quat, dtype=np.float64)
    mujoco.mj_forward(mj_model, mj_data)
    return True


def _half_height(mj_model: mujoco.MjModel, body_id: int) -> float:
    """
    Nua chieu cao cua vat the, tinh theo LOAI geom (khong phai luon size[2]).

      box      -> size[2]      cylinder -> size[1]      sphere -> size[0]
      mesh     -> khoang cach tu goc toi day duoi (mesh khong o goc 0)

    Tra 0.02 (mac dinh) neu khong xac dinh duoc.
    """
    gid = mj_model.body_geomadr[body_id]
    if gid < 0:
        return 0.02
    gs = mj_model.geom_size[gid]
    gt = mj_model.geom_type[gid]
    if gt == mujoco.mjtGeom.mjGEOM_BOX:
        return float(gs[2])
    if gt in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
        return float(gs[1])
    if gt == mujoco.mjtGeom.mjGEOM_SPHERE:
        return float(gs[0])
    if gt == mujoco.mjtGeom.mjGEOM_MESH:
        mid = mj_model.geom_dataid[gid]
        if mid >= 0:
            a = mj_model.mesh_vertadr[mid]
            z = mj_model.mesh_vert[a:a + mj_model.mesh_vertnum[mid], 2]
            return float(-z.min())
    return 0.02


def randomize_objects(mj_model: mujoco.MjModel, mj_data: mujoco.MjData,
                      region: Sequence[float] = (0.25, 0.60, -0.15, 0.15),
                      table_z: Optional[float] = None,
                      names: Optional[Sequence[str]] = None,
                      min_gap: float = 0.06,
                      rng: Optional[np.random.Generator] = None) -> Dict[str, np.ndarray]:
    """
    Rải vat the ngau nhien tren mat ban (vung hinh chu nhat tren mat phang XY).

    Do cao Z = table_z + nua chieu cao vat the, nen vat luon nam DUNG tren mat ban.

    Args:
        region: (x_min, x_max, y_min, y_max) met.
        table_z: cao do mat ban (met). None = doc tu model (khuyen dung). Chi
            truyen khi mat ban khong ten 'table_top'.
        names: danh sach vat the can rai; None = tat ca vat co free joint.
        min_gap: khoang cach toi thieu giua tam cac vat (m). Vat nao khong tim
            duoc cho sau vai lan thu se giu nguyen vi tri cu.
        rng: numpy Generator de tai lap ket qua (np.random.default_rng(seed)).

    Returns:
        {ten: vi tri moi} cho cac vat da dat duoc.
    """
    if table_z is None:
        table_z = _table_z_of(mj_model)
    rng = rng or np.random.default_rng()
    x0, x1, y0, y1 = region

    objs = get_objects(mj_model, mj_data)
    if names is None:
        names = list(objs)

    placed = []
    result = {}
    for name in names:
        if name not in objs:
            continue
        bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
        z = table_z + _half_height(mj_model, bid)
        # ponytail: thu 20 lan roi bo cuoc; vung rai nho thi dung chia o phuc tap hon
        for _ in range(20):
            x = rng.uniform(x0, x1)
            y = rng.uniform(y0, y1)
            if all(np.hypot(x - px, y - py) >= min_gap for px, py in placed):
                placed.append((x, y))
                if set_object_pose(mj_model, mj_data, name, [x, y, z]):
                    result[name] = np.array([x, y, z])
                break
    return result


def reset_scene(mj_model: mujoco.MjModel, mj_data: mujoco.MjData,
                joint_names: Optional[list] = None,
                initial_qpos: Optional[np.ndarray] = None,
                initial_gripper: float = 0.0) -> bool:
    """
    Reset toan bo mo phong MuJoCo ve trang thai mac dinh:
    - Khoi phuc vi tri goc cua toan bo cac vat the (obj_cube, obj_box, obj_cyl, obj_sphere, obj_plate, table).
    - Reset toan bo van toc qvel va gia toc qacc ve 0.
    - Dat lai cac goc khop cua robot theo initial_qpos.
    
    Args:
        mj_model: MjModel MuJoCo.
        mj_data: MjData MuJoCo.
        joint_names: Danh sach ten cac khop cua robot (neu muon dat goc cu the).
        initial_qpos: Mang goc khop ban dau (6 khop, radians).
        initial_gripper: Do mo tay kep (meters).
        
    Returns:
        bool: True neu reset thanh cong.
    """
    if mj_model is None or mj_data is None:
        return False

    # 1. Reset toan bo qpos, qvel, act, warmup ve gia tri goc qpos0 trong XML
    mujoco.mj_resetData(mj_model, mj_data)

    # 2. Neu co chi dinh goc khop ban dau cho robot
    if initial_qpos is not None and joint_names is not None:
        q_target = np.asarray(initial_qpos, dtype=np.float64)[:len(joint_names)]
        for i, name in enumerate(joint_names):
            try:
                addr = mj_model.joint(name).qposadr[0]
                mj_data.qpos[addr] = q_target[i]
                act_id = mj_model.actuator(f"pos_j{i+1}").id
                mj_data.ctrl[act_id] = q_target[i]
            except Exception:
                pass

    # 3. Dat lai gripper neu co actuator pos_left_gripper
    try:
        g_id = mj_model.actuator("pos_left_gripper").id
        mj_data.ctrl[g_id] = float(initial_gripper)
    except Exception:
        pass

    # 4. Cap nhat dong hoc toan he thong
    mujoco.mj_forward(mj_model, mj_data)
    return True

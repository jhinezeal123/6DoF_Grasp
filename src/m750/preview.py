"""preview.py - Web xem truoc mo phong myArm M750 (http://<ip>:8081/).

6 slider x/y/z/rx/ry/rz -> mo phong cap nhat NGAY khi keo (khong doi tha chuot).
Nut realtime: doc goc khop THAT tu driver roi ve lai (bam lai -> ve simulate
don thuan). Nut sync: chay move_gripper_to THAT voi so tren slider.
Chuot: keo trai = quay, lan = zoom, keo phai = tinh tien.

Luu y EGL (da gap that):
  - phai dat MUJOCO_GL=egl TRUOC khi nap mujoco (Jetson khong X)
  - EGL context gan chat voi THREAD tao ra no -> 1 thread rieng giu renderer,
    request chi gui viec vao hang doi (bat buoc, khong phai cho dep)
  - nap mujoco NGAY tai module import: _solve_ik -> scipy.optimize keo theo
    1 ban EGL khac, nap sau se hong (eglQueryString = None)
"""
from __future__ import annotations

import atexit
import json
import os
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ.setdefault("MUJOCO_GL", "egl")  # Jetson khong X -> EGL, phai dat truoc khi nap
# aarch64: libgomp phai vao static TLS block TRUOC cac thu vien khac, neu khong
# se bao "libgomp.so.1: cannot allocate memory in static TLS block" khi nap
# cv2/mujoco. Da gap that.
try:
    import ctypes
    ctypes.CDLL("libgomp.so.1", mode=ctypes.RTLD_GLOBAL)
except OSError:
    pass

import numpy as np  # noqa: E402
import pinocchio as pin  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402

try:  # nap mujoco NGAY tai day (xem module docstring)
    import mujoco  # noqa: E402
except Exception:  # may khong co mujoco thi PreviewServer bao loi ro rang
    mujoco = None  # noqa: F811

import cv2  # noqa: E402

from .arm import MyArmM750
from .control import ArmController
from .ik import IKSolver
from .kinematics import ArmKinematics
from .spec import JOINT_NAMES, RobotSpec, model_dir

# Scene mo phong cua preview: ban MuJoCo don gian cua server (simu/assets).
# Thu muc simu/ thuoc ve tung may (khong commit); khong tim thay thi preview
# bao loi ro rang, phan con lai cua package van chay binh thuong.
SIM_SCENE = os.path.join(os.path.dirname(model_dir()), "..", "..", "simu",
                         "assets", "myarm_m750.xml")

_PAGE = """<!doctype html><meta charset=utf-8><title>myArm M750 preview</title>
<style>
 body{margin:0;background:#111;color:#ddd;font:13px system-ui;overflow:hidden}
 #v{display:block;width:100vw;height:100vh;cursor:grab}
 #v.drag{cursor:grabbing}
 #ui{position:fixed;top:10px;left:10px;background:#000b;padding:10px 12px;border-radius:8px;width:250px}
 #ui h3{margin:0 0 8px;font-size:13px;font-weight:600}
 .row{display:flex;align-items:center;gap:6px;margin:3px 0}
 .row label{width:22px;color:#8ac}
 .row input[type=range]{flex:1;min-width:0}
 .row span{width:58px;text-align:right;font-variant-numeric:tabular-nums;color:#eee}
 .num{width:62px;background:#1a222a;color:#eee;border:1px solid #456;border-radius:4px;
      padding:2px 4px;font:12px ui-monospace,monospace;text-align:right}
 .num:focus{outline:1px solid #4caf50;border-color:#4caf50}
 .num.bad{border-color:#c62828;color:#ff8a80}
 button{background:#2a3a4a;color:#ddd;border:1px solid #456;border-radius:5px;
         padding:5px 10px;cursor:pointer;margin-right:6px;font-size:12px}
 button.on{background:#2e7d32;border-color:#4caf50;color:#fff}
 button:disabled{opacity:.45;cursor:default}
 #st{position:fixed;bottom:10px;left:10px;background:#000b;padding:6px 10px;
     border-radius:6px;font-variant-numeric:tabular-nums;max-width:70vw}
</style>
<canvas id=v></canvas>
<div id=ui>
  <h3>myArm M750</h3>
  <div id=sl></div>
  <div style="margin-top:8px">
    <button id=rt>realtime</button><button id=sync>sync</button><button id=home>ve 0</button>
  </div>
</div>
<div id=st>dang tai...</div>
<script>
const AX=[['x',.3,700,'mm'],['y',-500,500,'mm'],['z',-100,700,'mm'],
          ['rx',-180,180,'do'],['ry',-180,180,'do'],['rz',-180,180,'do']];
const sl=document.getElementById('sl'),st=document.getElementById('st'),cv=document.getElementById('v');
const vals=[500,0,250,0,0,0];
let realtime=false,busy=false,cam={az:135,el:-20,d:1.1,pan:[0,0,0]};
AX.forEach(([n,lo,hi,u],i)=>{
  sl.insertAdjacentHTML('beforeend',
    `<div class=row><label>${n}</label><input type=range id=s${i} min=${lo} max=${hi} step=1 value=${vals[i]}>`
   +`<input type=number id=t${i} class=num step=1 value=${vals[i]}></div>`);
});

function draw(im){                           // im phai la Image DA DECODE, khong phai Blob:
  cv.width=im.width;cv.height=im.height;     // Blob -> drawImage ve ra canvas TRONG SUOT ->
  cv.getContext('2d').drawImage(im,0,0)      // chi thay mau nen #111 = den thui. Da gap that.
}
function decode(b){return new Promise((ok,no)=>{  // Blob -> Image
  const u=URL.createObjectURL(b),i=new Image();
  i.onload=()=>{URL.revokeObjectURL(u);ok(i)};i.onerror=no;i.src=u})}
async function frame(){                      // lay 1 khung tu server, tra Image hoac null
  const r=await fetch('render',{method:'POST',body:JSON.stringify({pos:vals,cam:cam})});
  if(!r.ok)return null;                      // 422 = ngoai tam voi, giu khung cu
  return decode(await r.blob());
}

async function tick(){                       // vong lap 1 khung/lan: khong chong len nhau
  if(busy){setTimeout(tick,20);return}
  busy=true;
  let note=msg();
  try{
    if(realtime){                            // realtime: hoi thang goc khop that tu driver
      const r=await fetch('state');
      if(r.ok){const j=await r.json();
        if(j.grip){for(let i=0;i<6;i++)setVal(i,j.grip[i])}
        else note='REALTIME KHONG DOC DUOC: '+(j.err||'?')}   // giu nguyen, khong bi msg() de
    }
    if(dirty||realtime){                     // simulate: chi ve khi slider doi
      const im=await frame();if(im)draw(im);dirty=false
    }
  }catch(e){note='loi: '+e.message}
  st.textContent=note;
  busy=false;setTimeout(tick,realtime?50:200);
}
function msg(){return `gripper  x=${vals[0].toFixed(1)} y=${vals[1].toFixed(1)} z=${vals[2].toFixed(1)}`
  +`  |  rpy=${vals[3].toFixed(1)},${vals[4].toFixed(1)},${vals[5].toFixed(1)}`
  +`  |  ${realtime?'REALTIME':'simulate'}`}

let dirty=true;
function setVal(i,v,syncSlider){              // 1 cho duy nhat ghi gia tri -> slider va o nhap luon khop
  const [n,lo,hi]=AX[i];
  v=Math.max(lo,Math.min(hi,Number(v)));
  if(!isFinite(v))return false;
  vals[i]=v;
  document.getElementById('s'+i).value=v;
  document.getElementById('t'+i).value=+v.toFixed(2);   // +..toFixed bo so 0 thua
  if(syncSlider)document.getElementById('t'+i).classList.remove('bad');
  return true;
}
AX.forEach((_,i)=>{
  document.getElementById('s'+i).oninput=e=>{
    setVal(i,e.target.value);dirty=true;if(realtime)rt.classList.remove('on'),realtime=false};
  const t=document.getElementById('t'+i);
  t.oninput=e=>{                              // go tới đâu vẽ tới đó, khong doi Enter
    const v=Number(e.target.value);
    if(e.target.value===''||!isFinite(v)){t.classList.add('bad');return}  // dang go do/dau tru
    t.classList.remove('bad');setVal(i,v);dirty=true;
    if(realtime)rt.classList.remove('on'),realtime=false};
  t.onchange=e=>{                             // roi o: ep ve gia tri hop le trong [lo,hi]
    const v=Number(e.target.value);
    if(!isFinite(v)){setVal(i,vals[i]);return}   // go chu -> tra ve so cu
    setVal(i,v);t.classList.remove('bad');dirty=true};
  t.onkeydown=e=>{if(e.key==='Enter')t.blur()};
});

document.getElementById('rt').onclick=e=>{  // bat/tat realtime
  realtime=!realtime;e.target.classList.toggle('on',realtime);dirty=true;
  if(realtime)fetch('free');               // nha port de doc state
};
document.getElementById('sync').onclick=async e=>{  // chay THAT tren tay
  e.target.disabled=true;
  try{const r=await fetch('sync',{method:'POST',body:JSON.stringify({pos:vals})});
      st.textContent=await r.text()}
  catch(err){st.textContent='sync loi: '+err.message}
  e.target.disabled=false;
};
document.getElementById('home').onclick=()=>{
  const d0=[500,0,250,0,0,0];for(let i=0;i<6;i++)setVal(i,d0[i]);dirty=true};

// chuot: keo trai = quay, lan = zoom, keo phai = tinh tien
let drag=null;
cv.oncontextmenu=e=>e.preventDefault();
cv.onpointerdown=e=>{drag={x:e.clientX,y:e.clientY,b:e.button};cv.classList.add('drag');
                     cv.setPointerCapture(e.pointerId)};
cv.onpointerup=e=>{drag=null;cv.classList.remove('drag')};
cv.onpointermove=e=>{
  if(!drag)return;
  const dx=e.clientX-drag.x,dy=e.clientY-drag.y;drag={x:e.clientX,y:e.clientY,b:drag.b};
  if(drag.b===0){cam.az=(cam.az-dx*0.4)%360;cam.el=Math.max(-89,Math.min(89,cam.el+dy*0.4))}
  else{cam.pan[0]-=dx*cam.d*0.0015;cam.pan[2]+=dy*cam.d*0.0015}
  dirty=true;
};
cv.onwheel=e=>{e.preventDefault();cam.d=Math.max(.15,Math.min(6,cam.d*(1+Math.sign(e.deltaY)*0.1)));
               dirty=true};
tick();
</script>"""


class PreviewServer:
    """Server web xem truoc mo phong. start()/stop() la method that (khong con monkey-patch)."""

    def __init__(self, port: int = 8081, w: int = 960, h: int = 720,
                 read_hz: float = 8.0, scene: str = SIM_SCENE,
                 controller: ArmController | None = None) -> None:
        self.port, self.w, self.h, self.read_hz = port, w, h, read_hz
        self.scene = scene
        self.spec = controller.spec if controller else RobotSpec()
        self.kin = controller.kin if controller else ArmKinematics(self.spec)
        self.ik = IKSolver(self.kin)
        self.controller = controller
        # arm chi mo khi /state hoac /sync duoc goi (realtime/sync can doc that;
        # preview don thuan khong chiem tay)
        self._arm = controller.arm if controller else MyArmM750(self.spec)
        self._jobs = queue.Queue()
        self._out = {}
        self._cam = None
        self._srv = None

    def start(self) -> "PreviewServer":
        """Khoi dong render worker + HTTP server. Tra self. Idempotent."""
        if mujoco is None:
            print("khong nap duoc mujoco -> khong mo duoc preview")
            return self
        if self._srv is not None:
            return self
        m = mujoco.MjModel.from_xml_path(self.scene)
        d = mujoco.MjData(m)
        self._cam = mujoco.MjvCamera()
        box = {}

        def worker():
            # EGL context gan chat voi THREAD tao ra no. ThreadingHTTPServer phuc vu moi
            # request o thread khac -> eglMakeCurrent bao EGL_BAD_ACCESS. Nen: 1 thread
            # rieng giu renderer, request chi gui viec vao hang doi. Bat buoc.
            r = mujoco.Renderer(m, self.h, self.w)  # tao TRONG thread nay
            mujoco.mjv_defaultFreeCamera(m, self._cam)
            base = d.xpos[m.body("base_link").id].copy()
            tip = self.kin.grip_mm(self.kin.fk_tool0([0, 30, -20, 0, 80, 0])) / 1000.0
            self._cam.lookat[:] = (base + base + tip) / 2.0
            self._cam.distance, self._cam.azimuth, self._cam.elevation = 1.1, 135.0, -20.0
            box["ready"] = True
            while True:
                job = self._jobs.get()
                if job is None:
                    break
                key, q, c = job
                try:
                    if c:
                        self._cam.azimuth = float(c.get("az", self._cam.azimuth))
                        self._cam.elevation = max(-89.0, min(89.0, float(c.get("el", self._cam.elevation))))
                        self._cam.distance = max(0.15, min(6.0, float(c.get("d", self._cam.distance))))
                        if c.get("pan"):
                            self._cam.lookat[:] = np.array(self._cam.lookat, float) + np.array(c["pan"], float) * 0.5
                    d.qpos[:] = 0
                    for i, n in enumerate(JOINT_NAMES):
                        d.qpos[m.jnt_qposadr[m.joint(n).id]] = np.radians(q[i])
                    for n in ("left_gripper_joint", "right_gripper_joint"):
                        d.qpos[m.jnt_qposadr[m.joint(n).id]] = 0.0173
                    mujoco.mj_forward(m, d)
                    r.update_scene(d, self._cam)
                    self._out[key] = cv2.imencode(
                        ".jpg", cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
                except Exception as e:
                    self._out[key] = e
            r.close()

        threading.Thread(target=worker, daemon=True).start()
        while not box.get("ready"):
            time.sleep(0.05)
        self._seed = [[0.0, 30.0, -20.0, 0.0, 80.0, 0.0]]  # nghiem IK truoc do, lam diem khoi dau cho lan sau

        server = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _body(self):
                return json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")

            def _ok(self, ctype, data, code=200):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path.startswith("/state"):
                    # Mo port NGAY tai day neu dang ranh: start() khong mo luc khoi
                    # dong (de khong chiem tay khi nguoi dung chi muon xem mo phong),
                    # nhung realtime thi can.
                    if not server._arm.is_open:
                        try:
                            server._arm.open()
                        except Exception as e:
                            return self._ok("application/json",
                                            json.dumps({"err": str(e)[:200]}).encode())
                    q = server._arm.q_deg
                    if q is None:
                        return self._ok("application/json", b'{"err":"doc goc loi"}')
                    return self._ok("application/json",
                                    json.dumps({"grip": server._gripper_pose()}).encode())
                if self.path.startswith("/free"):
                    return self._ok("text/plain", b"ok")  # goc khop di qua fetch(), khong chiem port o day
                self._ok("text/html; charset=utf-8", _PAGE.encode())

            def do_POST(self):
                b = self._body()
                if self.path.startswith("/render"):
                    c = b.get("cam") or {}
                    pos = [float(v) for v in b["pos"]]
                    Rg = R.from_euler("xyz", pos[3:], degrees=True).as_matrix()
                    p_tool = np.array(pos[:3]) - Rg @ np.array([0.0, 0.0, server.spec.grip_l_mm])
                    # Seed IK lay tu chinh khung hinh cuoi (khong doc q that: ham do MO PORT
                    # serial -> preview se chet neu tien trinh khac dang giu tay). Preview chi ve.
                    q, ep, eo = server.ik.solve(
                        pin.SE3(Rg, p_tool / 1000.0), server._seed[0], n_restart=3)
                    # Phai kiem tra sai so: IK luon tra ve 1 nghiem nao do, ke ca khi
                    # q5 dung tran (ry lon) -> ve ra tu the SAI ma nhin khong biet. Da gap that.
                    if q is None or ep > 1.0 or eo > 1.0:
                        return self._ok("text/plain", b"", 422)  # ngoai tam voi: giu khung cu
                    server._seed[0] = q
                    return self._ok("image/jpeg", server._snap(q, c))
                if self.path.startswith("/sync"):
                    pos = [float(v) for v in b["pos"]]
                    ok = server._move_gripper(*pos)
                    return self._ok("text/plain",
                                    ("sync: %s" % ("TOI" if ok else "KHONG TOI")).encode())
                self._ok("text/plain", b"", 404)

        self._srv = ThreadingHTTPServer(("0.0.0.0", self.port), H)
        self._srv.daemon_threads = True
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()
        atexit.register(self.stop)
        print("preview: http://0.0.0.0:%d/   (server KHONG tu mo port serial; /sync chi chay khi da mo)"
              % self.port)
        return self

    def _gripper_pose(self):
        q = self._arm.q_deg
        if q is None:
            return None
        M = self.kin.fk_tool0(q)
        return ([round(float(v), 2) for v in self.kin.grip_mm(M)]
                + [round(float(v), 2) for v in R.from_matrix(M.rotation).as_euler("xyz", degrees=True)])

    def _move_gripper(self, x, y, z, rx=0.0, ry=0.0, rz=0.0):
        if self.controller is not None:
            return self.controller.move_gripper_to(x, y, z, rx, ry, rz)
        # fallback khi khoi tao khong co controller: tao tam
        ctrl = ArmController(arm=self._arm, kin=self.kin, spec=self.spec)
        return ctrl.move_gripper_to(x, y, z, rx, ry, rz)

    def _snap(self, q, c=None):
        """Nho thread render ve 1 khung (goc khop do) -> bytes JPEG. Nem loi neu render hong."""
        key = object()
        self._jobs.put((key, q, c))
        while key not in self._out:
            time.sleep(0.005)
        v = self._out.pop(key)
        if isinstance(v, Exception):
            raise v
        return v

    def stop(self) -> None:
        """Dung server. Idempotent. Khong dong serial (tay giu trang thai)."""
        if self._srv is None:
            return
        self._jobs.put(None)
        self._srv.shutdown()
        self._srv.server_close()
        self._srv = None


def preview(port: int = 8081, w: int = 960, h: int = 720,
            read_hz: float = 8.0) -> PreviewServer | None:
    """Compat voi ham preview() cu: tra PreviewServer DA start."""
    return PreviewServer(port, w, h, read_hz).start()


__all__ = ["PreviewServer", "preview", "SIM_SCENE"]

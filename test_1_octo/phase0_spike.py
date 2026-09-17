#!/usr/bin/env python3
"""
Phase 0 spike - chot backend Octo tren ktmt.

Tra loi 4 cau hoi cua Phase 0:
  1. Model co load duoc qua transformers.AutoModel khong?
  2. Neu khong, API chinh thuc OctoModel.load_pretrained("hf://...") co chay khong?
  3. jax.devices() nhan GPU Jetson hay chi CPU?
  4. Load time / inference latency / memory.

Chay:
    /home/ktmt-agx-xv/Data/octo_env/bin/python phase0_spike.py [n_latency_runs]

Ghi ket qua ra stdout va phase0_result.json (cung thu muc).
"""

import json
import os
import resource
import sys
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np

MODEL_ID = "rail-berkeley/octo-base-1.5"
HF_PATH = "hf://" + MODEL_ID
INSTRUCTION = "pick up the red cube"
WINDOW = 2
RESULT = {}


def head(title):
    print("\n" + "=" * 62)
    print("  " + title)
    print("=" * 62, flush=True)


def rss_mb():
    """Peak RSS cua tien trinh nay, MB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def mem_available_mb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024.0
    return float("nan")


# ---------------------------------------------------------------- 0. moi truong
head("0. MOI TRUONG")
import platform  # noqa: E402

RESULT["python"] = sys.version.split()[0]
RESULT["executable"] = sys.executable
RESULT["arch"] = platform.machine()
RESULT["cpu_count"] = os.cpu_count()
print("python      :", RESULT["python"], "->", RESULT["executable"])
print("arch        :", RESULT["arch"])
print("cpu_count   :", RESULT["cpu_count"])
print("rss         : %.0f MB" % rss_mb())

# ---------------------------------------------------------------- 1. JAX
head("1. JAX  (Phase 0 buoc 3: jax.devices())")
import jax  # noqa: E402
import jaxlib  # noqa: E402

RESULT["jax"] = jax.__version__
RESULT["jaxlib"] = jaxlib.__version__
RESULT["backend"] = jax.default_backend()
RESULT["devices"] = [str(d) for d in jax.devices()]
try:
    RESULT["device_kind"] = jax.devices()[0].device_kind
except Exception as exc:  # pragma: no cover
    RESULT["device_kind"] = "unknown (%s)" % exc
print("jax         :", RESULT["jax"])
print("jaxlib      :", RESULT["jaxlib"])
print("backend     :", RESULT["backend"])
print("devices     :", RESULT["devices"])
print("device_kind :", RESULT["device_kind"])
print("rss         : %.0f MB" % rss_mb())

# ------------------------------------------- 2. transformers.AutoModel duoc khong
head("2. transformers.AutoModel  (Phase 0 buoc 1)")
try:
    from transformers import AutoConfig, AutoModel

    try:
        cfg = AutoConfig.from_pretrained(MODEL_ID)
        RESULT["autoconfig"] = "OK: " + type(cfg).__name__
        print("AutoConfig  : OK ->", type(cfg).__name__)
    except Exception as exc:
        RESULT["autoconfig"] = "FAILED: %s" % type(exc).__name__
        print("AutoConfig  : FAILED -> %s: %s" % (type(exc).__name__, str(exc)[:300]))

    try:
        mdl = AutoModel.from_pretrained(MODEL_ID)
        RESULT["automodel"] = "OK: " + type(mdl).__name__
        print("AutoModel   : OK ->", type(mdl).__name__)
    except Exception as exc:
        RESULT["automodel"] = "FAILED: %s" % type(exc).__name__
        print("AutoModel   : FAILED -> %s: %s" % (type(exc).__name__, str(exc)[:300]))
except Exception as exc:
    RESULT["automodel"] = "transformers unusable: %s" % exc
    print("transformers khong dung duoc:", exc)

# ------------------------------------------------- 3. tai checkpoint (do rieng)
head("3. TAI CHECKPOINT  (do rieng, khong tinh vao load time)")
t0 = time.time()
try:
    from huggingface_hub import snapshot_download

    ckpt_dir = snapshot_download(MODEL_ID)
    dl_s = time.time() - t0
    size_mb = sum(
        os.path.getsize(os.path.join(root, f))
        for root, _, files in os.walk(ckpt_dir)
        for f in files
    ) / 1e6
    RESULT["download_s"] = dl_s
    RESULT["checkpoint_dir"] = ckpt_dir
    RESULT["checkpoint_mb"] = size_mb
    print("snapshot    : %s" % ckpt_dir)
    print("dung luong  : %.1f MB" % size_mb)
    print("thoi gian   : %.1f s" % dl_s)
except Exception as exc:
    RESULT["download_s"] = None
    print("snapshot_download loi (co the da cache san):", exc)
print("rss         : %.0f MB" % rss_mb())

# ------------------------------------------------- 4. OctoModel.load_pretrained
head("4. OctoModel.load_pretrained  (Phase 0 buoc 2)")
from octo.model.octo_model import OctoModel  # noqa: E402

t0 = time.time()
model = OctoModel.load_pretrained(HF_PATH)
RESULT["load_s"] = time.time() - t0
RESULT["rss_after_load_mb"] = rss_mb()
print("load time   : %.1f s" % RESULT["load_s"])
print("rss         : %.0f MB" % RESULT["rss_after_load_mb"])
print("MemAvailable: %.0f MB" % mem_available_mb())

try:
    spec = model.get_pretty_spec()
    RESULT["spec"] = spec
    print("\n--- observation/action spec ---")
    print(spec)
except Exception as exc:
    print("get_pretty_spec loi:", exc)

n_params = None
try:
    leaves = jax.tree_util.tree_leaves(model.params)
    n_params = int(sum(np.prod(p.shape) for p in leaves))
    RESULT["n_params"] = n_params
    print("\nparams      : %s (%.1fM)" % (n_params, n_params / 1e6))
except Exception as exc:
    print("dem params loi:", exc)

# ------------------------------------------------- 5. inference
head("5. INFERENCE  (Phase 0 buoc 4)")
n_runs = int(sys.argv[1]) if len(sys.argv) > 1 else 3

# Anh gia 256x256x3, history 2 frame. Bo image_wrist: pad_mask_dict = False cho key
# vang mat (theo FAQ cua Octo), dung voi giai doan chua co camera wrist.
img = np.zeros((1, WINDOW, 256, 256, 3), dtype=np.uint8)
observation = {
    "image_primary": img,
    "timestep_pad_mask": np.full((1, WINDOW), True, dtype=bool),
}
task = model.create_tasks(texts=[INSTRUCTION])
stats = model.dataset_statistics["bridge_dataset"]["action"]
RESULT["instruction"] = INSTRUCTION
RESULT["obs_shapes"] = {k: list(np.asarray(v).shape) for k, v in observation.items()}
print("instruction :", INSTRUCTION)
for k, v in observation.items():
    print("obs[%-17s] %s %s" % (k + "]", np.asarray(v).shape, np.asarray(v).dtype))

print("\n-- run 1 (bao gom JIT compile) --", flush=True)
t0 = time.time()
actions = model.sample_actions(observation, task, unnormalization_statistics=stats)
first_s = time.time() - t0
actions = np.asarray(actions)
RESULT["t_first_s"] = first_s
print("  thoi gian : %.1f s" % first_s)

print("\n-- %d run tiep theo (steady state) --" % n_runs, flush=True)
times = []
for i in range(n_runs):
    t0 = time.time()
    a = np.asarray(
        model.sample_actions(observation, task, unnormalization_statistics=stats)
    )
    dt = time.time() - t0
    times.append(dt)
    print("  run %d: %.1f s" % (i + 2, dt), flush=True)

RESULT["t_steady_s"] = times
RESULT["t_steady_mean_s"] = float(np.mean(times))
RESULT["hz"] = 1.0 / float(np.mean(times))
RESULT["rss_after_infer_mb"] = rss_mb()
RESULT["actions_shape"] = list(actions.shape)
RESULT["actions_dtype"] = str(actions.dtype)
RESULT["actions_finite"] = bool(np.isfinite(actions).all())
RESULT["actions"] = actions.tolist()

print("\nshape       :", actions.shape, actions.dtype)
print("all finite  :", RESULT["actions_finite"])
print("mean latency: %.1f s  ->  %.3f Hz" % (RESULT["t_steady_mean_s"], RESULT["hz"]))
print("rss peak    : %.0f MB" % RESULT["rss_after_infer_mb"])
labels = ["x", "y", "z", "yaw", "pitch", "roll", "grasp"]
print("\n4 action dau tien (chunk, da unnormalize):")
for i, row in enumerate(actions[0]):
    print("  [%d] " % i + "  ".join("%-6s=%+.4f" % (l, v) for l, v in zip(labels, row)))

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phase0_result.json")
with open(out, "w") as fh:
    json.dump(RESULT, fh, indent=2)
print("\nJSON -> %s" % out)
print("\nPHASE0_SPIKE_DONE")

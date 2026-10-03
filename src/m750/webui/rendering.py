"""Process render MuJoCo, độc lập với request HTTP và trạng thái UI."""

import time

import numpy as np

from m750.media import ImageCompressor


def _mujoco_render_worker(scene_path, camera_specs, fps, command_recv, frame_send):
    """Render MuJoCo frames outside the HTTP/ROS process.

    On the headless Jetson, the first MuJoCo/EGL render can block while holding
    the Python interpreter lock.  Keeping it in the web process prevents the
    RosBridge spin thread from running, so the UI reports NaN even though the
    driver is publishing feedback.  This worker owns its own model, data and
    EGL contexts; the parent only sends qpos snapshots and receives JPEGs.
    """
    renderers = []
    try:
        # Imports must happen in the spawned process after it has its own EGL
        # context.  Importing MuJoCo in the parent is still needed by FakeRobot.
        import mujoco

        model = mujoco.MjModel.from_xml_path(scene_path)
        data = mujoco.MjData(model)
        for key, camera_name, width, height in camera_specs:
            renderer = mujoco.Renderer(model, height=int(height), width=int(width))
            try:
                flags = renderer._scene.flags
                flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
                flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
            except Exception:
                pass
            cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
            renderers.append((key, renderer, cam_id))

        latest_qpos = None
        preview_active = False
        preview_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tool0_preview")
        interval = 1.0 / max(float(fps), 1.0)
        next_frame = time.monotonic()
        while True:
            # Drain the pipe so a burst of slider updates cannot make the
            # worker render stale poses.
            while command_recv.poll():
                packet = command_recv.recv()
                if packet is None:
                    return
                if isinstance(packet, tuple):
                    packet, preview_active = packet
                latest_qpos = np.asarray(packet, dtype=np.float64).reshape(-1)

            if latest_qpos is not None and latest_qpos.size == model.nq:
                data.qpos[:] = latest_qpos
                mujoco.mj_forward(model, data)
            if preview_site_id >= 0:
                model.site_rgba[preview_site_id, 3] = 0.95 if preview_active else 0.0

            for key, renderer, cam_id in renderers:
                if cam_id >= 0:
                    renderer.update_scene(data, camera=cam_id)
                else:
                    renderer.update_scene(data)
                frame = renderer.render()
                jpeg = ImageCompressor.encode_jpeg(frame, quality=75)
                if jpeg:
                    try:
                        frame_send.send((key, jpeg))
                    except (BrokenPipeError, EOFError, OSError):
                        return

            next_frame += interval
            wait = next_frame - time.monotonic()
            if wait > 0:
                # poll() both sleeps without a busy loop and lets the parent
                # stop the worker promptly by sending the None sentinel.
                if command_recv.poll(wait):
                    packet = command_recv.recv()
                    if packet is None:
                        return
                    if isinstance(packet, tuple):
                        packet, preview_active = packet
                    latest_qpos = np.asarray(packet, dtype=np.float64).reshape(-1)
            else:
                next_frame = time.monotonic()
    except (EOFError, OSError, BrokenPipeError):
        pass
    except Exception as exc:
        print(f"[WebRenderWorker] Loi render: {exc}", flush=True)
    finally:
        for _, renderer, _ in renderers:
            try:
                renderer.close()
            except Exception:
                pass

"""Keyboard teleoperation page for collecting demonstrations.

The page deliberately stays separate from the normal Web Control page.  It
shows the two physical USB camera streams and sends small Cartesian deltas to
the server, where the current TCP target is accumulated and validated before
``Robot.set_tcp_pose()`` is called.
"""
from __future__ import annotations

import html as html_lib
from typing import Iterable, Tuple


def render_html(cameras: Iterable[Tuple[str, str]]) -> bytes:
    """Render the standalone teleoperation page."""
    windows = []
    for key, device in cameras:
        safe_key = html_lib.escape(str(key), quote=True)
        safe_device = html_lib.escape(str(device), quote=True)
        windows.append(
            '<article class="camera-card">'
            f'<div class="camera-title">Camera {safe_key}</div>'
            f'<img src="/stream/cam_{safe_key}.mjpg" alt="Camera {safe_key}">'
            f'<div class="camera-device">{safe_device}</div>'
            '</article>'
        )
    camera_windows = "".join(windows)
    if not camera_windows:
        camera_windows = (
            '<article class="camera-card camera-empty">'
            'No USB camera is configured on this server.'
            '</article>'
        )

    template = r'''<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>myArm M750 - Teleop Data Collection</title>
  <style>
    * { box-sizing: border-box; }
    body {
      margin: 0;
      padding: 18px;
      background: #0d1117;
      color: #c9d1d9;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 16px;
      padding-bottom: 12px;
      border-bottom: 1px solid #30363d;
    }
    h1, h2, p { margin: 0; }
    h1 { color: #58a6ff; font-size: 1.3rem; font-weight: 650; }
    h2 { color: #f0f6fc; font-size: 1rem; margin-bottom: 12px; }
    .subtitle { color: #8b949e; font-size: .78rem; margin-top: 4px; }
    .nav-link {
      color: #c9d1d9;
      border: 1px solid #30363d;
      border-radius: 6px;
      padding: 8px 12px;
      text-decoration: none;
      font-size: .8rem;
      white-space: nowrap;
    }
    .nav-link:hover { color: #fff; border-color: #58a6ff; }
    .camera-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
      margin-bottom: 14px;
    }
    .camera-card, .panel {
      background: #161b22;
      border: 1px solid #30363d;
      border-radius: 8px;
      padding: 12px;
    }
    .camera-title {
      color: #f0f6fc;
      font-size: .82rem;
      font-weight: 650;
      margin-bottom: 8px;
    }
    .camera-card img {
      display: block;
      width: 100%;
      aspect-ratio: 4 / 3;
      object-fit: contain;
      background: #000;
      border-radius: 5px;
    }
    .camera-device {
      color: #8b949e;
      font: .7rem monospace;
      margin-top: 7px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .camera-empty { min-height: 160px; color: #8b949e; }
    .teleop-grid {
      display: grid;
      grid-template-columns: minmax(280px, .9fr) minmax(280px, 1.1fr);
      gap: 14px;
    }
    .status-row {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      padding: 7px 0;
      border-bottom: 1px solid #21262d;
      font: .78rem monospace;
    }
    .status-row:last-child { border-bottom: 0; }
    .status-label { color: #8b949e; }
    .status-value { color: #58a6ff; text-align: right; }
    .status-value.ok { color: #3fb950; }
    .status-value.warn { color: #d29922; }
    .status-value.error { color: #f85149; }
    .key-layout {
      display: grid;
      grid-template-columns: repeat(3, 54px);
      grid-template-rows: repeat(2, 48px);
      gap: 7px;
      justify-content: center;
      margin: 8px 0 15px;
    }
    .key {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      background: #21262d;
      border: 1px solid #484f58;
      border-radius: 6px;
      color: #f0f6fc;
      font-weight: 700;
      user-select: none;
    }
    .key small { color: #8b949e; font: .62rem monospace; margin-top: 2px; }
    .key.held { background: #1f6feb; border-color: #58a6ff; }
    .key-w { grid-column: 2; grid-row: 1; }
    .key-a { grid-column: 1; grid-row: 2; }
    .key-s { grid-column: 2; grid-row: 2; }
    .key-d { grid-column: 3; grid-row: 2; }
    .z-key {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 7px;
      margin-bottom: 13px;
      color: #8b949e;
      font-size: .75rem;
    }
    .z-key kbd, kbd {
      padding: 3px 7px;
      background: #21262d;
      border: 1px solid #484f58;
      border-radius: 4px;
      color: #f0f6fc;
      font: .72rem monospace;
    }
    .settings {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      margin: 12px 0;
      color: #8b949e;
      font-size: .78rem;
    }
    .settings input {
      width: 80px;
      padding: 5px 7px;
      background: #0d1117;
      border: 1px solid #30363d;
      border-radius: 5px;
      color: #c9d1d9;
      text-align: right;
    }
    button {
      padding: 7px 10px;
      border: 1px solid #30363d;
      border-radius: 5px;
      background: #21262d;
      color: #c9d1d9;
      cursor: pointer;
      font-size: .75rem;
    }
    button:hover { border-color: #58a6ff; color: #fff; }
    button.danger { color: #ff7b72; border-color: #f85149; }
    .button-row { display: flex; gap: 8px; flex-wrap: wrap; }
    .help {
      color: #8b949e;
      font-size: .74rem;
      line-height: 1.45;
      margin-top: 13px;
    }
    .error-box {
      min-height: 19px;
      margin-top: 10px;
      color: #ff7b72;
      font-size: .74rem;
      line-height: 1.35;
    }
    @media (max-width: 760px) {
      body { padding: 10px; }
      header { align-items: flex-start; }
      .camera-grid, .teleop-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Teleop Data Collection</h1>
      <p class="subtitle">Điều khiển TCP bằng bàn phím và quan sát đồng thời hai camera.</p>
    </div>
    <a class="nav-link" href="/">← Web Control</a>
  </header>

  <main>
    <section class="camera-grid">
      <!-- CAMERA_WINDOWS -->
    </section>

    <section class="teleop-grid">
      <div class="panel">
        <h2>Keyboard Teleop</h2>
        <div class="key-layout">
          <div class="key key-w" data-code="KeyW">W<small>+Y</small></div>
          <div class="key key-a" data-code="KeyA">A<small>-X</small></div>
          <div class="key key-s" data-code="KeyS">S<small>-Y</small></div>
          <div class="key key-d" data-code="KeyD">D<small>+X</small></div>
        </div>
        <div class="z-key"><kbd>Z</kbd><span>+Z</span><kbd>Shift + Z</kbd><span>-Z</span></div>
        <div class="settings">
          <label for="stepMm">Bước dịch chuyển</label>
          <span><input id="stepMm" type="number" min="1" max="50" step="1" value="5"> mm</span>
        </div>
        <div class="button-row">
          <button id="btnReset">Reset theo pose robot</button>
          <button id="btnStop" class="danger">DỪNG KHẨN</button>
        </div>
        <p class="help">
          Click vào trang rồi giữ phím để di chuyển liên tục. Thả phím để dừng gửi lệnh.
          Teleop giữ quaternion cố định <code>[0, 0, 0, 1]</code>, gripper hướng vuông góc xuống.
        </p>
        <div id="errorBox" class="error-box"></div>
      </div>

      <div class="panel">
        <h2>Pose và trạng thái</h2>
        <div class="status-row"><span class="status-label">Target TCP</span><span id="targetPose" class="status-value">--</span></div>
        <div class="status-row"><span class="status-label">TCP feedback</span><span id="actualPose" class="status-value">--</span></div>
        <div class="status-row"><span class="status-label">Quaternion cố định</span><span class="status-value">[0, 0, 0, 1]</span></div>
        <div class="status-row"><span class="status-label">Safety</span><span id="safety" class="status-value">--</span></div>
        <div class="status-row"><span class="status-label">Hardware</span><span id="hardware" class="status-value">--</span></div>
        <div class="status-row"><span class="status-label">Phím đang giữ</span><span id="heldKeys" class="status-value">--</span></div>
      </div>
    </section>
  </main>

  <script>
    const held = new Set();
    let busy = false;
    let targetPosition = null;

    function setError(message) {
      document.getElementById("errorBox").innerText = message || "";
    }

    function post(url, body) {
      return fetch(url, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: body ? JSON.stringify(body) : undefined
      }).then(r => r.json()).then(data => {
        if (!data || data.status === "error") {
          setError((data && data.message) || "Lệnh teleop thất bại");
        } else {
          setError("");
        }
        return data;
      }).catch(err => {
        setError("Lỗi kết nối: " + err);
        return null;
      });
    }

    function fmtPose(p) {
      if (!p || p.length !== 3) return "--";
      return "[" + p.map(v => Number(v).toFixed(3)).join(", ") + "] m";
    }

    function refreshHeldLabels() {
      document.querySelectorAll(".key[data-code]").forEach(el => {
        el.classList.toggle("held", held.has(el.dataset.code));
      });
      const labels = Array.from(held)
        .filter(code => !code.startsWith("Shift"))
        .map(code => code.replace("Key", ""));
      if (held.has("ShiftLeft") || held.has("ShiftRight")) labels.push("Shift");
      document.getElementById("heldKeys").innerText = labels.length ? labels.join(" + ") : "--";
    }

    function getDelta() {
      const direction = [
        (held.has("KeyD") ? 1 : 0) - (held.has("KeyA") ? 1 : 0),
        (held.has("KeyW") ? 1 : 0) - (held.has("KeyS") ? 1 : 0),
        held.has("KeyZ") ? ((held.has("ShiftLeft") || held.has("ShiftRight")) ? -1 : 1) : 0
      ];
      if (!direction.some(v => v !== 0)) return null;
      const mm = Math.min(50, Math.max(1, Number(document.getElementById("stepMm").value) || 5));
      document.getElementById("stepMm").value = mm;
      const step = mm / 1000.0;
      return direction.map(v => v * step);
    }

    function sendHeldStep() {
      const delta = getDelta();
      if (!delta || busy) return;
      busy = true;
      post("/api/teleop/step", {dx: delta[0], dy: delta[1], dz: delta[2]})
        .then(data => {
          if (data && data.status === "ok" && data.position) targetPosition = data.position;
          busy = false;
          updatePoseLabels();
        });
    }

    function updatePoseLabels(data) {
      if (data) {
        if (data.position) targetPosition = data.position;
        document.getElementById("actualPose").innerText = fmtPose(data.actual_position);
        const safety = document.getElementById("safety");
        safety.innerText = data.is_armed ? (data.safety_state + " — armed") : (data.safety_state + " — cần khôi phục");
        safety.className = "status-value " + (data.is_armed ? "ok" : "warn");
        const hardware = document.getElementById("hardware");
        hardware.innerText = data.is_real_connected ? "Online" : "Offline";
        hardware.className = "status-value " + (data.is_real_connected ? "ok" : "error");
      }
      document.getElementById("targetPose").innerText = fmtPose(targetPosition);
    }

    function fetchStatus() {
      fetch("/api/teleop/status")
        .then(r => r.json())
        .then(data => updatePoseLabels(data))
        .catch(err => setError("Lỗi đọc trạng thái: " + err));
    }

    function clearHeld() {
      held.clear();
      refreshHeldLabels();
    }

    window.addEventListener("keydown", event => {
      const code = event.code;
      if (!["KeyW", "KeyA", "KeyS", "KeyD", "KeyZ", "ShiftLeft", "ShiftRight"].includes(code)) return;
      if (event.target && ["INPUT", "TEXTAREA", "SELECT"].includes(event.target.tagName)) return;
      event.preventDefault();
      held.add(code);
      refreshHeldLabels();
      sendHeldStep();
    });
    window.addEventListener("keyup", event => {
      if (held.has(event.code)) {
        event.preventDefault();
        held.delete(event.code);
        refreshHeldLabels();
      }
    });
    window.addEventListener("blur", clearHeld);
    document.addEventListener("visibilitychange", () => { if (document.hidden) clearHeld(); });

    document.getElementById("btnReset").addEventListener("click", () => {
      clearHeld();
      post("/api/teleop/reset").then(data => { if (data) updatePoseLabels(data); });
    });
    document.getElementById("btnStop").addEventListener("click", () => {
      clearHeld();
      post("/api/stop").then(() => fetchStatus());
    });

    setInterval(sendHeldStep, 180);
    fetchStatus();
    setInterval(fetchStatus, 400);
  </script>
</body>
</html>
'''
    return template.replace("<!-- CAMERA_WINDOWS -->", camera_windows).encode("utf-8")


__all__ = ["render_html"]

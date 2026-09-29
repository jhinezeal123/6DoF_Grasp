"""Offline Unix-socket integration checks; no robot or model is started."""

from __future__ import annotations

import json
import socket
import threading
from pathlib import Path

import numpy as np
import pytest

from m750 import PerceptionRequest
from m750.perception.adapters.grasppose import GraspPosePerceptionAdapter
from m750.perception.adapters.grasppose_worker import WorkerError, WorkerGraspEstimator


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="Unix sockets are unavailable")
def test_worker_estimator_runs_in_separate_process_protocol(tmp_path):
    socket_path = tmp_path / "worker.sock"
    received = []
    image_exists_during_request = []
    errors = []
    responses = [
        {"ok": True, "state": "ready", "prompt_ids": ["cube"]},
        {
            "ok": True,
            "run_id": "offline-run",
            "grasps": [{
                "score": 0.9,
                "width_m": 0.03,
                "translation_m": [0.1, 0.2, 0.3],
                "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            }],
            "depth_m": 0.4,
            "detection_count": 1,
            "mask_pixels": 12,
            "grasp_count": 1,
            "server_ms": 8.0,
        },
    ]

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        listener.bind(str(socket_path))
        listener.listen(2)

        def fake_worker():
            try:
                for response in responses:
                    connection, _ = listener.accept()
                    with connection:
                        with connection.makefile("rb") as stream:
                            request = json.loads(stream.readline())
                        received.append(request)
                        if request["op"] == "infer":
                            image_exists_during_request.append(Path(request["image"]).is_file())
                        connection.sendall((json.dumps(response) + "\n").encode("utf-8"))
            except Exception as exc:
                errors.append(exc)

        thread = threading.Thread(target=fake_worker, daemon=True)
        thread.start()
        adapter = GraspPosePerceptionAdapter(WorkerGraspEstimator(socket_path, timeout=3))
        adapter.open()
        result = adapter.infer(PerceptionRequest(
            image=np.zeros((2, 3, 3), dtype=np.uint8),
            prompt_id="cube",
            camera_matrix=np.array([[500, 0, 320], [0, 510, 240], [0, 0, 1]]),
            camera_matrix_size=(640, 480),
            camera_from_volume=np.eye(4),
            top=2,
        ))
        adapter.close()
        thread.join(timeout=3)

    assert not thread.is_alive()
    assert not errors
    assert received[0] == {"op": "status"}
    assert received[1]["camera_k"] == [500.0, 510.0, 320.0, 240.0]
    assert received[1]["camera_k_size"] == [640, 480]
    assert received[1]["top"] == 2
    assert received[1]["render"] is False
    assert image_exists_during_request == [True]
    assert not Path(received[1]["image"]).exists()
    assert result.depth_m == pytest.approx(0.4)
    assert result.grasps[0].position_m == pytest.approx((0.1, 0.2, 0.3))
    assert result.grasps[0].quaternion_xyzw == pytest.approx((0, 0, 0, 1))
    assert result.raw.request_id == "offline-run"


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="Unix sockets are unavailable")
def test_missing_worker_socket_reports_location(tmp_path):
    missing = tmp_path / "missing.sock"
    estimator = WorkerGraspEstimator(missing, timeout=1)
    with pytest.raises(WorkerError, match="pipeline worker socket not found"):
        estimator.load()

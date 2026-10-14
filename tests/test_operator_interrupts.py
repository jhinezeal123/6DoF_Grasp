"""SIGINT thật giữa menu và process con; không mở phần cứng."""
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHILD = """import os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'src'))
from m750.robot.cli import main
from m750.robot.contracts import EmergencyStopController
root = Path(sys.argv[1])
(root / 'child.pid').write_text(str(os.getpid()))
def record(value):
    with (root / 'events').open('a') as stream:
        stream.write(value + '\\n')
class Driver(EmergencyStopController):
    def move_joints(self, joints):
        record('ready')
        while True:
            time.sleep(.05)
    def emergency_stop(self):
        record('stop-begin')
        time.sleep(float(sys.argv[2]))
        record('stop-complete')
        return True
    def close(self):
        record('close')
raise SystemExit(main(['joints', '0', '0', '0', '0', '0', '0'],
                      driver_factory=lambda args: Driver()))
"""
PARENT = """import sys
from pathlib import Path
import start
from _m750_menu.terminal import CommandRunner, Terminal
try:
    CommandRunner(Path.cwd(), Terminal()).run(
        [sys.executable, sys.argv[1], sys.argv[2], sys.argv[3]],
        capture=sys.argv[4] == 'capture',
    )
except KeyboardInterrupt:
    raise SystemExit(130)
"""


class OperatorInterruptTests(unittest.TestCase):
    def launch(self, directory, capture=True, delay=1):
        child = directory / "child.py"
        child.write_text(CHILD, encoding="utf-8")
        parent = directory / "parent.py"
        parent.write_text(PARENT, encoding="utf-8")
        process = subprocess.Popen(
            [sys.executable, str(parent), str(child), str(directory), str(delay),
             "capture" if capture else "stream"],
            cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, start_new_session=True,
            env={**os.environ, "PYTHONPATH": str(ROOT)},
        )
        self.addCleanup(self.cleanup, process, directory)
        self.wait_event(process, directory, "ready")
        return process

    @staticmethod
    def cleanup(process, directory):
        child_pid = directory / "child.pid"
        if child_pid.exists():
            try:
                os.kill(int(child_pid.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        if process.stdout:
            process.stdout.close()
        if process.stderr:
            process.stderr.close()

    def wait_event(self, process, directory, expected):
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            events = directory / "events"
            if events.exists() and expected in events.read_text().splitlines():
                return
            if process.poll() is not None:
                self.fail(f"Process exited before {expected}: {process.communicate()}")
            time.sleep(.02)
        self.fail("Timed out waiting for " + expected)

    def test_first_ctrl_c_allows_child_stop_and_close(self):
        for capture in (True, False):
            temporary = tempfile.TemporaryDirectory()
            self.addCleanup(temporary.cleanup)
            with self.subTest(capture=capture):
                directory = Path(temporary.name)
                process = self.launch(directory, capture=capture)
                os.killpg(process.pid, signal.SIGINT)
                _output, error = process.communicate(timeout=8)
                self.assertEqual(process.returncode, 130, error)
                self.assertEqual(
                    (directory / "events").read_text().splitlines(),
                    ["ready", "stop-begin", "stop-complete", "close"],
                )

    def test_second_ctrl_c_explicitly_forces_child_to_exit(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        directory = Path(temporary.name)
        process = self.launch(directory, delay=30)
        os.kill(process.pid, signal.SIGINT)
        self.wait_event(process, directory, "stop-begin")
        os.kill(process.pid, signal.SIGINT)
        output, error = process.communicate(timeout=8)
        self.assertEqual(process.returncode, 130, error)
        self.assertIn("buộc dừng", output)
        self.assertNotIn("stop-complete", (directory / "events").read_text())


if __name__ == "__main__":
    unittest.main()


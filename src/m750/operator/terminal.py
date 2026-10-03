"""Terminal UTF-8: chọn tác vụ và xem lệnh; không dùng shell để execute."""

import shlex
import subprocess
import sys


class Back(Exception):
    pass


class Terminal:
    def __init__(self, stdin=None, stdout=None):
        self.stdin = sys.stdin if stdin is None else stdin
        self.stdout = sys.stdout if stdout is None else stdout

    def say(self, text=""):
        print(text, file=self.stdout, flush=True)

    def ask(self, label, default=""):
        suffix = " [%s]" % default if default != "" else ""
        self.stdout.write(label + suffix + ": ")
        self.stdout.flush()
        line = self.stdin.readline()
        if not line:
            raise EOFError()
        value = line.strip()
        if value == ":q":
            raise Back()
        if value == "-":
            return ""
        return value if value else str(default)

    def choose(self, title, choices):
        self.say("\n" + title)
        for key, label in choices.items():
            self.say("  %s. %s" % (key, label))
        self.say("  0. Quay lại / thoát")
        while True:
            selected = self.ask("Chọn", "0")
            if selected == "0":
                return None
            if selected in choices:
                return selected
            self.say("Chọn một số trong danh sách.")

    def confirm(self, text):
        return self.ask(text + " (y/N)").lower() in ("y", "yes", "c", "có")


class CommandRunner:
    def __init__(self, root, terminal, dry_run=False):
        self.root, self.terminal, self.dry_run = root, terminal, dry_run

    def run(self, argv, environment=None, capture=False, confirm=None):
        import os

        argv = [str(value) for value in argv]
        prefix = ["%s=%s" % (k, shlex.quote(str(v))) for k, v in (environment or {}).items()]
        self.terminal.say("Lệnh: " + " ".join(prefix + [shlex.quote(v) for v in argv]))
        if confirm and not self.dry_run and not self.terminal.confirm(confirm):
            raise Back()
        if self.dry_run:
            self.terminal.say("Chỉ xem lệnh; chưa chạy tác vụ.")
            return subprocess.CompletedProcess(argv, 0, "", "")
        result = subprocess.run(
            argv,
            cwd=str(self.root),
            env={**os.environ, **(environment or {})},
            text=True,
            capture_output=capture,
            check=False,
        )
        if capture and result.stderr:
            self.terminal.say(result.stderr.rstrip())
        if result.returncode in (130, -2):
            raise KeyboardInterrupt()
        if result.returncode:
            if capture and result.stdout:
                self.terminal.say(result.stdout.rstrip())
            raise RuntimeError(
                "Tác vụ chưa hoàn tất (mã %s). Xem lỗi phía trên." % result.returncode
            )
        return result

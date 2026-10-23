"""Run ``bash start-miner.sh --practice`` one evaluation at a time.

Practice mode writes ``score_run/`` next to the sample BAM. A per-trial config
path does not isolate that directory, so this runner holds one lock and
refuses ``--workers`` above 1.
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from pathlib import Path

from tuning.errors import TuningError

try:
    import fcntl
except ImportError:  # pragma: no cover - Linux and macOS provide fcntl
    fcntl = None


def practice_command(repo_root: Path, config_path: Path, sample_id: str) -> list[str]:
    script = (repo_root / "start-miner.sh").resolve()
    return [
        "bash",
        str(script),
        "--practice",
        "--config",
        str(Path(config_path).resolve()),
        "--sample-id",
        sample_id,
    ]


# flock is per process on Linux, so a second thread in this process also needs this lock.
_THREAD_LOCK = threading.Lock()


class EvaluationLock:
    """One practice evaluation on this machine at a time."""

    def __init__(self, path: Path):
        self.path = path
        self._handle = None
        self._thread_held = False

    def __enter__(self) -> "EvaluationLock":
        if fcntl is None:
            raise TuningError("practice evaluations need fcntl to keep score_run from colliding")
        _THREAD_LOCK.acquire()
        self._thread_held = True
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open("a+", encoding="utf-8")
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX)
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._handle is not None:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            self._handle.close()
            self._handle = None
        if self._thread_held:
            _THREAD_LOCK.release()
            self._thread_held = False


def assert_workers(workers: int) -> None:
    if workers != 1:
        raise TuningError(
            "workers must be 1. Practice mode writes score_run next to the sample BAM, "
            "so overlapping evaluations can mix VCFs. Isolation is not implemented."
        )


def _drain(pipe, lines: list[str], echo: bool) -> None:
    try:
        for line in iter(pipe.readline, ""):
            lines.append(line)
            if echo and line.strip():
                print(line, end="" if line.endswith("\n") else line + "\n", flush=True)
    finally:
        pipe.close()


def run_practice(
    repo_root: Path,
    config_path: Path,
    sample_id: str,
    *,
    timeout_seconds: int,
    lock_path: Path,
    echo: bool = True,
) -> tuple[int | None, str, str, float, bool]:
    """Return exit code, stdout, stderr, seconds, and whether the run timed out.

    Child output is printed as it arrives. A heartbeat is printed while the
    child is quiet, because HaplotypeCaller itself does not write to this terminal.
    """
    command = practice_command(repo_root, config_path, sample_id)
    with EvaluationLock(lock_path):
        if echo:
            print(f"[tune] practice {sample_id}  config {config_path}", flush=True)
        started = time.monotonic()
        process = subprocess.Popen(
            command,
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        stdout_lines: list[str] = []
        stderr_lines: list[str] = []
        threads = [
            threading.Thread(target=_drain, args=(process.stdout, stdout_lines, echo), daemon=True),
            threading.Thread(target=_drain, args=(process.stderr, stderr_lines, echo), daemon=True),
        ]
        for thread in threads:
            thread.start()
        deadline = started + timeout_seconds
        next_beat = started + 20
        timed_out = False
        while process.poll() is None:
            now = time.monotonic()
            if now >= deadline:
                _kill_group(process)
                timed_out = True
                break
            if echo and now >= next_beat:
                print(f"[tune] HaplotypeCaller still running ({now - started:.0f}s)  sample {sample_id}", flush=True)
                next_beat = now + 20
            time.sleep(0.2)
        process.wait()
        for thread in threads:
            thread.join(timeout=2)
        elapsed = time.monotonic() - started
    return process.returncode, "".join(stdout_lines), "".join(stderr_lines), elapsed, timed_out


def _kill_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        process.kill()
    except PermissionError:
        process.kill()

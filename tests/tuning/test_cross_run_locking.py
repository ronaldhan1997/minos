"""Practice evaluations from different run directories share one lock."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import fcntl
import pytest

from tuning.models import EvaluationResult
from tuning.optimize import run_phase
from tuning.practice_runner import reap_minos_containers, run_practice, shared_practice_lock_path
from minos_layout import minos_root, project_root

REPO = minos_root()
BASELINE = REPO / "configs" / "gatk.conf"
TRAIN_ID = "2ea4a48e-ba7f-43d8-86e7-81ae5a99196f"


def test_one_checkout_has_one_practice_lock(tmp_path: Path) -> None:
    repo = tmp_path / "checkout"
    assert shared_practice_lock_path(repo) == (repo / "datasets" / "practice" / ".tuner-practice.lock").resolve()


def test_optimize_locks_the_checkout_rather_than_the_run_directory(tmp_path: Path, monkeypatch) -> None:
    seen: dict[str, Path] = {}

    def fake_eval(**kwargs):
        seen["lock"] = kwargs["lock_path"]
        return EvaluationResult(
            sample_id=kwargs["sample_id"],
            config_sha256=kwargs["config_sha256"],
            status="success",
            combined_final=0.86,
            elapsed_seconds=0.01,
        )

    monkeypatch.setattr("tuning.optimize.evaluate_sample", fake_eval)
    run_dir = tmp_path / "run-a"
    run_phase(
        repo_root=REPO,
        run_dir=run_dir,
        baseline_path=BASELINE,
        phase=1,
        trials=1,
        train_samples=[TRAIN_ID],
        scoring_version="v2",
        seed=3,
        robustness_penalty=0.25,
        workers=1,
        timeout_seconds=5,
    )
    assert seen["lock"] == shared_practice_lock_path(REPO)
    assert seen["lock"] != (run_dir / "evaluation.lock").resolve()


def test_a_second_process_cannot_take_the_shared_lock(tmp_path: Path) -> None:
    lock_path = shared_practice_lock_path(tmp_path / "checkout")
    ready = tmp_path / "ready"
    release = tmp_path / "release"
    script = tmp_path / "hold_lock.py"
    script.write_text(
        "\n".join(
            [
                "import sys, time",
                "from pathlib import Path",
                "from tuning.practice_runner import EvaluationLock",
                "lock_path, ready, release = sys.argv[1:]",
                "with EvaluationLock(Path(lock_path)):",
                "    Path(ready).write_text('held', encoding='utf-8')",
                "    deadline = time.time() + 5",
                "    while time.time() < deadline and not Path(release).exists():",
                "        time.sleep(0.05)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root()) + os.pathsep + env.get("PYTHONPATH", "")
    process = subprocess.Popen([sys.executable, str(script), str(lock_path), str(ready), str(release)], env=env)
    try:
        deadline = time.time() + 5
        while time.time() < deadline and not ready.exists():
            time.sleep(0.05)
        assert ready.exists()
        handle = lock_path.open("a+", encoding="utf-8")
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            handle.close()
    finally:
        release.write_text("go", encoding="utf-8")
        process.wait(timeout=5)


def test_timeout_reaps_only_containers_started_during_the_run(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "tuning.practice_runner.practice_command",
        lambda *_args, **_kwargs: ["sleep", "30"],
    )
    calls = {"n": 0}

    def lists() -> set[str]:
        calls["n"] += 1
        if calls["n"] == 1:
            return {"minos-gatk-old"}
        return {"minos-gatk-old", "minos-gatk-new"}

    removed: list[list[str]] = []

    def reap(before: set[str], after: set[str]) -> list[str]:
        names = sorted(after - before)
        removed.append(names)
        return names

    monkeypatch.setattr("tuning.practice_runner.reap_minos_containers", reap)
    _code, _stdout, _stderr, _elapsed, timed_out = run_practice(
        tmp_path,
        tmp_path / "gatk.conf",
        "sample",
        timeout_seconds=1,
        lock_path=tmp_path / "evaluation.lock",
        echo=False,
        list_containers=lists,
    )
    assert timed_out is True
    assert removed == [["minos-gatk-new"]]


def test_reap_refuses_names_outside_the_minos_prefix(monkeypatch) -> None:
    commands: list[list[str]] = []

    def fake_run(cmd, **_kwargs):
        commands.append(list(cmd))

        class Result:
            returncode = 0
            stdout = ""

        return Result()

    monkeypatch.setattr("tuning.practice_runner.subprocess.run", fake_run)
    removed = reap_minos_containers({"minos-gatk-old"}, {"minos-gatk-old", "minos-happy-new", "other"})
    assert removed == ["minos-happy-new"]
    assert commands == [["docker", "rm", "-f", "minos-happy-new"]]

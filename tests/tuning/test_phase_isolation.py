"""Phase 1 and phase 2 trial directories stay independent. No GATK process is started."""

from __future__ import annotations

import json
from pathlib import Path

import optuna
import pytest

from tuning.baseline import config_sha256, load_baseline
from tuning.cli import main
from tuning.errors import TuningError
from tuning.legacy import migrate_legacy_run
from tuning.optimize import run_phase
from tuning.reports import trial_records
from tuning.storage import append_jsonl, atomic_json
from minos_layout import minos_root

REPO = minos_root()
BASELINE = REPO / "configs" / "gatk.conf"
TRAIN_ID = "2ea4a48e-ba7f-43d8-86e7-81ae5a99196f"


def _recordable(score: str = "0.860000") -> str:
    return (
        f"COMBINED_FINAL:   {score}   <-- what a validator records\n"
        "ADVANCED SCORE:   86.0000 / 100  (scoring v2)\n"
    )


def _run(phase: int, trials: int, run_dir: Path, runner) -> dict:
    return run_phase(
        repo_root=REPO,
        run_dir=run_dir,
        baseline_path=BASELINE,
        phase=phase,
        trials=trials,
        train_samples=[TRAIN_ID],
        scoring_version="v2",
        seed=17,
        robustness_penalty=0.25,
        workers=1,
        timeout_seconds=5,
        runner=runner,
    )


def test_same_trial_number_uses_separate_phase_directories(tmp_path: Path) -> None:
    def runner(**kwargs):
        return 0, _recordable("0.870000"), str(kwargs["config_path"]), 0.01, False

    run_dir = tmp_path / "run"
    _run(1, 1, run_dir, runner)
    _run(2, 1, run_dir, runner)
    phase1 = run_dir / "phase1" / "trials" / "000000"
    phase2 = run_dir / "phase2" / "trials" / "000000"
    assert (phase1 / "trial.json").is_file()
    assert (phase2 / "trial.json").is_file()
    assert (phase1 / "gatk.conf").read_bytes() != b""
    assert (phase2 / TRAIN_ID / "attempt1.stderr.log").read_text(encoding="utf-8") != (
        phase1 / TRAIN_ID / "attempt1.stderr.log"
    ).read_text(encoding="utf-8")
    assert len(trial_records(run_dir, 1)) == 1
    assert len(trial_records(run_dir, 2)) == 1
    assert not (run_dir / "trials").exists()
    again = _run(1, 1, run_dir, runner)
    assert again["finished"] == 1
    assert len(list((run_dir / "phase1" / "trials").glob("*/trial.json"))) == 1


def test_legacy_migration_copies_an_unambiguous_phase_and_keeps_the_original(tmp_path: Path) -> None:
    run_dir = tmp_path / "legacy"
    run_dir.mkdir()
    storage = "sqlite:///" + str((run_dir / "optuna.sqlite3").resolve())
    study = optuna.create_study(study_name="phase1", storage=storage, direction="maximize")
    trial = study.ask()
    study.tell(trial, 0.5)
    record = {
        "phase": 1,
        "trial": trial.number,
        "status": "success",
        "objective": 0.5,
        "config_sha256": "abc",
        "parameters": {"min_pruning": 2},
    }
    atomic_json(run_dir / "trials" / f"{trial.number:06d}" / "trial.json", record)
    append_jsonl(run_dir / "trial_results.jsonl", record)
    original = (run_dir / "trials" / f"{trial.number:06d}" / "trial.json").read_bytes()
    payload = migrate_legacy_run(run_dir)
    assert payload["status"] == "migrated"
    assert payload["copied"] == 1
    copied = run_dir / "phase1" / "trials" / f"{trial.number:06d}" / "trial.json"
    assert copied.read_bytes() == original
    assert (run_dir / "trials" / f"{trial.number:06d}" / "trial.json").read_bytes() == original
    assert (run_dir / "legacy_archive").is_dir()
    assert len(trial_records(run_dir, 1)) == 1
    assert trial_records(run_dir, 2) == []
    second = migrate_legacy_run(run_dir)
    assert second["status"] == "migrated"
    assert len(list((run_dir / "legacy_archive").iterdir())) == 1


def test_overlapping_legacy_phases_are_untrusted_and_not_merged(tmp_path: Path) -> None:
    run_dir = tmp_path / "overlap"
    run_dir.mkdir()
    storage = "sqlite:///" + str((run_dir / "optuna.sqlite3").resolve())
    phase1 = optuna.create_study(study_name="phase1", storage=storage, direction="maximize")
    phase2 = optuna.create_study(study_name="phase2", storage=storage, direction="maximize")
    left = phase1.ask()
    phase1.tell(left, 0.4)
    right = phase2.ask()
    phase2.tell(right, 0.6)
    assert left.number == right.number
    record = {"phase": 1, "trial": left.number, "status": "success", "objective": 0.4, "config_sha256": "abc"}
    atomic_json(run_dir / "trials" / f"{left.number:06d}" / "trial.json", record)
    append_jsonl(run_dir / "trial_results.jsonl", record)
    append_jsonl(
        run_dir / "trial_results.jsonl",
        {"phase": 2, "trial": right.number, "status": "success", "objective": 0.6, "config_sha256": "def"},
    )
    payload = migrate_legacy_run(run_dir)
    assert payload["status"] == "untrusted"
    assert payload["copied"] == 0
    assert not (run_dir / "phase1").exists()
    assert not (run_dir / "phase2").exists()
    assert (run_dir / "trials" / f"{left.number:06d}" / "trial.json").is_file()
    assert (run_dir / "legacy_archive").is_dir()
    with pytest.raises(TuningError, match="cannot be attributed"):
        _run(1, 1, run_dir, lambda **kwargs: (0, _recordable(), "", 0.01, False))


def test_pipeline_prints_the_campaign_without_starting_trials(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    train = tmp_path / "train.txt"
    train.write_text(TRAIN_ID + "\n", encoding="utf-8")
    run_dir = tmp_path / "campaign"
    code = main(
        [
            "--repo-root",
            str(REPO),
            "pipeline",
            "--run-dir",
            str(run_dir),
            "--train-samples",
            str(train),
            "--baseline",
            str(BASELINE),
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["executes_gatk"] is False
    assert any("optimize" in command and "--phase" in command and "1" in command for command in payload["commands"])
    assert not (run_dir / "optuna.sqlite3").exists()
    assert not (run_dir / "phase1").exists()

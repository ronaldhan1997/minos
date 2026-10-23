"""Practice-mode tuner. These tests never launch GATK."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from tuning.baseline import config_sha256, file_sha256, load_baseline, type_counts
from tuning.errors import ScoringVersionAbort, TuningError
from tuning.objective import aggregate, trial_objective
from tuning.parameters import fixed_names, tunable_names, validate_candidate
from tuning.practice_runner import EvaluationLock, assert_workers, practice_command
from tuning.score_parser import parse_practice_output
from tuning.cli import main
from tuning.models import EvaluationResult
from tuning.optimize import run_phase
from minos_layout import minos_root
from tuning.validate import export_promoted, leakage_errors, promote, run_validation

REPO = minos_root()
BASELINE = REPO / "configs" / "gatk.conf"
TRAIN_ID = "2ea4a48e-ba7f-43d8-86e7-81ae5a99196f"
HOLDOUT = [
    "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaa1",
    "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaa2",
    "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaa3",
]


def _recordable(score: str = "0.860000", version: str = "v2") -> str:
    return (
        f"COMBINED_FINAL:   {score}   <-- what a validator records\n"
        f"ADVANCED SCORE:   86.0000 / 100  (scoring {version})\n"
    )


def _runner(score: str = "0.860000", version: str = "v2"):
    def run(**kwargs):
        return 0, _recordable(score, version), "", 0.01, False

    return run


def test_real_miner_config_is_not_rewritten(tmp_path: Path) -> None:
    before = BASELINE.read_bytes()
    digest = file_sha256(BASELINE)
    code = main(
        [
            "--repo-root",
            str(REPO),
            "preflight",
            "--baseline",
            str(BASELINE),
        ]
    )
    assert code == 0
    assert BASELINE.read_bytes() == before
    assert file_sha256(BASELINE) == digest
    values = load_baseline(BASELINE)
    assert type_counts(values) == {"int": 14, "float": 7, "bool": 2, "enum": 2}
    assert values["emit_ref_confidence"] == "NONE"
    assert values["sample_ploidy"] == 2
    assert values["contamination_fraction_to_filter"] == 0.0
    assert tunable_names(1) == [
        "min_base_quality_score",
        "min_mapping_quality_score",
        "base_quality_score_threshold",
        "standard_min_confidence_threshold_for_calling",
        "pcr_indel_model",
        "min_pruning",
        "active_probability_threshold",
        "max_reads_per_alignment_start",
    ]
    assert len(fixed_names(1)) == 17
    assert tunable_names(2) == tunable_names(1)
    for name in ("emit_ref_confidence", "sample_ploidy", "contamination_fraction_to_filter"):
        assert name in fixed_names(1)


def test_parser_accepts_plain_and_rich_and_rejects_discards() -> None:
    plain = parse_practice_output(_recordable(), expected_version="v2")
    assert plain["status"] == "success"
    assert plain["combined_final"] == pytest.approx(0.86)
    rich = (
        "\x1b[32m[bold]COMBINED_FINAL  0.850000[/bold]\x1b[0m\n"
        "This is exactly what a validator would record (scoring v2).\n"
    )
    parsed = parse_practice_output(rich, expected_version="v2")
    assert parsed["status"] == "success"
    assert parsed["combined_final"] == pytest.approx(0.85)
    zero = parse_practice_output(
        "COMBINED_FINAL:   0.250000   <-- ZERO-INPUT: a validator discards this (scoring v2)\n",
        expected_version="v2",
    )
    assert zero["status"] == "discarded"
    assert zero["combined_final"] == pytest.approx(0.25)
    out_of_range = parse_practice_output(
        "COMBINED_FINAL:   1.200000 out of range; a validator discards this. (scoring v2)\n",
        expected_version="v2",
    )
    assert out_of_range["status"] == "discarded"
    mismatch = parse_practice_output(_recordable(version="v1"), expected_version="v2")
    assert mismatch["status"] == "scoring_version_mismatch"
    missing = parse_practice_output(
        "COMBINED_FINAL:   0.860000   <-- what a validator records\n",
        expected_version="v2",
    )
    assert missing["status"] == "score_error"


def test_objective_uses_population_std_and_rejects_a_failed_sample() -> None:
    assert aggregate([0.8, 0.6], 0.25) == pytest.approx(0.675)
    assert aggregate([0.86], 0.25) == pytest.approx(0.86)
    ok = EvaluationResult(sample_id="a", config_sha256="b", status="success", combined_final=0.5)
    bad = EvaluationResult(sample_id="c", config_sha256="b", status="discarded", combined_final=None)
    assert trial_objective([ok, bad], 0.25) is None


def test_invalid_config_and_cross_field_constraint() -> None:
    values = load_baseline(BASELINE)
    broken = dict(values)
    broken["sample_ploidy"] = 3
    assert any("sample_ploidy" in item for item in validate_candidate(broken, 1))
    broken = dict(values)
    broken["min_assembly_region_size"] = 400
    broken["max_assembly_region_size"] = 200
    assert any("min_assembly_region_size" in item for item in validate_candidate(broken, 2))
    assert validate_candidate(values, 1) == []
    assert validate_candidate(values, 2) == []


def test_practice_command_is_a_list_and_workers_stay_one(tmp_path: Path) -> None:
    config = tmp_path / "gatk.conf"
    config.write_text("x=1\n", encoding="utf-8")
    command = practice_command(REPO, config, TRAIN_ID)
    assert command[0] == "bash"
    assert "--practice" in command
    assert command[command.index("--config") + 1] == str(config.resolve())
    assert command[command.index("--sample-id") + 1] == TRAIN_ID
    with pytest.raises(TuningError):
        assert_workers(2)


def test_lock_is_exclusive(tmp_path: Path) -> None:
    order: list[str] = []
    lock = EvaluationLock(tmp_path / "evaluation.lock")

    def hold() -> None:
        with lock:
            order.append("in")
            time.sleep(0.2)
            order.append("out")

    thread = threading.Thread(target=hold)
    thread.start()
    time.sleep(0.05)
    with lock:
        order.append("second")
    thread.join()
    assert order == ["in", "out", "second"]


def _ids(path: Path, rows: list[str]) -> Path:
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_fake_phase_resumes_and_does_not_touch_the_miner_config(tmp_path: Path) -> None:
    before = BASELINE.read_bytes()
    calls = {"n": 0}

    def runner(**kwargs):
        calls["n"] += 1
        return 0, _recordable("0.870000"), "", 0.01, False

    run_dir = tmp_path / "run"
    train = _ids(tmp_path / "train.txt", [TRAIN_ID])
    summary = run_phase(
        repo_root=REPO,
        run_dir=run_dir,
        baseline_path=BASELINE,
        phase=1,
        trials=1,
        train_samples=[TRAIN_ID],
        scoring_version="v2",
        seed=17,
        robustness_penalty=0.25,
        workers=1,
        timeout_seconds=5,
        runner=runner,
    )
    assert summary["finished"] == 1
    assert summary["single_sample_exploratory"] is True
    assert calls["n"] == 1
    again = run_phase(
        repo_root=REPO,
        run_dir=run_dir,
        baseline_path=BASELINE,
        phase=1,
        trials=1,
        train_samples=[TRAIN_ID],
        scoring_version="v2",
        seed=17,
        robustness_penalty=0.25,
        workers=1,
        timeout_seconds=5,
        runner=runner,
    )
    assert again["finished"] == 1
    assert calls["n"] == 1
    trial = json.loads(next((run_dir / "trials").glob("*/trial.json")).read_text(encoding="utf-8"))
    assert len(trial["parameters"]) == 25
    assert trial["suggested_count"] == 8
    assert trial["parameters"]["emit_ref_confidence"] == "NONE"
    assert config_sha256(trial["parameters"]) == trial["config_sha256"]
    assert BASELINE.read_bytes() == before
    with pytest.raises(ScoringVersionAbort):
        run_phase(
            repo_root=REPO,
            run_dir=tmp_path / "mismatch",
            baseline_path=BASELINE,
            phase=1,
            trials=1,
            train_samples=[TRAIN_ID],
            scoring_version="v2",
            seed=17,
            robustness_penalty=0.25,
            workers=1,
            timeout_seconds=5,
            runner=_runner(version="v1"),
        )


def test_promotion_gate_and_export_refusal(tmp_path: Path) -> None:
    base = {"a": 0.80, "b": 0.80, "c": 0.80}
    good = {"a": 0.83, "b": 0.81, "c": 0.82}
    worse = {"a": 0.70, "b": 0.81, "c": 0.82}
    assert promote(base, good)[0] is True
    assert promote(base, worse)[0] is False
    assert leakage_errors(["a"], ["a"], {}) 
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "train_samples": [TRAIN_ID],
                "scoring_version": "v2",
                "robustness_penalty": 0.25,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "finalists_manifest.json").write_text(
        json.dumps(
            {
                "finalists": [
                    {
                        "trial": 0,
                        "objective": 0.9,
                        "config_sha256": "abc",
                        "parameters": load_baseline(BASELINE),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(TuningError, match="at least 3"):
        run_validation(
            repo_root=REPO,
            run_dir=run_dir,
            baseline_path=BASELINE,
            validation_samples=HOLDOUT[:2],
            timeout_seconds=5,
            runner=_runner(),
        )
    scores = {"baseline": 0.80, "candidate": 0.84}

    def runner(**kwargs):
        sample = kwargs["sample_id"]
        config = Path(kwargs["config_path"]).read_text(encoding="utf-8")
        # The baseline and the finalist are the same bytes in this fixture, so
        # both sides receive a recordable score. The gate then sees zero lift.
        assert sample in HOLDOUT
        assert "emit_ref_confidence=NONE" in config
        return 0, _recordable(f"{scores['baseline']:.6f}"), "", 0.01, False

    report = run_validation(
        repo_root=REPO,
        run_dir=run_dir,
        baseline_path=BASELINE,
        validation_samples=HOLDOUT,
        timeout_seconds=5,
        runner=runner,
    )
    assert report["decision"] == "NOT_PROMOTED"
    with pytest.raises(TuningError, match="NOT_PROMOTED"):
        export_promoted(run_dir, tmp_path / "candidate.conf", exploratory=False)
    exported = export_promoted(run_dir, tmp_path / "candidate.exploratory.conf", exploratory=True)
    assert exported.exists()
    with pytest.raises(TuningError, match="Refusing"):
        export_promoted(run_dir, tmp_path / "configs" / "gatk.conf", exploratory=True)


def test_tune_dry_run_needs_no_extra_flags(tmp_path: Path) -> None:
    before = BASELINE.read_bytes()
    code = main(["--repo-root", str(REPO), "tune", "--dry-run", "--run-dir", str(tmp_path / "my_run")])
    assert code == 0
    assert BASELINE.read_bytes() == before
    assert (tmp_path / "my_run" / "train_samples.txt").read_text(encoding="utf-8").strip() == TRAIN_ID
    assert (tmp_path / "my_run" / "dry_run" / "gatk.conf").is_file()
    assert not (tmp_path / "my_run" / "optuna.sqlite3").exists()


def test_cli_dry_run_prints_the_practice_command(tmp_path: Path) -> None:
    before = BASELINE.read_bytes()
    train = _ids(tmp_path / "train.txt", [TRAIN_ID])
    code = main(
        [
            "--repo-root",
            str(REPO),
            "optimize",
            "--phase",
            "1",
            "--trials",
            "100",
            "--train-samples",
            str(train),
            "--run-dir",
            str(tmp_path / "dry"),
            "--dry-run",
        ]
    )
    assert code == 0
    assert BASELINE.read_bytes() == before
    assert (tmp_path / "dry" / "dry_run" / "gatk.conf").is_file()
    assert not (tmp_path / "dry" / "optuna.sqlite3").exists()

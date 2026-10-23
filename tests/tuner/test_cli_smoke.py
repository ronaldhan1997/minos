"""Dry-run CLI smoke test. GATK and hap.py are not started."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from typer.testing import CliRunner

from tuner.cli import app
from tuner.reporting import write_report
from tuner.settings import StudyPaths
from tuner.storage import Evaluation, Store

ROOT = Path(__file__).resolve().parents[2]
runner = CliRunner()


def test_help_lists_the_contract():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for name in ("doctor", "prepare", "test", "tune", "validate", "report", "export"):
        assert name in result.stdout


def test_prepare_and_dry_run_tune_resume(tuner_home):
    common = [
        "--experiments",
        str(tuner_home.experiments_dir),
        "--data-dir",
        str(tuner_home.datasets_dir),
    ]
    prepared = runner.invoke(app, common + ["prepare", "--samples", "chr20"])
    assert prepared.exit_code == 0, prepared.stdout + prepared.stderr
    missed = runner.invoke(app, common + ["prepare", "--samples", "chr18"])
    assert missed.exit_code == 1
    assert "chromosome filter" in missed.stdout
    garbage = runner.invoke(app, common + ["prepare", "--samples", "not-an-id"])
    assert garbage.exit_code == 2

    first = runner.invoke(
        app,
        common + ["tune", "--trials", "3", "--workers", "1", "--study", "smoke", "--dry-run", "--samples", "chr20", "--allow-partial-catalog"],
    )
    assert first.exit_code == 0, first.stdout + first.stderr
    meta = json.loads((tuner_home.experiments_dir / "smoke" / "study.json").read_text())
    assert meta["val_sample_ids"]
    assert set(meta["train_sample_ids"]).isdisjoint(meta["val_sample_ids"])
    db = tuner_home.experiments_dir / "smoke" / "dry-run" / "evaluations.sqlite3"
    rows = _rows(db)
    assert rows
    assert all(row["score"] is None for row in rows)
    assert all(row["status"] == "dry_run" for row in rows)
    assert set(meta["val_sample_ids"]).isdisjoint({row["sample_id"] for row in rows})
    count = len(rows)
    second = runner.invoke(
        app,
        common + ["tune", "--trials", "3", "--workers", "1", "--study", "smoke", "--dry-run", "--allow-partial-catalog"],
    )
    assert second.exit_code == 0, second.stdout + second.stderr
    assert len(_rows(db)) == count
    third = runner.invoke(
        app,
        common + ["tune", "--trials", "4", "--workers", "1", "--study", "smoke", "--dry-run", "--allow-partial-catalog"],
    )
    assert third.exit_code == 0, third.stdout + third.stderr
    assert len(_rows(db)) == count + len(meta["train_sample_ids"])


def test_one_explicit_sample_skips_the_full_catalog(tuner_home):
    result = runner.invoke(
        app,
        [
            "--experiments",
            str(tuner_home.experiments_dir),
            "--data-dir",
            str(tuner_home.datasets_dir),
            "tune",
            "--trials",
            "1",
            "--workers",
            "1",
            "--study",
            "one-sample",
            "--dry-run",
            "--samples",
            "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaa1",
        ],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "selected 1 sample id for this study" in result.stdout
    assert "catalog 75" not in result.stdout


def test_dry_run_test_command_has_no_score(tuner_home):
    result = runner.invoke(
        app,
        [
            "--experiments",
            str(tuner_home.experiments_dir),
            "--data-dir",
            str(tuner_home.datasets_dir),
            "test",
            "--config",
            str(ROOT / "configs" / "baseline.conf"),
            "--sample",
            "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaa1",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "score=None" in result.stdout
    assert "dry_run" in result.stdout


def test_report_writes_csv_and_json(tuner_home):
    paths = StudyPaths.for_study(tuner_home, "reported")
    store = Store(paths.evaluations_db)
    store.record(
        Evaluation(
            cache_key="report-1",
            study="reported",
            trial_number=1,
            sample_id="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaa1",
            role="train",
            parameters={"min_pruning": 2},
            status="failed",
            score=None,
            config_hash="abc",
            minos_commit="abc",
            gatk_image="broadinstitute/gatk:4.5.0.0",
            scorer_name="Advanced",
            scoring_version="v1",
            failure_reason="Timeout after 10s",
            runtime_seconds=10.0,
        )
    )
    store.close()
    output = tuner_home.experiments_dir / "report.csv"
    summary = write_report(tuner_home, "reported", output)
    assert output.exists()
    assert Path(summary["json"]).exists()
    text = Path(summary["json"]).read_text(encoding="utf-8")
    assert "Timeout after 10s" in text
    assert summary["parameter_importance"] is None


def test_tuner_does_not_call_live_submission():
    for path in (ROOT / "tuner").glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "submit_config" not in text
        assert "submit_result" not in text
        assert "subtensor" not in text


def _rows(path: Path):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        return list(conn.execute("SELECT sample_id, status, score FROM evaluations"))
    finally:
        conn.close()

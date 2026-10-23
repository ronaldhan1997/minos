"""Phase 2 tunes 15 parameters and keeps phase-1 values on its seeds. No GATK process is started."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tuning.baseline import config_sha256, load_baseline
from tuning.cli import main
from tuning.optimize import run_phase
from tuning.parameters import PHASE2_EXTRA, fixed_names, tunable_names, validate_candidate
from tuning.reports import trial_records
from tuning.storage import atomic_json
from minos_layout import minos_root

REPO = minos_root()
BASELINE = REPO / "configs" / "gatk.conf"
TRAIN_ID = "2ea4a48e-ba7f-43d8-86e7-81ae5a99196f"
PHASE2_FIELDS = (
    "min_dangling_branch_length",
    "max_num_haplotypes_in_population",
    "assembly_region_padding",
    "pair_hmm_gap_continuation_penalty",
    "phred_scaled_global_read_mismapping_rate",
    "heterozygosity",
    "indel_heterozygosity",
)


def _recordable(score: str) -> str:
    return (
        f"COMBINED_FINAL:   {score}   <-- what a validator records\n"
        "ADVANCED SCORE:   80.0000 / 100  (scoring v2)\n"
    )


def _run(phase: int, trials: int, run_dir: Path) -> None:
    def runner(**kwargs):
        return 0, _recordable("0.800000"), "", 0.01, False

    run_phase(
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


def test_phase2_space_contains_phase1_and_seven_additions() -> None:
    assert len(tunable_names(1)) == 8
    assert tunable_names(2)[:8] == tunable_names(1)
    assert list(PHASE2_EXTRA) == list(PHASE2_FIELDS)
    assert tunable_names(2)[8:] == list(PHASE2_FIELDS)
    assert len(fixed_names(2)) == 10
    baseline = load_baseline(BASELINE)
    assert validate_candidate(baseline, 2) == []
    outside = dict(baseline)
    outside["min_dangling_branch_length"] = 1
    assert any("min_dangling_branch_length" in item for item in validate_candidate(outside, 2))


def test_phase2_dry_run_emits_a_complete_config(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    train = tmp_path / "train.txt"
    train.write_text(TRAIN_ID + "\n", encoding="utf-8")
    code = main(
        [
            "--repo-root",
            str(REPO),
            "optimize",
            "--phase",
            "2",
            "--trials",
            "2",
            "--train-samples",
            str(train),
            "--run-dir",
            str(tmp_path / "dry"),
            "--dry-run",
            "--baseline",
            str(BASELINE),
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["suggested"] == 15
    assert payload["fixed"] == 10
    text = (tmp_path / "dry" / "dry_run" / "gatk.conf").read_text(encoding="utf-8")
    assert len([line for line in text.splitlines() if "=" in line]) == 25
    assert not (tmp_path / "dry" / "optuna.sqlite3").exists()


def test_phase2_seeds_keep_scored_phase1_values_and_baseline_extras(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _run(1, 1, run_dir)
    baseline = load_baseline(BASELINE)
    kept = dict(baseline)
    kept["min_pruning"] = 7
    kept["min_dangling_branch_length"] = 9
    failed = dict(baseline)
    failed["min_pruning"] = 10
    incomplete = dict(baseline)
    incomplete.pop("min_pruning")
    atomic_json(
        run_dir / "phase1" / "trials" / "000003" / "trial.json",
        {
            "phase": 1,
            "trial": 3,
            "status": "success",
            "objective": 0.95,
            "config_sha256": config_sha256(kept),
            "parameters": kept,
        },
    )
    atomic_json(
        run_dir / "phase1" / "trials" / "000004" / "trial.json",
        {
            "phase": 1,
            "trial": 4,
            "status": "failed",
            "objective": 0.99,
            "config_sha256": config_sha256(failed),
            "parameters": failed,
        },
    )
    atomic_json(
        run_dir / "phase1" / "trials" / "000005" / "trial.json",
        {
            "phase": 1,
            "trial": 5,
            "status": "success",
            "objective": 0.98,
            "config_sha256": "incomplete",
            "parameters": incomplete,
        },
    )
    _run(2, 1, run_dir)
    seeded = trial_records(run_dir, 2)[0]["parameters"]
    assert seeded["min_pruning"] == 7
    assert seeded["min_dangling_branch_length"] == baseline["min_dangling_branch_length"]
    for name in PHASE2_FIELDS:
        assert seeded[name] == baseline[name]
    for name in tunable_names(1):
        assert seeded[name] == kept[name]
    assert len(seeded) == 25
    assert seeded["emit_ref_confidence"] == "NONE"

"""One real practice evaluation. Skipped unless MINOS_RUN_PRACTICE=1.

This launches GATK through ``start-miner.sh --practice``. The default suite
does not run it.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tuning.baseline import load_baseline
from tuning.practice_runner import run_practice, shared_practice_lock_path
from tuning.score_parser import parse_practice_output
from tuning.storage import write_trial_config
from minos_layout import minos_root

pytestmark = pytest.mark.skipif(
    os.environ.get("MINOS_RUN_PRACTICE") != "1",
    reason="set MINOS_RUN_PRACTICE=1 to score the baseline once through practice mode",
)

REPO = minos_root()
BASELINE = REPO / "configs" / "gatk.conf"
TRAIN_ID = "2ea4a48e-ba7f-43d8-86e7-81ae5a99196f"


def test_baseline_practice_prints_a_recordable_score(tmp_path: Path) -> None:
    before = BASELINE.read_bytes()
    config = write_trial_config(tmp_path / "baseline", load_baseline(BASELINE))
    code, stdout, stderr, elapsed, timed_out = run_practice(
        REPO,
        config,
        TRAIN_ID,
        timeout_seconds=3600,
        lock_path=shared_practice_lock_path(REPO),
        echo=True,
    )
    parsed = parse_practice_output(f"{stdout}\n{stderr}", expected_version=None)
    assert timed_out is False
    assert code == 0
    assert parsed["status"] == "success"
    assert parsed["combined_final"] is not None
    assert 0.0 < parsed["combined_final"] <= 1.0
    assert parsed["scoring_version"] in {"v1", "v2"}
    assert elapsed > 0
    assert BASELINE.read_bytes() == before

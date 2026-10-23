"""Opt-in checks against the local Docker images and staged practice data.

Run:
    MINOS_TUNER_INTEGRATION=1 ../.venv/bin/python -m pytest tests/tuner/test_integration_optional.py -m integration

A real HaplotypeCaller plus hap.py score is a second switch, because one
practice region can take many minutes:

    MINOS_TUNER_INTEGRATION=1 MINOS_TUNER_RUN_GATK=1 ../.venv/bin/python -m pytest tests/tuner/test_integration_optional.py -m integration
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.integration

if os.environ.get("MINOS_TUNER_INTEGRATION") != "1":
    pytest.skip("set MINOS_TUNER_INTEGRATION=1 to run tuner integration checks", allow_module_level=True)

from typer.testing import CliRunner

from tuner.cli import app
from tuner.datasets import discover_samples
from tuner.settings import load_settings

runner = CliRunner()


def test_doctor_on_this_machine():
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0, result.stdout
    assert "scoring_version" in result.stdout
    assert "ready: True" in result.stdout


def test_prepare_indexes_staged_practice_samples():
    settings = load_settings()
    samples = discover_samples(settings, hash_files=False)
    assert samples, "no datasets/practice samples are staged"
    result = runner.invoke(app, ["prepare"])
    assert result.exit_code in {0, 1}
    assert "ready" in result.stdout or "blocked" in result.stdout


@pytest.mark.skipif(os.environ.get("MINOS_TUNER_RUN_GATK") != "1", reason="set MINOS_TUNER_RUN_GATK=1 to run GATK")
def test_one_official_baseline_score():
    """Score the shipped baseline on the demo practice sample.

    The sample id is the public demo id in neurons/miner.py. The numeric score
    must come from AdvancedScorer; this test only checks that a finite official
    score was stored.
    """
    demo = "d7b99f4a-061c-4202-8b12-2c1425bd4974"
    result = runner.invoke(
        app,
        [
            "test",
            "--config",
            "configs/baseline.conf",
            "--sample",
            demo,
            "--timeout",
            "3600",
            "--threads",
            "1",
            "--memory-gb",
            "8",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "score=" in result.stdout
    assert "score=None" not in result.stdout

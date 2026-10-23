"""Train-set objective on validator-recordable COMBINED_FINAL scores.

The frozen formula is ``mean - robustness_penalty * population_std``.
A trial counts only when every training sample is recordable. Missing
samples are not filled with zero.
"""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any, Callable

from tuning.models import EvaluationResult
from tuning.score_parser import parse_practice_output

Runner = Callable[..., tuple[int | None, str, str, float, bool]]


def aggregate(scores: list[float], robustness_penalty: float) -> float | None:
    if not scores:
        return None
    mean = statistics.fmean(scores)
    std = statistics.pstdev(scores) if len(scores) >= 2 else 0.0
    return mean - robustness_penalty * std


def classify_process_output(
    *,
    sample_id: str,
    config_sha256: str,
    expected_version: str,
    exit_code: int | None,
    stdout: str,
    stderr: str,
    elapsed: float,
    timed_out: bool,
    log_dir: str,
    attempt: int,
) -> EvaluationResult:
    if timed_out:
        return EvaluationResult(
            sample_id=sample_id,
            config_sha256=config_sha256,
            status="timeout",
            elapsed_seconds=elapsed,
            exit_code=exit_code,
            log_dir=log_dir,
            error=f"practice timed out after {elapsed:.0f}s",
            attempt=attempt,
        )
    if exit_code not in (0, None):
        return EvaluationResult(
            sample_id=sample_id,
            config_sha256=config_sha256,
            status="execution_error",
            elapsed_seconds=elapsed,
            exit_code=exit_code,
            log_dir=log_dir,
            error=f"practice exited {exit_code}",
            attempt=attempt,
        )
    parsed = parse_practice_output(f"{stdout}\n{stderr}", expected_version=expected_version)
    return EvaluationResult(
        sample_id=sample_id,
        config_sha256=config_sha256,
        scoring_version=parsed["scoring_version"],
        status=parsed["status"],
        combined_final=parsed["combined_final"] if parsed["status"] == "success" else None,
        advanced_score=parsed["advanced_score"],
        elapsed_seconds=elapsed,
        exit_code=exit_code,
        log_dir=log_dir,
        metrics=parsed["metrics"],
        error=parsed["error"],
        attempt=attempt,
    )


def evaluate_sample(
    *,
    repo_root: Path,
    config_path: Path,
    sample_id: str,
    config_sha256: str,
    expected_version: str,
    timeout_seconds: int,
    lock_path: Path,
    log_dir: Path,
    runner: Runner | None = None,
) -> EvaluationResult:
    """Run practice, retry a timeout once, and write the logs."""
    log_dir.mkdir(parents=True, exist_ok=True)
    last: EvaluationResult | None = None
    for attempt in (1, 2):
        if runner is None:
            from tuning.practice_runner import run_practice

            exit_code, stdout, stderr, elapsed, timed_out = run_practice(
                repo_root,
                config_path,
                sample_id,
                timeout_seconds=timeout_seconds,
                lock_path=lock_path,
            )
        else:
            exit_code, stdout, stderr, elapsed, timed_out = runner(
                repo_root=repo_root,
                config_path=config_path,
                sample_id=sample_id,
                timeout_seconds=timeout_seconds,
            )
        (log_dir / f"attempt{attempt}.stdout.log").write_text(stdout, encoding="utf-8")
        (log_dir / f"attempt{attempt}.stderr.log").write_text(stderr, encoding="utf-8")
        last = classify_process_output(
            sample_id=sample_id,
            config_sha256=config_sha256,
            expected_version=expected_version,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            elapsed=elapsed,
            timed_out=timed_out,
            log_dir=str(log_dir),
            attempt=attempt,
        )
        if last.status != "timeout":
            return last
    assert last is not None
    return last


def trial_objective(results: list[EvaluationResult], robustness_penalty: float) -> float | None:
    if not results or any(item.status != "success" or item.combined_final is None for item in results):
        return None
    return aggregate([float(item.combined_final) for item in results], robustness_penalty)

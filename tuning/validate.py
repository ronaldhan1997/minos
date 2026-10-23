"""Held-out promotion gate. Validation scores never re-enter Optuna."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable

from tuning.baseline import load_baseline, write_conf
from tuning.errors import TuningError
from tuning.objective import evaluate_sample
from tuning.parameters import validate_candidate
from tuning.storage import atomic_json, read_json, utc_now
from utils.path_utils import safe_round_dir_name

MAX_REGRESSION = 0.02


def _file_sha(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sample_fingerprints(repo_root: Path, sample_ids: list[str]) -> dict[str, dict[str, str | None]]:
    fingerprints = {}
    for sample_id in sample_ids:
        directory = repo_root / "datasets" / "practice" / safe_round_dir_name(sample_id)
        fingerprints[sample_id] = {
            "bam": _file_sha(directory / "input.bam"),
            "truth": _file_sha(directory / "truth.vcf.gz"),
        }
    return fingerprints


def leakage_errors(train_ids: list[str], validation_ids: list[str], fingerprints: dict[str, dict[str, str | None]]) -> list[str]:
    errors = []
    overlap = sorted(set(train_ids) & set(validation_ids))
    if overlap:
        errors.append("training and validation share sample ids: " + ", ".join(overlap))
    for train_id in train_ids:
        for val_id in validation_ids:
            if train_id == val_id:
                continue
            for kind in ("bam", "truth"):
                left = fingerprints.get(train_id, {}).get(kind)
                right = fingerprints.get(val_id, {}).get(kind)
                if left and right and left == right:
                    errors.append(f"{val_id} {kind} matches training sample {train_id}")
    return errors


def promote(baseline_scores: dict[str, float], candidate_scores: dict[str, float]) -> tuple[bool, list[str]]:
    reasons = []
    if set(baseline_scores) != set(candidate_scores):
        reasons.append("baseline and candidate were not scored on the same validation samples")
        return False, reasons
    lifts = []
    for sample_id, base in baseline_scores.items():
        score = candidate_scores[sample_id]
        delta = score - base
        lifts.append(delta)
        if delta < -MAX_REGRESSION:
            reasons.append(f"{sample_id} regressed by {-delta:.4f}, above the 0.02 limit")
    if not lifts or sum(lifts) / len(lifts) <= 0:
        reasons.append("mean validation lift is not positive")
    return not reasons, reasons


def run_validation(
    *,
    repo_root: Path,
    run_dir: Path,
    baseline_path: Path,
    validation_samples: list[str],
    timeout_seconds: int,
    runner: Callable | None = None,
    exploratory: bool = False,
    finalist_limit: int | None = None,
) -> dict[str, Any]:
    if not exploratory and len(validation_samples) < 3:
        raise TuningError(
            "Phase 3 needs at least 3 validation sample ids that were not used in training. "
            "A one-sample result stays exploratory and is not certified."
        )
    manifest = read_json(run_dir / "manifest.json")
    finalists_path = run_dir / "finalists_manifest.json"
    if not finalists_path.exists():
        raise TuningError("finalists_manifest.json is missing. Finish phase 2 before validation.")
    finalists = read_json(finalists_path)["finalists"]
    if finalist_limit is not None:
        if finalist_limit < 1:
            raise TuningError("--finalists must be at least 1")
        finalists = finalists[:finalist_limit]
    if not finalists:
        raise TuningError("There are no successful phase-2 finalists to validate.")
    fingerprints = sample_fingerprints(repo_root, list(manifest["train_samples"]) + validation_samples)
    leaks = leakage_errors(manifest["train_samples"], validation_samples, fingerprints)
    if leaks:
        raise TuningError("Held-out set overlaps training data: " + "; ".join(leaks))
    baseline = load_baseline(baseline_path)
    baseline_errors = validate_candidate(baseline, 2)
    if baseline_errors:
        raise TuningError("baseline failed validation: " + "; ".join(baseline_errors))
    out_dir = run_dir / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline_scores = _score_config(
        repo_root=repo_root,
        run_dir=run_dir,
        config_dir=out_dir / "baseline",
        values=baseline,
        samples=validation_samples,
        manifest=manifest,
        timeout_seconds=timeout_seconds,
        runner=runner,
        label="baseline",
    )
    decisions = []
    for index, finalist in enumerate(finalists):
        scores = _score_config(
            repo_root=repo_root,
            run_dir=run_dir,
            config_dir=out_dir / f"finalist_{index:02d}",
            values=finalist["parameters"],
            samples=validation_samples,
            manifest=manifest,
            timeout_seconds=timeout_seconds,
            runner=runner,
            label=f"finalist_{index:02d}",
        )
        ok, reasons = promote(baseline_scores, scores)
        decisions.append(
            {
                "config_sha256": finalist["config_sha256"],
                "trial": finalist.get("trial"),
                "promoted": ok,
                "reasons": reasons,
                "validation_scores": scores,
                "mean_lift": _mean_lift(baseline_scores, scores),
            }
        )
    promoted = [item for item in decisions if item["promoted"]]
    report = {
        "created_at": utc_now(),
        "decision": "PROMOTED" if promoted else "NOT_PROMOTED",
        "exploratory": exploratory,
        "validation_samples": validation_samples,
        "baseline_scores": baseline_scores,
        "finalists": decisions,
        "chosen": max(promoted, key=lambda item: item["mean_lift"]) if promoted else None,
        "note": "Failed gates are not sent back to Optuna.",
    }
    atomic_json(run_dir / "validation_report.json", report)
    return report


def _mean_lift(baseline_scores: dict[str, float], scores: dict[str, float]) -> float | None:
    if set(baseline_scores) != set(scores) or not scores:
        return None
    deltas = [scores[sample_id] - baseline_scores[sample_id] for sample_id in scores]
    return sum(deltas) / len(deltas)


def _score_config(
    *,
    repo_root: Path,
    run_dir: Path,
    config_dir: Path,
    values: dict[str, Any],
    samples: list[str],
    manifest: dict[str, Any],
    timeout_seconds: int,
    runner: Callable | None,
    label: str,
) -> dict[str, float]:
    from tuning.baseline import config_sha256

    path = config_dir / "gatk.conf"
    write_conf(path, values)
    digest = config_sha256(values)
    scores: dict[str, float] = {}
    for sample_id in samples:
        result = evaluate_sample(
            repo_root=repo_root,
            config_path=path,
            sample_id=sample_id,
            config_sha256=digest,
            expected_version=manifest["scoring_version"],
            timeout_seconds=timeout_seconds,
            lock_path=run_dir / "evaluation.lock",
            log_dir=config_dir / sample_id,
            runner=runner,
        )
        if result.status != "success" or result.combined_final is None:
            raise TuningError(f"{label} on {sample_id} is {result.status}: {result.error}. Not a promotion.")
        scores[sample_id] = float(result.combined_final)
    return scores


def export_promoted(run_dir: Path, output: Path, *, exploratory: bool) -> Path:
    resolved = output.resolve()
    if resolved.name == "gatk.conf" and resolved.parent.name == "configs":
        raise TuningError(f"Refusing to write {resolved}. Copy a promoted file onto the miner config yourself.")
    report_path = run_dir / "validation_report.json"
    if exploratory:
        if "exploratory" not in output.name:
            raise TuningError("An uncertified export filename must contain 'exploratory'.")
        manifest_path = run_dir / "finalists_manifest.json"
        if not manifest_path.exists():
            raise TuningError("No finalists to export.")
        finalists = read_json(manifest_path)["finalists"]
        if not finalists:
            raise TuningError("No finalists to export.")
        write_conf(output, finalists[0]["parameters"])
        header = output.with_suffix(output.suffix + ".EXPLORATORY.txt")
        header.write_text(
            "Exploratory export. Phase 3 did not promote this config. "
            "Do not treat it as a certified miner configuration.\n",
            encoding="utf-8",
        )
        return output
    if not report_path.exists():
        raise TuningError("validation_report.json is missing. Run validate before export.")
    report = read_json(report_path)
    if report.get("decision") != "PROMOTED" or not report.get("chosen"):
        raise TuningError("Phase 3 decision is NOT_PROMOTED. No certified config is exported.")
    chosen = report["chosen"]["config_sha256"]
    match = next(item for item in read_json(run_dir / "finalists_manifest.json")["finalists"] if item["config_sha256"] == chosen)
    write_conf(output, match["parameters"])
    return output

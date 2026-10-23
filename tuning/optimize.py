"""Phase 1 (100) and phase 2 (400) Optuna loops.

``--trials`` is the total for that phase. A resume continues until that many
Optuna trials exist. The untouched baseline run is not one of those trials.
The enqueued baseline point is trial 0 and does count.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Callable

import optuna
from optuna.samplers import TPESampler
from optuna.trial import TrialState

from tuning.baseline import canonical_text, config_sha256, load_baseline, write_conf
from tuning.errors import ScoringVersionAbort, TuningError
from tuning.models import EvaluationResult
from tuning.objective import evaluate_sample, trial_objective
from tuning.legacy import assert_legacy_resolved
from tuning.parameters import (
    PHASE2_EXTRA,
    SEARCH_SPACE_VERSION,
    enqueue_point,
    fixed_names,
    merge_candidate,
    phase_spec,
    suggest,
    tunable_names,
    validate_candidate,
)
from tuning.practice_runner import assert_workers, shared_practice_lock_path
from tuning.storage import (
    append_jsonl,
    assert_baseline_unchanged,
    atomic_json,
    cache_hit,
    read_json,
    result_from_cache,
    snapshot_baseline,
    utc_now,
    write_trial_config,
)

EVALUATOR_VERSION = "tuning-0.2.0"
ARTIFACT_LAYOUT = "phase-trials-v1"
optuna.logging.set_verbosity(optuna.logging.WARNING)


def git_commit(repo_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def git_dirty(repo_root: Path) -> bool | None:
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(repo_root),
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return bool(output.strip())


def spec_sha256(phase: int) -> str:
    blob = json.dumps(phase_spec(phase), sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def freeze_manifest(
    run_dir: Path,
    *,
    baseline_path: Path,
    train_samples: list[str],
    scoring_version: str,
    seed: int,
    robustness_penalty: float,
    repo_root: Path,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    digest = snapshot_baseline(run_dir, baseline_path)
    path = run_dir / "manifest.json"
    commit = git_commit(repo_root)
    dirty = git_dirty(repo_root)
    payload = {
        "baseline_path": str(baseline_path.resolve()),
        "baseline_sha256": digest,
        "train_samples": list(train_samples),
        "scoring_version": scoring_version,
        "seed": seed,
        "robustness_penalty": robustness_penalty,
        "objective": "mean(combined_final) - robustness_penalty * pstdev(combined_final)",
        "single_sample_exploratory": len(train_samples) < 2,
        "git_commit": commit,
        "git_dirty": dirty,
        "cache_enabled": commit is not None and dirty is False,
        "evaluator_version": EVALUATOR_VERSION,
        "search_space_version": SEARCH_SPACE_VERSION,
        "artifact_layout": ARTIFACT_LAYOUT,
        "phase1_spec_sha256": spec_sha256(1),
        "phase2_spec_sha256": spec_sha256(2),
        "created_at": utc_now(),
    }
    if path.exists():
        existing = read_json(path)
        keys = (
            "baseline_sha256",
            "train_samples",
            "scoring_version",
            "seed",
            "robustness_penalty",
            "git_commit",
            "git_dirty",
            "phase1_spec_sha256",
            "phase2_spec_sha256",
            "evaluator_version",
            "search_space_version",
            "artifact_layout",
        )
        for key in keys:
            if existing.get(key) != payload[key]:
                raise TuningError(
                    f"Run manifest {key} is {existing.get(key)!r}; this command would use "
                    f"{payload[key]!r}. Start a new --run-dir. Scores from different "
                    "versions or sample sets are not merged."
                )
        return existing
    atomic_json(path, payload)
    return payload


def _finished(study: optuna.Study) -> int:
    return sum(1 for trial in study.trials if trial.state in {TrialState.COMPLETE, TrialState.FAIL, TrialState.PRUNED})


def _fail_stale(study: optuna.Study) -> None:
    for trial in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
        try:
            study.tell(trial, state=TrialState.FAIL)
        except Exception:
            continue


def _cache_key(manifest: dict[str, Any], config_hash: str, sample_id: str) -> str:
    blob = json.dumps(
        {
            "config": config_hash,
            "sample_id": sample_id,
            "scoring_version": manifest["scoring_version"],
            "git_commit": manifest["git_commit"],
            "evaluator_version": manifest["evaluator_version"],
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def run_phase(
    *,
    repo_root: Path,
    run_dir: Path,
    baseline_path: Path,
    phase: int,
    trials: int,
    train_samples: list[str],
    scoring_version: str,
    seed: int,
    robustness_penalty: float,
    workers: int,
    timeout_seconds: int,
    runner: Callable | None = None,
) -> dict[str, Any]:
    if phase not in (1, 2):
        raise TuningError("phase must be 1 or 2")
    if trials < 1:
        raise TuningError("trials must be >= 1")
    assert_workers(workers)
    assert_legacy_resolved(run_dir)
    manifest = freeze_manifest(
        run_dir,
        baseline_path=baseline_path,
        train_samples=train_samples,
        scoring_version=scoring_version,
        seed=seed,
        robustness_penalty=robustness_penalty,
        repo_root=repo_root,
    )
    assert_baseline_unchanged(baseline_path, manifest["baseline_sha256"])
    baseline = load_baseline(baseline_path)
    study = optuna.create_study(
        study_name=f"phase{phase}",
        storage="sqlite:///" + str((run_dir / "optuna.sqlite3").resolve()),
        sampler=TPESampler(
            seed=seed,
            n_startup_trials=min(10, trials),
            multivariate=True,
        ),
        direction="maximize",
        load_if_exists=True,
    )
    _fail_stale(study)
    _enqueue(study, baseline, phase, run_dir)
    jsonl = run_dir / "trial_results.jsonl"
    while _finished(study) < trials:
        trial = study.ask()
        _execute_trial(
            trial=trial,
            study=study,
            phase=phase,
            baseline=baseline,
            manifest=manifest,
            run_dir=run_dir,
            repo_root=repo_root,
            timeout_seconds=timeout_seconds,
            jsonl=jsonl,
            runner=runner,
            trial_budget=trials,
        )
    assert_baseline_unchanged(baseline_path, manifest["baseline_sha256"])
    from tuning.reports import write_phase_report

    summary = write_phase_report(run_dir, phase, manifest)
    summary["finished"] = _finished(study)
    summary["tunable"] = tunable_names(phase)
    summary["fixed"] = fixed_names(phase)
    return summary


def _enqueue(study: optuna.Study, baseline: dict[str, Any], phase: int, run_dir: Path) -> None:
    if len(study.trials) != 0:
        return
    if phase == 1:
        point = enqueue_point(baseline, 1)
        if point is not None:
            study.enqueue_trial(point)
        return
    for point in phase1_seed_points(run_dir, baseline)[:8]:
        study.enqueue_trial(point)


def phase1_seed_points(run_dir: Path, baseline: dict[str, Any]) -> list[dict[str, Any]]:
    """Distinct successful phase-1 configs, best objective first.

    The seven phase-2 fields stay at the baseline on every seed. Failed,
    incomplete, and invalid records are skipped. The baseline is used when
    phase 1 produced no usable seed.
    """
    from tuning.reports import trial_records

    rows = [row for row in trial_records(run_dir, 1) if row.get("status") == "success" and row.get("objective") is not None]
    rows.sort(key=lambda row: row["objective"], reverse=True)
    seen_configs: set[str] = set()
    seen_points: set[str] = set()
    points: list[dict[str, Any]] = []
    for row in rows:
        digest = row.get("config_sha256")
        if not digest or digest in seen_configs:
            continue
        seen_configs.add(digest)
        point = _phase2_seed_point(row, baseline)
        if point is None:
            continue
        signature = json.dumps(point, sort_keys=True, default=str)
        if signature in seen_points:
            continue
        seen_points.add(signature)
        points.append(point)
        if len(points) >= 8:
            break
    if not points:
        fallback = enqueue_point(baseline, 2)
        if fallback is not None:
            points.append(fallback)
    return points


def _phase2_seed_point(row: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any] | None:
    parameters = row.get("parameters")
    if not isinstance(parameters, dict):
        return None
    merged = dict(baseline)
    for name in tunable_names(1):
        if name not in parameters:
            return None
        merged[name] = parameters[name]
    for name in PHASE2_EXTRA:
        merged[name] = baseline[name]
    if validate_candidate(merged, 2):
        return None
    return enqueue_point(merged, 2)


def _execute_trial(
    *,
    trial: optuna.Trial,
    study: optuna.Study,
    phase: int,
    baseline: dict[str, Any],
    manifest: dict[str, Any],
    run_dir: Path,
    repo_root: Path,
    timeout_seconds: int,
    jsonl: Path,
    runner: Callable | None,
    trial_budget: int,
) -> None:
    suggested = suggest(trial, phase)
    values = merge_candidate(baseline, suggested)
    errors = validate_candidate(values, phase)
    from tuning.reports import trial_dir as phase_trial_dir

    trial_dir = phase_trial_dir(run_dir, phase, trial.number)
    trial_dir.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "phase": phase,
        "trial": trial.number,
        "artifact_dir": str(trial_dir.relative_to(run_dir)),
        "parameters": values,
        "tunable": sorted(suggested),
        "fixed_count": len(fixed_names(phase)),
        "suggested_count": len(suggested),
    }
    if errors:
        record.update(status="invalid_config", objective=None, error="; ".join(errors), samples=[])
        atomic_json(trial_dir / "trial.json", record)
        append_jsonl(jsonl, record)
        study.tell(trial, state=TrialState.FAIL)
        return
    config_path = write_trial_config(trial_dir, values)
    digest = config_sha256(values)
    record["config_sha256"] = digest
    record["config_text_sha256"] = hashlib.sha256(canonical_text(values).encode("utf-8")).hexdigest()
    results: list[EvaluationResult] = []
    changed = [name for name in suggested if values.get(name) != baseline.get(name)]
    for sample_id in manifest["train_samples"]:
        print(
            f"[tune] phase {phase}  trial {trial.number + 1}/{trial_budget}  "
            f"sample {sample_id}  changed {len(changed)} params",
            flush=True,
        )
        sample_dir = trial_dir / sample_id
        key = _cache_key(manifest, digest, sample_id)
        cached = cache_hit(jsonl, key) if manifest.get("cache_enabled") else None
        if cached is not None:
            result = result_from_cache(cached)
        else:
            result = evaluate_sample(
                repo_root=repo_root,
                config_path=config_path,
                sample_id=sample_id,
                config_sha256=digest,
                expected_version=manifest["scoring_version"],
                timeout_seconds=timeout_seconds,
                lock_path=shared_practice_lock_path(repo_root),
                log_dir=sample_dir,
                runner=runner,
            )
        results.append(result)
        score = "none" if result.combined_final is None else f"{result.combined_final:.6f}"
        print(
            f"[tune] phase {phase}  trial {trial.number + 1}/{trial_budget}  "
            f"{result.status}  combined_final {score}  {result.elapsed_seconds:.0f}s",
            flush=True,
        )
        if result.status == "scoring_version_mismatch":
            record.update(status=result.status, objective=None, error=result.error, samples=[item.model_dump() for item in results])
            atomic_json(trial_dir / "trial.json", record)
            append_jsonl(jsonl, record)
            study.tell(trial, state=TrialState.FAIL)
            raise ScoringVersionAbort(result.error or "scoring version mismatch")
    objective = trial_objective(results, manifest["robustness_penalty"])
    record.update(
        status="success" if objective is not None else "failed",
        objective=objective,
        error=None if objective is not None else _failure_text(results),
        samples=[item.model_dump() for item in results],
        cache_key=None,
    )
    atomic_json(trial_dir / "trial.json", record)
    for result in results:
        row = dict(record)
        row["cache_key"] = _cache_key(manifest, digest, result.sample_id)
        row["result"] = result.model_dump()
        row["sample_id"] = result.sample_id
        row["status"] = result.status
        append_jsonl(jsonl, row)
    if objective is None:
        study.tell(trial, state=TrialState.FAIL)
        print(f"[tune] phase {phase}  trial {trial.number + 1}/{trial_budget}  objective failed", flush=True)
    else:
        study.tell(trial, float(objective))
        best = study.best_value
        print(
            f"[tune] phase {phase}  trial {trial.number + 1}/{trial_budget}  "
            f"objective {objective:.6f}  best {best:.6f}",
            flush=True,
        )


def _failure_text(results: list[EvaluationResult]) -> str:
    return "; ".join(f"{item.sample_id}: {item.status} {item.error or ''}".strip() for item in results)


def dry_run_trial(
    *,
    repo_root: Path,
    run_dir: Path,
    baseline_path: Path,
    phase: int,
    sample_id: str,
) -> dict[str, Any]:
    """Write one validated config and the practice command. Do not launch GATK."""
    baseline = load_baseline(baseline_path)
    errors = validate_candidate(baseline, phase)
    if errors:
        raise TuningError("baseline is not a valid phase config: " + "; ".join(errors))
    trial_dir = run_dir / "dry_run"
    path = trial_dir / "gatk.conf"
    write_conf(path, baseline)
    from tuning.practice_runner import practice_command

    command = practice_command(repo_root, path, sample_id)
    return {
        "config": str(path.resolve()),
        "command": command,
        "shell": False,
        "cwd": str(repo_root.resolve()),
        "suggested": len(tunable_names(phase)),
        "fixed": len(fixed_names(phase)),
    }

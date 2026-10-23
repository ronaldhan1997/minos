"""Optuna TPE search over a verified GATK subset.

The objective is the mean official Minos score on the training samples.
Failures are stored and marked ``FAIL`` in Optuna. They are not given a
stand-in score. There is no intermediate official measurement during
HaplotypeCaller, so the pruner is a no-op: a trial is not stopped early on a
proxy.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

import optuna
from optuna.pruners import NopPruner
from optuna.samplers import TPESampler
from optuna.trial import TrialState

from tuner.config_generator import merge_options
from tuner.errors import StudyConflict, TunerError
from tuner.objective import evaluate_candidate, official_mean
from tuner.parameters import SearchSpace, baseline_point, suggest
from tuner.reporting import leaderboard_summary
from tuner.storage import Evaluation, Store

logger = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def assert_worker_budget(workers: int, threads: int, *, cpu: int | None, allow: bool) -> None:
    available = cpu or os.cpu_count() or 1
    if workers < 1 or threads < 1:
        raise TunerError("workers and GATK threads must be >= 1")
    if not allow and workers * threads > available:
        raise TunerError(
            f"{workers} workers × {threads} GATK threads needs {workers * threads} CPUs "
            f"and this machine reports {available}. Each worker also holds the configured "
            f"container memory. The default is 1 worker. Pass --allow-oversubscribe to proceed."
        )


def startup_trials(n_trials: int) -> int:
    """Random warmup before TPE. Capped at 10 and at the trial budget."""
    return min(10, max(1, n_trials))


def run_study(
    *,
    settings,
    study_name: str,
    n_trials: int,
    space: SearchSpace,
    train_samples: list,
    baseline_options: dict[str, Any],
    provenance,
    dry_run: bool = False,
    evaluate_fn: Callable | None = None,
) -> dict[str, Any]:
    if n_trials < 1:
        raise TunerError("trials must be >= 1")
    assert_worker_budget(
        settings.workers,
        settings.gatk_threads,
        cpu=os.cpu_count(),
        allow=settings.allow_oversubscribe,
    )
    from tuner.settings import StudyPaths

    study_paths = StudyPaths.for_study(settings, study_name, dry_run=dry_run)
    study_paths.root.mkdir(parents=True, exist_ok=True)
    _write_space(study_paths.root / "search_space.json", space)
    store = Store(study_paths.evaluations_db)
    try:
        _guard_version(store, study_name, provenance.scoring_version)
        point = baseline_point(space, baseline_options)
        if point is None:
            logger.info(
                "baseline values sit outside the search box; the baseline is still "
                "scored and is not enqueued as an Optuna trial"
            )
        _ensure_baseline(
            store=store,
            study_name=study_name,
            samples=train_samples,
            baseline_options=baseline_options,
            settings=settings,
            provenance=provenance,
            dry_run=dry_run,
            evaluate_fn=evaluate_fn,
        )
        study = optuna.create_study(
            study_name=study_name,
            storage="sqlite:///" + str(study_paths.optuna_db.resolve()),
            sampler=TPESampler(seed=settings.seed, n_startup_trials=startup_trials(n_trials)),
            pruner=NopPruner(),
            direction="maximize",
            load_if_exists=True,
        )
        _fail_stale(study)
        if point is not None and len(study.trials) == 0:
            study.enqueue_trial(point)
        lock = threading.Lock()

        def _execute(trial: optuna.trial.Trial) -> None:
            evaluations: list[Evaluation] = []
            try:
                overrides = suggest(trial, space)
                merged = merge_options(baseline_options, overrides)
                evaluations = _evaluate_all(
                    samples=train_samples,
                    options=merged,
                    store=store,
                    study_name=study_name,
                    settings=settings,
                    provenance=provenance,
                    role="train",
                    trial_number=trial.number,
                    dry_run=dry_run,
                    evaluate_fn=evaluate_fn,
                )
                objective = official_mean(evaluations)
                reason = _trial_reason(evaluations)
                status = "success" if objective is not None else (
                    evaluations[0].status if evaluations else "failed"
                )
            except Exception as exc:  # noqa: BLE001 - one bad trial must not kill the study
                logger.exception("trial crashed")
                objective = None
                reason = str(exc)
                status = "failed"
            config_hash_value = evaluations[0].config_hash if evaluations else ""
            with lock:
                trial.set_user_attr("status", status)
                trial.set_user_attr("failure_reason", reason or "")
                trial.set_user_attr("config_hash", config_hash_value)
                trial.set_user_attr("scoring_version", provenance.scoring_version)
                trial.set_user_attr("scorer_name", provenance.scorer_name)
                trial.set_user_attr("minos_commit", provenance.minos_commit)
                if objective is None:
                    study.tell(trial, state=TrialState.FAIL)
                else:
                    study.tell(trial, float(objective))

        if settings.workers == 1:
            while _finished_count(study) < n_trials:
                _execute(study.ask())
        else:
            reserved = _finished_count(study)

            def _reserved_one() -> None:
                nonlocal reserved
                with lock:
                    if reserved >= n_trials:
                        return
                    reserved += 1
                    trial = study.ask()
                _execute(trial)

            with ThreadPoolExecutor(max_workers=settings.workers) as pool:
                futures = [pool.submit(_reserved_one) for _ in range(n_trials - _finished_count(study))]
                for future in as_completed(futures):
                    future.result()
        logger.info(
            "study=%s trials_finished=%s workers=%s threads=%s memory_gb=%s "
            "estimated_container_ram_gb=%s scoring_version=%s commit=%s",
            study_name,
            _finished_count(study),
            settings.workers,
            settings.gatk_threads,
            settings.memory_gb,
            settings.workers * settings.memory_gb,
            provenance.scoring_version,
            provenance.minos_commit,
        )
        best = _best_value(study)
        summary = leaderboard_summary(
            store.list_study(study_name),
            [sample.sample_id for sample in train_samples],
        )
        return {
            "study": study_name,
            "finished": _finished_count(study),
            "best_value": best,
            "best_adv_score": None if best is None else best / 100.0,
            "dry_run": dry_run,
            "path": str(study_paths.root),
            **summary,
        }
    finally:
        store.close()


def _evaluate_all(
    *,
    samples,
    options,
    store,
    study_name,
    settings,
    provenance,
    role,
    trial_number,
    dry_run,
    evaluate_fn,
) -> list[Evaluation]:
    evaluations = []
    for sample in samples:
        if evaluate_fn is not None:
            evaluations.append(
                evaluate_fn(
                    store=store,
                    study=study_name,
                    sample=sample,
                    options=options,
                    settings=settings,
                    provenance=provenance,
                    role=role,
                    trial_number=trial_number,
                    dry_run=dry_run,
                )
            )
        else:
            evaluations.append(
                evaluate_candidate(
                    store=store,
                    study=study_name,
                    sample=sample,
                    options=options,
                    settings=settings,
                    provenance=provenance,
                    role=role,
                    trial_number=trial_number,
                    dry_run=dry_run,
                )
            )
    return evaluations


def _ensure_baseline(
    *,
    store: Store,
    study_name: str,
    samples: list,
    baseline_options: dict,
    settings,
    provenance,
    dry_run: bool,
    evaluate_fn,
) -> None:
    existing = {
        (item.sample_id, item.config_hash)
        for item in store.list_study(study_name)
        if item.role == "baseline"
    }
    from tuner.config_generator import config_hash, normalize_options

    normalized = normalize_options(baseline_options)
    hashed = config_hash(normalized)
    for sample in samples:
        if (sample.sample_id, hashed) in existing:
            continue
        _evaluate_all(
            samples=[sample],
            options=normalized,
            store=store,
            study_name=study_name,
            settings=settings,
            provenance=provenance,
            role="baseline",
            trial_number=-1,
            dry_run=dry_run,
            evaluate_fn=evaluate_fn,
        )


def _guard_version(store: Store, study_name: str, scoring_version: str) -> None:
    present = store.scoring_versions(study_name)
    others = present - {scoring_version}
    if others:
        raise StudyConflict(
            f"Study {study_name} already has scoring version(s) {sorted(others)}. "
            f"This run resolved {scoring_version}. v1 and v2 are different scales and "
            f"are not mixed. Start a new --study name."
        )


def _fail_stale(study: optuna.Study) -> None:
    for trial in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
        try:
            study.tell(trial.number, state=TrialState.FAIL)
        except Exception:  # noqa: BLE001 - a stale row should not block resume
            logger.warning("could not mark stale trial %s failed", trial.number)


def _finished_count(study: optuna.Study) -> int:
    return sum(
        1
        for trial in study.trials
        if trial.state in {TrialState.COMPLETE, TrialState.FAIL, TrialState.PRUNED}
    )


def _best_value(study: optuna.Study) -> float | None:
    complete = [trial for trial in study.trials if trial.state == TrialState.COMPLETE]
    if not complete:
        return None
    return max(trial.value for trial in complete if trial.value is not None)


def _trial_reason(evaluations: list[Evaluation]) -> str:
    if not evaluations:
        return "no evaluations"
    if all(item.status == "success" for item in evaluations):
        return ""
    parts = [f"{item.sample_id}: {item.status} {item.failure_reason or ''}".strip() for item in evaluations]
    return "; ".join(parts)


def _write_space(path, space: SearchSpace) -> None:
    path.write_text(
        json.dumps(space.model_dump(), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

"""CSV and JSON reports from the evaluation store.

Parameter importance is omitted when Optuna cannot estimate it. That is a
statement about the trial table, not a score.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import optuna
from optuna.trial import TrialState

from tuner.storage import Store
from tuner.validation import HEURISTIC, load_meta


def leaderboard_summary(rows, train_ids: list[str]) -> dict[str, Any]:
    """Baseline mean on the 0–1 leaderboard scale, plus capped samples.

    ``best_value`` from Optuna is the 0–100 official mean. Leaderboard
    ADV. SCORE is that number divided by 100. A sample whose overcall
    penalty is at the official cap of 45 cannot move until a config
    lowers its false-positive rate.
    """
    wanted = set(train_ids)
    by_sample = {}
    for row in rows:
        if row.role != "baseline" or row.sample_id not in wanted:
            continue
        if row.status != "success" or row.score is None:
            continue
        by_sample[row.sample_id] = row
    scores = [row.score for row in by_sample.values()]
    mean = sum(scores) / len(scores) if scores else None
    capped = sorted(
        sample_id
        for sample_id, row in by_sample.items()
        if _penalty_capped(row.submetrics)
    )
    return {
        "baseline_mean": mean,
        "baseline_adv_score": None if mean is None else mean / 100.0,
        "penalty_capped_samples": capped,
        "adv_score_scale": (
            "Leaderboard ADV. SCORE is the official AdvancedScorer value divided by 100. "
            "0.85 on that board is an official mean of 85."
        ),
    }


def _penalty_capped(submetrics: dict | None) -> bool:
    if not submetrics:
        return False
    penalty = submetrics.get("overcall_penalty")
    return isinstance(penalty, (int, float)) and not isinstance(penalty, bool) and penalty >= 44.999


def write_report(settings, study_name: str, output: Path) -> dict[str, Any]:
    from tuner.settings import StudyPaths

    paths = StudyPaths.for_study(settings, study_name, dry_run=False)
    if not paths.evaluations_db.exists():
        raise FileNotFoundError(
            f"No evaluations for study {study_name}. Run tune before report. Expected {paths.evaluations_db}"
        )
    store = Store(paths.evaluations_db)
    try:
        rows = store.list_study(study_name)
    finally:
        store.close()
    versions = sorted({row.scoring_version for row in rows if row.status != "dry_run"})
    warning = None
    if len(versions) > 1:
        warning = (
            "This study contains more than one scoring version "
            f"({', '.join(versions)}). Those numbers are different scales and are not a ranking."
        )
    records = [_flat(row) for row in rows]
    importance, importance_note = _importance(paths.optuna_db, study_name)
    summary = {
        "study": study_name,
        "scoring_versions": versions,
        "warning": warning,
        "heuristic_note": HEURISTIC,
        "trials": records,
        "score_by_sample": _by_sample(rows),
        "failure_reasons": dict(Counter(
            row.failure_reason for row in rows if row.failure_reason
        )),
        "runtime_seconds": _runtime(rows),
        "parameter_importance": importance,
        "parameter_importance_note": importance_note,
        "split": load_meta(paths.meta).model_dump() if paths.meta.exists() else None,
        "selection": json.loads(paths.selection.read_text(encoding="utf-8")) if paths.selection.exists() else None,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    frame = pd.DataFrame.from_records(records)
    frame.to_csv(output, index=False)
    json_path = output.with_suffix(".json")
    json_path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    summary["csv"] = str(output)
    summary["json"] = str(json_path)
    return summary


def _flat(row) -> dict[str, Any]:
    payload = {
        "study": row.study,
        "trial_number": row.trial_number,
        "sample_id": row.sample_id,
        "role": row.role,
        "status": row.status,
        "score": row.score,
        "combined_final": row.combined_final,
        "adv_score": row.combined_final,
        "runtime_seconds": row.runtime_seconds,
        "config_hash": row.config_hash,
        "scoring_version": row.scoring_version,
        "scorer_name": row.scorer_name,
        "minos_commit": row.minos_commit,
        "minos_dirty": row.minos_dirty,
        "gatk_image": row.gatk_image,
        "docker_image_id": row.docker_image_id,
        "region": row.region,
        "failure_reason": row.failure_reason,
        "cached": row.cached,
        "validator_would_record": row.validator_would_record,
        "created_at": row.created_at,
    }
    for key, value in sorted(row.parameters.items()):
        payload[f"param_{key}"] = value
    return payload


def _by_sample(rows) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row.sample_id, []).append(
            {
                "trial_number": row.trial_number,
                "role": row.role,
                "status": row.status,
                "score": row.score,
                "scoring_version": row.scoring_version,
                "config_hash": row.config_hash,
            }
        )
    return grouped


def _runtime(rows) -> dict[str, float | int | None]:
    values = [row.runtime_seconds for row in rows if row.runtime_seconds is not None]
    if not values:
        return {"count": 0, "sum": None, "mean": None, "max": None}
    return {
        "count": len(values),
        "sum": sum(values),
        "mean": sum(values) / len(values),
        "max": max(values),
    }


def _importance(optuna_db: Path, study_name: str) -> tuple[dict[str, float] | None, str]:
    if not optuna_db.exists():
        return None, "No Optuna study file, so parameter importance was not computed."
    try:
        study = optuna.load_study(study_name=study_name, storage="sqlite:///" + str(optuna_db.resolve()))
        complete = [trial for trial in study.trials if trial.state == TrialState.COMPLETE]
        if len(complete) < 2:
            return None, "Fewer than 2 complete trials, so parameter importance is not meaningful."
        values = optuna.importance.get_param_importances(study)
        if not values:
            return None, "Optuna returned no importances."
        return {key: float(value) for key, value in values.items()}, "Fanova importance over complete trials."
    except Exception as exc:  # noqa: BLE001 - importance is optional
        return None, f"Parameter importance is not meaningful for this study: {exc}"

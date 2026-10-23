"""Held-out scoring and the baseline gate.

Optuna only sees the training ids. Ranking uses official scores. The value
``mean - 0.5 * std`` only orders tuned candidates; it is not a Minos score.
"""

from __future__ import annotations

import json
import os
import statistics
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from tuner.config_generator import config_hash, load_conf, normalize_options
from tuner.errors import StudyConflict, TunerError
from tuner.storage import Evaluation


HEURISTIC = (
    "mean(official score) - 0.5 * std(official score) orders tuned candidates only. "
    "It is not the official Minos score."
)


class StudyMeta(BaseModel):
    name: str
    search_space_version: str
    profile: str
    train_sample_ids: list[str]
    val_sample_ids: list[str]
    split_description: str
    seed: int
    minos_commit: str
    scoring_version: str
    created_at: str
    gatk_threads: int
    memory_gb: int


class CandidateSummary(BaseModel):
    label: str
    config_hash: str
    parameters: dict[str, Any]
    per_sample: dict[str, float | None]
    statuses: dict[str, str]
    mean_score: float | None = None
    min_score: float | None = None
    variance: float | None = None
    std: float | None = None
    success_rate: float
    robustness: float | None = None
    complete: bool
    is_baseline: bool = False


class Selection(BaseModel):
    study: str
    scoring_version: str
    heuristic: str = HEURISTIC
    selected_label: str | None = None
    selected_config_hash: str | None = None
    selected_parameters: dict[str, Any] | None = None
    selected_reason: str
    baseline_preferred: bool = False
    tuned_leader_hash: str | None = None
    tuned_leader_parameters: dict[str, Any] | None = None
    tuned_leader_complete: bool = False
    allow_force_tuned: bool = False
    leaderboard: list[dict[str, Any]] = Field(default_factory=list)


def split_samples(
    sample_ids: list[str],
    chromosomes: dict[str, str] | None = None,
) -> tuple[list[str], list[str], str]:
    """Deterministic holdout. Never a score-based split.

    When the ready samples cover two or more chromosomes, the last id on each
    chromosome is held out. Otherwise the lexicographic tail is held out:
    five or more samples keep three held out.
    """
    ordered = sorted(set(sample_ids))
    stratified = _stratified_holdout(ordered, chromosomes)
    if stratified is not None:
        return stratified
    count = len(ordered)
    if count >= 5:
        return ordered[:-3], ordered[-3:], (
            "Sorted sample ids. The last 3 are held out and are not sent to Optuna."
        )
    if count == 4:
        return ordered[:-2], ordered[-2:], (
            "Only 4 ready samples, so 2 are held out. Three held-out samples were not available."
        )
    if count == 3:
        return ordered[:-1], ordered[-1:], (
            "Only 3 ready samples, so 1 is held out. Three held-out samples were not available."
        )
    if count == 2:
        return ordered[:1], ordered[1:], (
            "Only 2 ready samples, so 1 is held out."
        )
    return ordered, [], "Fewer than 2 ready samples, so there is no held-out set."


def _stratified_holdout(
    ordered: list[str],
    chromosomes: dict[str, str] | None,
) -> tuple[list[str], list[str], str] | None:
    """Hold out the lexicographic last id on each chromosome.

    Returns None when the samples do not support that split, so the caller
    keeps the lexicographic tail.
    """
    if not chromosomes or len(ordered) < 5:
        return None
    groups: dict[str, list[str]] = {}
    for sample_id in ordered:
        chromosome = chromosomes.get(sample_id)
        if not chromosome:
            return None
        groups.setdefault(chromosome, []).append(sample_id)
    if len(groups) < 2:
        return None
    heldout = [members[-1] for members in groups.values() if len(members) >= 2]
    if len(heldout) < 2:
        return None
    held_set = set(heldout)
    train = [sample_id for sample_id in ordered if sample_id not in held_set]
    if not train:
        return None
    chromosomes_used = ", ".join(sorted(groups))
    return train, sorted(heldout), (
        f"Held out the last sample id on each of {len(heldout)} chromosomes "
        f"({chromosomes_used}). Those ids are not sent to Optuna."
    )


def load_meta(path) -> StudyMeta | None:
    if not path.exists():
        return None
    return StudyMeta.model_validate(json.loads(path.read_text(encoding="utf-8")))


def save_meta(path, meta: StudyMeta) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(meta.model_dump(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def ensure_split(
    *,
    meta_path,
    study_name: str,
    ready_ids: list[str],
    space,
    settings,
    scoring_version: str,
    commit: str,
    chromosomes: dict[str, str] | None = None,
) -> StudyMeta:
    existing = load_meta(meta_path)
    if existing is not None:
        if existing.search_space_version != space.version or existing.profile != space.profile:
            raise StudyConflict(
                f"Study {study_name} was created with search space "
                f"{existing.profile}/{existing.search_space_version}. "
                f"This run uses {space.profile}/{space.version}. Use a new --study name."
            )
        if existing.scoring_version != scoring_version:
            raise StudyConflict(
                f"Study {study_name} is locked to scoring version {existing.scoring_version}. "
                f"This process resolved {scoring_version}. The two scales are not mixed."
            )
        return existing
    train, heldout, description = split_samples(ready_ids, chromosomes)
    if not train:
        raise TunerError(
            "No training sample is available. prepare must leave at least one runnable sample."
        )
    meta = StudyMeta(
        name=study_name,
        search_space_version=space.version,
        profile=space.profile,
        train_sample_ids=train,
        val_sample_ids=heldout,
        split_description=description,
        seed=settings.seed,
        minos_commit=commit,
        scoring_version=scoring_version,
        created_at=datetime.now(timezone.utc).isoformat(),
        gatk_threads=settings.gatk_threads,
        memory_gb=settings.memory_gb,
    )
    save_meta(meta_path, meta)
    return meta


def summarize_candidate(
    *,
    label: str,
    config_hash_value: str,
    parameters: dict[str, Any],
    rows: list[Evaluation],
    expected_samples: list[str],
    is_baseline: bool,
) -> CandidateSummary:
    by_sample: dict[str, Evaluation] = {}
    for row in rows:
        by_sample[row.sample_id] = row
    per_sample: dict[str, float | None] = {}
    statuses: dict[str, str] = {}
    successes: list[float] = []
    for sample_id in expected_samples:
        row = by_sample.get(sample_id)
        if row is None:
            per_sample[sample_id] = None
            statuses[sample_id] = "missing"
            continue
        statuses[sample_id] = row.status
        per_sample[sample_id] = row.score
        if row.status == "success" and row.score is not None:
            successes.append(row.score)
    complete = len(successes) == len(expected_samples) and len(expected_samples) > 0
    mean = sum(successes) / len(successes) if successes else None
    std = statistics.pstdev(successes) if len(successes) >= 2 else (0.0 if successes else None)
    variance = None if std is None else std ** 2
    robustness = None if mean is None or std is None else mean - 0.5 * std
    return CandidateSummary(
        label=label,
        config_hash=config_hash_value,
        parameters=parameters,
        per_sample=per_sample,
        statuses=statuses,
        mean_score=mean,
        min_score=min(successes) if successes else None,
        variance=variance,
        std=std,
        success_rate=(len(successes) / len(expected_samples)) if expected_samples else 0.0,
        robustness=robustness,
        complete=complete,
        is_baseline=is_baseline,
    )


def select_from_summaries(
    *,
    study: str,
    scoring_version: str,
    baseline: CandidateSummary | None,
    tuned: list[CandidateSummary],
) -> Selection:
    ordered = sorted(
        tuned,
        key=lambda item: (
            item.complete,
            item.robustness if item.robustness is not None else float("-inf"),
            item.mean_score if item.mean_score is not None else float("-inf"),
            item.min_score if item.min_score is not None else float("-inf"),
        ),
        reverse=True,
    )
    leader = ordered[0] if ordered else None
    leaderboard = [item.model_dump() for item in ([baseline] if baseline else []) + ordered]
    if leader is None:
        if baseline and baseline.complete:
            return Selection(
                study=study,
                scoring_version=scoring_version,
                selected_label=baseline.label,
                selected_config_hash=baseline.config_hash,
                selected_parameters=baseline.parameters,
                selected_reason="No tuned candidate finished held-out scoring. Keeping the baseline.",
                baseline_preferred=True,
                allow_force_tuned=False,
                leaderboard=leaderboard,
            )
        return Selection(
            study=study,
            scoring_version=scoring_version,
            selected_reason="No held-out candidate has an official score on every validation sample.",
            leaderboard=leaderboard,
        )
    base_kwargs = dict(
        study=study,
        scoring_version=scoring_version,
        tuned_leader_hash=leader.config_hash,
        tuned_leader_parameters=leader.parameters,
        tuned_leader_complete=leader.complete,
        leaderboard=leaderboard,
    )
    if not leader.complete:
        if baseline and baseline.complete:
            return Selection(
                selected_label=baseline.label,
                selected_config_hash=baseline.config_hash,
                selected_parameters=baseline.parameters,
                selected_reason=(
                    "The robustness leader did not record an official score on every "
                    "held-out sample. Keeping the baseline."
                ),
                baseline_preferred=True,
                allow_force_tuned=False,
                **base_kwargs,
            )
        return Selection(
            selected_reason="Held-out scoring is incomplete for the tuned leader and the baseline.",
            allow_force_tuned=False,
            **base_kwargs,
        )
    if baseline is None or not baseline.complete:
        return Selection(
            selected_reason=(
                "The tuned leader has a complete held-out score, and the baseline does not, "
                "so there is no paired comparison. Pass --force to export the tuned leader."
            ),
            allow_force_tuned=True,
            baseline_preferred=False,
            **base_kwargs,
        )
    prefer, reason = _baseline_gate(leader, baseline)
    if prefer:
        return Selection(
            selected_label=baseline.label,
            selected_config_hash=baseline.config_hash,
            selected_parameters=baseline.parameters,
            selected_reason=reason,
            baseline_preferred=True,
            allow_force_tuned=True,
            **base_kwargs,
        )
    return Selection(
        selected_label=leader.label,
        selected_config_hash=leader.config_hash,
        selected_parameters=leader.parameters,
        selected_reason=reason,
        baseline_preferred=False,
        allow_force_tuned=False,
        **base_kwargs,
    )


def _baseline_gate(tuned: CandidateSummary, baseline: CandidateSummary) -> tuple[bool, str]:
    """Return (prefer_baseline, reason) using official per-sample scores."""
    shared = [
        sample_id
        for sample_id in tuned.per_sample
        if tuned.per_sample[sample_id] is not None and baseline.per_sample.get(sample_id) is not None
    ]
    losses = [
        sample_id
        for sample_id in shared
        if tuned.per_sample[sample_id] < baseline.per_sample[sample_id]
    ]
    if losses:
        return True, (
            "Tuned official score is lower than baseline on held-out sample(s): "
            + ", ".join(losses)
            + ". Keeping the baseline."
        )
    if tuned.mean_score is None or baseline.mean_score is None or tuned.mean_score <= baseline.mean_score:
        return True, (
            "Mean official held-out score is not higher than the baseline. Keeping the baseline."
        )
    if (
        tuned.min_score is not None
        and baseline.min_score is not None
        and tuned.min_score < baseline.min_score
    ):
        return True, (
            "The worst official held-out score is below the baseline minimum. Keeping the baseline."
        )
    return False, (
        "Tuned candidate is at least as high as the baseline on every shared held-out sample "
        "and its mean official score is higher."
    )


def baseline_hash_from_file(path) -> tuple[dict[str, Any], str]:
    options = normalize_options(load_conf(path))
    return options, config_hash(options)


def group_heldout(rows: list[Evaluation]) -> dict[str, list[Evaluation]]:
    grouped: dict[str, list[Evaluation]] = {}
    for row in rows:
        if row.role != "heldout":
            continue
        grouped.setdefault(row.config_hash, []).append(row)
    return grouped


def build_selection(
    *,
    study: str,
    scoring_version: str,
    expected_samples: list[str],
    heldout_rows: list[Evaluation],
    baseline_parameters: dict[str, Any],
    baseline_hash: str,
) -> Selection:
    grouped = group_heldout(heldout_rows)
    baseline_rows = grouped.get(baseline_hash, [])
    baseline = summarize_candidate(
        label="baseline",
        config_hash_value=baseline_hash,
        parameters=baseline_parameters,
        rows=baseline_rows,
        expected_samples=expected_samples,
        is_baseline=True,
    )
    tuned: list[CandidateSummary] = []
    for hashed, rows in grouped.items():
        if hashed == baseline_hash:
            continue
        trial_numbers = [row.trial_number for row in rows if row.trial_number >= 0]
        label = f"trial {max(trial_numbers)}" if trial_numbers else hashed[:12]
        tuned.append(
            summarize_candidate(
                label=label,
                config_hash_value=hashed,
                parameters=rows[0].parameters,
                rows=rows,
                expected_samples=expected_samples,
                is_baseline=False,
            )
        )
    return select_from_summaries(
        study=study,
        scoring_version=scoring_version,
        baseline=baseline,
        tuned=tuned,
    )


def top_training_configs(rows: list[Evaluation], limit: int) -> list[Evaluation]:
    """One representative row per config, best mean official training score first."""
    grouped: dict[str, list[Evaluation]] = {}
    for row in rows:
        if row.role != "train" or row.trial_number < 0 or row.status != "success" or row.score is None:
            continue
        grouped.setdefault(row.config_hash, []).append(row)
    ranked: list[tuple[float, Evaluation]] = []
    for items in grouped.values():
        by_sample: dict[str, Evaluation] = {}
        for item in items:
            by_sample[item.sample_id] = item
        scores = [item.score for item in by_sample.values() if item.score is not None]
        representative = max(by_sample.values(), key=lambda item: item.trial_number)
        ranked.append((sum(scores) / len(scores), representative))
    ranked.sort(key=lambda pair: pair[0], reverse=True)
    return [row for _, row in ranked[:limit]]

"""One trial's objective is the official Minos score.

A tool failure, a timeout, a dry-run, or a score the validator would discard
does not become a number for Optuna. The official number, when one exists, is
still stored.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Callable

from tuner.config_generator import canonical_conf, config_hash, dump_conf, merge_options, normalize_options
from tuner.config_validator import validate_gatk_options
from tuner.datasets import input_hashes
from tuner.runner import run_gatk
from tuner.scorer import score_vcf
from tuner.storage import Evaluation, Store, evaluation_cache_key, utc_now
from tuner.upstream import Provenance

logger = logging.getLogger(__name__)

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def official_mean(evaluations: list[Evaluation]) -> float | None:
    """Mean official score when every training sample would be recorded.

    A failure on one sample fails the trial. The missing sample is not filled
    with zero.
    """
    if not evaluations:
        return None
    if any(item.status != "success" or item.score is None for item in evaluations):
        return None
    return sum(item.score for item in evaluations) / len(evaluations)


def work_directory(root: Path, config_hash_value: str, sample_id: str) -> Path:
    sample_part = _SAFE.sub("_", sample_id).strip("._") or "sample"
    return root / config_hash_value[:16] / sample_part[:120]


def evaluate_candidate(
    *,
    store: Store,
    study: str,
    sample,
    options: dict[str, Any],
    settings,
    provenance: Provenance,
    role: str,
    trial_number: int,
    dry_run: bool = False,
    force_rerun: bool = False,
    gatk_runner: Callable | None = None,
    score_fn: Callable | None = None,
) -> Evaluation:
    """Validate, reuse, call GATK, and score. The returned row is stored."""
    normalized = normalize_options(options)
    hashed = config_hash(normalized)
    hashes = {}
    try:
        hashes = input_hashes(sample, refresh=not dry_run)
    except OSError as exc:
        hashes = {"error": str(exc)}
    cache_parts = {
        "config_hash": hashed,
        "sample_id": sample.sample_id,
        "region": sample.region,
        "input_hashes": hashes,
        "minos_commit": provenance.minos_commit,
        "minos_dirty": provenance.minos_dirty,
        "gatk_image": provenance.gatk_image,
        "docker_image_id": provenance.docker_image_id,
        "happy_image": provenance.happy_image,
        "bcftools_image": provenance.bcftools_image,
        "scoring_version": provenance.scoring_version,
        "scorer_name": provenance.scorer_name,
        "threads": settings.gatk_threads,
        "memory_gb": settings.memory_gb,
    }
    key = evaluation_cache_key(cache_parts)
    work = work_directory(store.path.parent / "work", hashed, sample.sample_id)
    # study paths keep work next to the db; Store.path is evaluations.sqlite3
    # and its parent is the study directory. Callers may pass a store whose
    # parent is that directory. Tests rely on this.
    base = _base_evaluation(
        study=study,
        sample=sample,
        options=normalized,
        provenance=provenance,
        role=role,
        trial_number=trial_number,
        config_hash_value=hashed,
        cache_key=key,
        input_hashes_value=hashes,
        settings=settings,
    )

    validation = validate_gatk_options(normalized)
    if not validation.valid:
        base.status = "failed"
        base.failure_reason = "invalid GATK parameters: " + "; ".join(validation.errors)
        return _logged(store.record(base))

    if not getattr(sample, "ready", False):
        base.status = "failed"
        base.failure_reason = "inputs are not runnable: " + "; ".join(sample.blockers or ["unknown"])
        return _logged(store.record(base))

    if dry_run:
        work.mkdir(parents=True, exist_ok=True)
        conf_path = work / "gatk.conf"
        conf_path.write_text(dump_conf(normalized), encoding="utf-8")
        base.status = "dry_run"
        base.score = None
        base.combined_final = None
        base.paths = {"config": str(conf_path), "work": str(work)}
        base.failure_reason = "dry-run: GATK and hap.py were not started"
        return _logged(store.record(base))

    if not force_rerun:
        cached = store.find_completed(key)
        if cached is not None and cached.scoring_version == provenance.scoring_version:
            reused = cached.model_copy(
                update={
                    "study": study,
                    "role": role,
                    "trial_number": trial_number,
                    "cached": True,
                    "created_at": utc_now(),
                }
            )
            logger.info(
                "cache hit sample=%s config_hash=%s scoring_version=%s score=%s",
                sample.sample_id,
                hashed,
                provenance.scoring_version,
                reused.score,
            )
            return _logged(store.record(reused))

    work.mkdir(parents=True, exist_ok=True)
    conf_path = work / "gatk.conf"
    conf_path.write_text(
        dump_conf(
            normalized,
            header_lines=[
                f"study {study} trial {trial_number} sample {sample.sample_id}",
                f"config_hash {hashed}",
                f"scoring_version {provenance.scoring_version} scorer {provenance.scorer_name}",
                f"minos_commit {provenance.minos_commit} dirty {provenance.minos_dirty}",
            ],
        ),
        encoding="utf-8",
    )
    runner = gatk_runner or run_gatk
    gatk = runner(sample, normalized, work, settings)
    base.runtime_seconds = gatk.runtime_seconds
    base.paths = {"config": str(conf_path), "work": str(work), "log": gatk.log_path, "vcf": gatk.vcf_path}
    base.submetrics = {"gatk_variant_count": gatk.variant_count}
    if not gatk.success:
        base.status = "timeout" if gatk.timed_out else "failed"
        base.score = None
        base.failure_reason = gatk.error
        return _logged(store.record(base))

    scorer = score_fn or score_vcf
    scored = scorer(sample, Path(gatk.vcf_path), work, provenance)
    base.paths.update(
        {
            "raw_metrics": scored.raw_metrics_path,
            "happy_summary": scored.summary_path,
            "happy_vcf": scored.happy_vcf_path,
        }
    )
    base.submetrics.update(scored.projected)
    base.gate_passed = scored.gate_passed
    base.validator_would_record = scored.validator_would_record
    if not scored.ok or scored.advanced_score is None:
        base.status = "failed"
        base.score = None
        base.combined_final = None
        base.failure_reason = scored.failure_reason or "scoring failed"
        return _logged(store.record(base))
    base.score = scored.advanced_score
    base.combined_final = scored.combined_final
    if scored.validator_would_record:
        base.status = "success"
        base.failure_reason = None
    else:
        base.status = "discarded"
        base.failure_reason = scored.failure_reason
    return _logged(store.record(base))


def run_trial_on_samples(
    samples: list,
    options: dict[str, Any],
    **kwargs,
) -> list[Evaluation]:
    return [evaluate_candidate(sample=sample, options=options, **kwargs) for sample in samples]


def _base_evaluation(
    *,
    study: str,
    sample,
    options: dict,
    provenance: Provenance,
    role: str,
    trial_number: int,
    config_hash_value: str,
    cache_key: str,
    input_hashes_value: dict,
    settings,
) -> Evaluation:
    return Evaluation(
        cache_key=cache_key,
        study=study,
        trial_number=trial_number,
        sample_id=sample.sample_id,
        role=role,
        parameters=options,
        status="failed",
        config_hash=config_hash_value,
        input_hashes=input_hashes_value,
        minos_commit=provenance.minos_commit,
        minos_dirty=provenance.minos_dirty,
        gatk_image=provenance.gatk_image,
        docker_image_id=provenance.docker_image_id,
        happy_image=provenance.happy_image,
        bcftools_image=provenance.bcftools_image,
        scorer_name=provenance.scorer_name,
        scoring_version=provenance.scoring_version,
        region=sample.region,
        hardware=provenance.hardware,
        gatk_threads=settings.gatk_threads,
        memory_gb=settings.memory_gb,
        paths={},
    )


def _logged(evaluation: Evaluation) -> Evaluation:
    logger.info(
        "evaluation status=%s score=%s sample=%s trial=%s role=%s commit=%s dirty=%s "
        "scorer=%s scoring_version=%s config_hash=%s cached=%s reason=%s",
        evaluation.status,
        evaluation.score,
        evaluation.sample_id,
        evaluation.trial_number,
        evaluation.role,
        evaluation.minos_commit,
        evaluation.minos_dirty,
        evaluation.scorer_name,
        evaluation.scoring_version,
        evaluation.config_hash,
        evaluation.cached,
        evaluation.failure_reason,
    )
    return evaluation


def rendered_conf(options: dict[str, Any]) -> str:
    return canonical_conf(normalize_options(options))

"""Official Minos score for one VCF.

``HappyScorer.score_vcf`` produces the metrics. ``AdvancedScorer`` turns them
into the number a validator records. Missing metric fields stay null. A hap.py
failure stays null. Scoring v2 does not fall back to v1.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Fields HappyScorer / the synthetic-metric pass may return. Absent keys stay None.
OFFICIAL_METRIC_KEYS = (
    "f1_snp",
    "f1_indel",
    "precision_snp",
    "precision_indel",
    "recall_snp",
    "recall_indel",
    "truth_total_snp",
    "truth_total_indel",
    "query_total_snp",
    "query_total_indel",
    "fp_snp",
    "fp_indel",
    "tp_snp",
    "tp_indel",
    "fn_snp",
    "fn_indel",
    "frac_na_snp",
    "frac_na_indel",
    "titv_truth_snp",
    "titv_query_snp",
    "hethom_truth_snp",
    "hethom_query_snp",
    "hethom_truth_indel",
    "hethom_query_indel",
    "weighted_f1",
    "fp_per_target",
    "overcall_penalty",
    "target_total_snp",
    "target_total_indel",
    "query_unk_snp",
    "query_unk_indel",
)

# Columns the official parser reads from hap.py summary.csv. Used by tests to
# confirm a real summary file still has that contract. This module does not
# recompute a score from those columns.
HAPPY_SUMMARY_COLUMNS = (
    "Type",
    "Filter",
    "TRUTH.TOTAL",
    "TRUTH.TP",
    "TRUTH.FN",
    "QUERY.TOTAL",
    "QUERY.FP",
    "QUERY.UNK",
    "METRIC.Recall",
    "METRIC.Precision",
    "METRIC.Frac_NA",
    "METRIC.F1_Score",
    "TRUTH.TOTAL.TiTv_ratio",
    "QUERY.TOTAL.TiTv_ratio",
    "TRUTH.TOTAL.het_hom_ratio",
    "QUERY.TOTAL.het_hom_ratio",
)


class ScoreRun(BaseModel):
    ok: bool
    advanced_score: float | None = None
    combined_final: float | None = None
    scoring_version: str
    scorer_name: str
    projected: dict[str, Any] = Field(default_factory=dict)
    raw_metrics_path: str | None = None
    summary_path: str | None = None
    happy_vcf_path: str | None = None
    failure_reason: str | None = None
    validator_would_record: bool | None = None
    gate_passed: bool | None = None


def project_metrics(metrics: dict[str, Any] | None) -> dict[str, Any]:
    """Copy official fields. Missing fields stay None and are not filled with 0."""
    projected = {key: None for key in OFFICIAL_METRIC_KEYS}
    if not metrics:
        return projected
    for key in OFFICIAL_METRIC_KEYS:
        if key in metrics and metrics[key] is not None:
            projected[key] = metrics[key]
    return projected


def validator_would_record(metrics: dict[str, Any], combined_final: float | None) -> bool:
    """Same discard rule as ``neurons.miner._run_and_score``.

    A finite combined score in (0, 1] is recorded, except the all-zero
    fingerprint the validator drops. This does not change the official score.
    """
    if not isinstance(combined_final, float) or not math.isfinite(combined_final):
        return False
    valid_range = 0.0 < combined_final <= 1.0
    zero_input = (
        (metrics.get("f1_snp") or 0.0) == 0.0
        and (metrics.get("f1_indel") or 0.0) == 0.0
        and combined_final is not None
        and 0.24999 <= combined_final <= 0.25001
    )
    return valid_range and not zero_input


def summary_path_for(happy_vcf: str | None) -> Path | None:
    if not happy_vcf:
        return None
    path = Path(happy_vcf)
    if path.name.endswith(".gz"):
        candidate = path.with_name(path.name[:-3] + ".summary.csv")
    else:
        candidate = path.with_name(path.name + ".summary.csv")
    return candidate if candidate.exists() else None


def score_from_official_metrics(
    metrics: dict[str, Any] | None,
    *,
    scoring_version: str,
    scorer_name: str,
    truth_vcf: str | None,
    work_dir: Path,
    commit: str,
) -> ScoreRun:
    """Apply AdvancedScorer to a metrics dict already produced by HappyScorer."""
    from utils.scoring import AdvancedScorer, plausibility_gate
    from utils.scoring_version import V2

    work_dir.mkdir(parents=True, exist_ok=True)
    projected = project_metrics(metrics)
    happy_vcf = None if not metrics else metrics.get("happy_vcf_path")
    summary = summary_path_for(happy_vcf if isinstance(happy_vcf, str) else None)
    raw_path = work_dir / "official_metrics.json"

    def _finish(run: ScoreRun) -> ScoreRun:
        raw_path.write_text(
            json.dumps(
                {
                    "minos_commit": commit,
                    "scoring_version": scoring_version,
                    "scorer_name": scorer_name,
                    "advanced_score": run.advanced_score,
                    "failure_reason": run.failure_reason,
                    "metrics": metrics,
                },
                indent=2,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )
        run.raw_metrics_path = str(raw_path)
        run.summary_path = str(summary) if summary else None
        run.happy_vcf_path = happy_vcf if isinstance(happy_vcf, str) else None
        run.projected = projected
        return run

    if metrics is None:
        return _finish(
            ScoreRun(
                ok=False,
                scoring_version=scoring_version,
                scorer_name=scorer_name,
                failure_reason="hap.py returned no metrics",
            )
        )

    gate = None
    try:
        if scoring_version == V2:
            score = _score_v2(metrics, truth_vcf, happy_vcf if isinstance(happy_vcf, str) else None)
            try:
                gate = bool(plausibility_gate(metrics))
            except Exception:  # noqa: BLE001 - the gate is recorded, not required to invent a score
                gate = None
        else:
            score = AdvancedScorer.compute_advanced_score(metrics)
    except Exception as exc:  # noqa: BLE001 - a scorer bug must not become a numeric score
        logger.exception("official scorer raised")
        return _finish(
            ScoreRun(
                ok=False,
                scoring_version=scoring_version,
                scorer_name=scorer_name,
                failure_reason=f"official scorer failed: {exc}",
                gate_passed=gate,
            )
        )

    if score is None or isinstance(score, bool) or not isinstance(score, (int, float)):
        return _finish(
            ScoreRun(
                ok=False,
                scoring_version=scoring_version,
                scorer_name=scorer_name,
                failure_reason=(
                    "v2 could not be computed. The v1 score is a different scale "
                    "and is not substituted."
                    if scoring_version == V2
                    else "official scorer returned no score"
                ),
                gate_passed=gate,
            )
        )
    numeric = float(score)
    if not math.isfinite(numeric):
        return _finish(
            ScoreRun(
                ok=False,
                scoring_version=scoring_version,
                scorer_name=scorer_name,
                failure_reason=f"official score is not finite: {numeric!r}",
                gate_passed=gate,
            )
        )
    combined = numeric / 100.0
    would = validator_would_record(metrics, combined)
    return _finish(
        ScoreRun(
            ok=True,
            advanced_score=numeric,
            combined_final=combined,
            scoring_version=scoring_version,
            scorer_name=scorer_name,
            failure_reason=None if would else "validator would discard this score",
            validator_would_record=would,
            gate_passed=gate,
        )
    )


def _score_v2(metrics: dict[str, Any], truth_vcf: str | None, happy_vcf: str | None) -> float | None:
    from utils.scoring import AdvancedScorer, difficulty_class_counts, parse_happy_vcf

    if not happy_vcf or not Path(happy_vcf).exists():
        logger.error("v2 score needs the annotated hap.py VCF")
        return None
    records = parse_happy_vcf(happy_vcf, truth_vcf_path=truth_vcf)
    if records is None:
        logger.error("v2: hap.py VCF could not be parsed completely")
        return None
    if not records:
        logger.error("v2: annotated VCF contained no variant records")
        return None
    return AdvancedScorer.compute_score_v2(metrics, difficulty_class_counts(records))


def score_vcf(sample, vcf_path: Path, work_dir: Path, provenance) -> ScoreRun:
    """Run hap.py through HappyScorer, then the official AdvancedScorer."""
    from utils.scoring import HappyScorer

    if not sample.sdf:
        return score_from_official_metrics(
            None,
            scoring_version=provenance.scoring_version,
            scorer_name=provenance.scorer_name,
            truth_vcf=sample.truth,
            work_dir=work_dir,
            commit=provenance.minos_commit,
        )
    scorer = HappyScorer()
    metrics = scorer.score_vcf(
        truth_vcf=str(sample.truth),
        query_vcf=str(vcf_path),
        reference_fasta=str(sample.reference),
        confident_bed=None,
        region=sample.region,
        reference_sdf=str(sample.sdf),
        mutations_vcf=str(sample.mutations) if sample.mutations else None,
    )
    run = score_from_official_metrics(
        metrics,
        scoring_version=provenance.scoring_version,
        scorer_name=provenance.scorer_name,
        truth_vcf=sample.truth,
        work_dir=work_dir,
        commit=provenance.minos_commit,
    )
    logger.info(
        "score sample=%s commit=%s dirty=%s scoring_version=%s scorer=%s score=%s",
        sample.sample_id,
        provenance.minos_commit,
        provenance.minos_dirty,
        provenance.scoring_version,
        provenance.scorer_name,
        run.advanced_score,
    )
    return run

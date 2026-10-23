"""Extract a validator-recordable COMBINED_FINAL from practice output.

A numeric COMBINED_FINAL is not enough. The line must say a validator would
record it. ZERO-INPUT, discard, and out-of-range lines stay failures even
when they contain a number.
"""

from __future__ import annotations

import re
from typing import Any

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_RICH = re.compile(r"\[/?[a-zA-Z0-9 _.#-]*\]")
_SCORE = re.compile(r"COMBINED_FINAL:?\s+([0-9]+(?:\.[0-9]+)?)")
_VERSION = re.compile(r"scoring\s+(v1|v2)\b", re.IGNORECASE)
_SNP = re.compile(
    r"SNP\s+F1=([0-9.]+)\s+recall=([0-9.]+)\s+prec=([0-9.]+)\s+FP=([0-9]+)",
    re.IGNORECASE,
)
_INDEL = re.compile(
    r"INDEL\s+F1=([0-9.]+)\s+recall=([0-9.]+)\s+prec=([0-9.]+)\s+FP=([0-9]+)",
    re.IGNORECASE,
)


def strip_markup(text: str) -> str:
    return _RICH.sub("", _ANSI.sub("", text or ""))


def parse_practice_output(text: str, *, expected_version: str | None) -> dict[str, Any]:
    """Classify combined stdout and stderr from one practice run."""
    plain = strip_markup(text or "")
    score_match = _SCORE.search(plain)
    number = float(score_match.group(1)) if score_match else None
    version_match = _VERSION.search(plain)
    found_version = version_match.group(1).lower() if version_match else None
    lowered = plain.lower()
    discarded = (
        "zero-input" in lowered
        or "validator discards" in lowered
        or "validator discards this" in lowered
        or "out of range" in lowered
    )
    recordable = (
        "what a validator records" in lowered
        or "what a validator would record" in lowered
    )
    metrics = _diagnostic_metrics(plain)
    if found_version and expected_version and found_version != expected_version:
        return _result(
            "scoring_version_mismatch",
            number,
            found_version,
            metrics,
            f"practice reported scoring {found_version}; study is pinned to {expected_version}",
        )
    if expected_version and found_version is None and score_match:
        return _result(
            "score_error",
            number,
            None,
            metrics,
            "practice output did not report a scoring version",
        )
    if discarded:
        return _result(
            "discarded",
            number,
            found_version,
            metrics,
            "validator would not record this result",
        )
    if number is None or not recordable:
        return _result("score_error", number, found_version, metrics, "recordable COMBINED_FINAL was not found")
    if not (number == number) or not (0.0 < number <= 1.0):
        return _result("discarded", number, found_version, metrics, "COMBINED_FINAL is outside (0, 1]")
    return _result("success", number, found_version or expected_version, metrics, None)


def _diagnostic_metrics(plain: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    snp = _SNP.search(plain)
    indel = _INDEL.search(plain)
    if snp:
        metrics.update(
            f1_snp=float(snp.group(1)),
            recall_snp=float(snp.group(2)),
            precision_snp=float(snp.group(3)),
            fp_snp=float(snp.group(4)),
        )
    if indel:
        metrics.update(
            f1_indel=float(indel.group(1)),
            recall_indel=float(indel.group(2)),
            precision_indel=float(indel.group(3)),
            fp_indel=float(indel.group(4)),
        )
    return metrics


def _result(status: str, number: float | None, version: str | None, metrics: dict, error: str | None) -> dict[str, Any]:
    advanced = None if number is None else number * 100.0
    return {
        "status": status,
        "combined_final": number if status == "success" else number,
        "advanced_score": advanced if status == "success" else None,
        "scoring_version": version,
        "metrics": metrics,
        "error": error,
    }

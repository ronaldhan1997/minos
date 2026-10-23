"""Typed practice-evaluation records."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

STATUSES = (
    "success",
    "invalid_config",
    "execution_error",
    "score_error",
    "discarded",
    "scoring_version_mismatch",
    "timeout",
    "interrupted",
    "dry_run",
)


class EvaluationResult(BaseModel):
    sample_id: str
    config_sha256: str
    scoring_version: str | None = None
    status: str
    combined_final: float | None = None
    advanced_score: float | None = None
    elapsed_seconds: float = 0.0
    exit_code: int | None = None
    log_dir: str = ""
    metrics: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    reused: bool = False
    attempt: int = 1

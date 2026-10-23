"""SQLite log of official evaluations.

A completed score is reused when every result-affecting input matches,
including the Minos commit, GATK image id, and scoring version. A different
scoring version is a different cache key. Failed runs are not reused, so a
transient container error can be retried.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def evaluation_cache_key(parts: dict[str, Any]) -> str:
    blob = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class Evaluation(BaseModel):
    cache_key: str
    study: str
    trial_number: int
    sample_id: str
    role: str
    parameters: dict[str, Any]
    status: str
    score: float | None = None
    combined_final: float | None = None
    submetrics: dict[str, Any] = Field(default_factory=dict)
    runtime_seconds: float | None = None
    config_hash: str
    input_hashes: dict[str, Any] = Field(default_factory=dict)
    minos_commit: str
    minos_dirty: bool = False
    gatk_image: str
    docker_image_id: str | None = None
    happy_image: str = ""
    bcftools_image: str = ""
    scorer_name: str
    scoring_version: str
    paths: dict[str, Any] = Field(default_factory=dict)
    failure_reason: str | None = None
    region: str | None = None
    hardware: dict[str, Any] = Field(default_factory=dict)
    validator_would_record: bool | None = None
    gate_passed: bool | None = None
    cached: bool = False
    gatk_threads: int = 1
    memory_gb: int = 4
    created_at: str = Field(default_factory=utc_now)


_COLUMNS = (
    "cache_key",
    "study",
    "trial_number",
    "sample_id",
    "role",
    "parameters_json",
    "status",
    "score",
    "combined_final",
    "submetrics_json",
    "runtime_seconds",
    "config_hash",
    "input_hashes_json",
    "minos_commit",
    "minos_dirty",
    "gatk_image",
    "docker_image_id",
    "happy_image",
    "bcftools_image",
    "scorer_name",
    "scoring_version",
    "paths_json",
    "failure_reason",
    "region",
    "hardware_json",
    "validator_would_record",
    "gate_passed",
    "cached",
    "gatk_threads",
    "memory_gb",
    "created_at",
)


class Store:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._init()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _init(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cache_key TEXT NOT NULL,
                study TEXT NOT NULL,
                trial_number INTEGER NOT NULL,
                sample_id TEXT NOT NULL,
                role TEXT NOT NULL,
                parameters_json TEXT NOT NULL,
                status TEXT NOT NULL,
                score REAL,
                combined_final REAL,
                submetrics_json TEXT NOT NULL,
                runtime_seconds REAL,
                config_hash TEXT NOT NULL,
                input_hashes_json TEXT NOT NULL,
                minos_commit TEXT NOT NULL,
                minos_dirty INTEGER NOT NULL,
                gatk_image TEXT NOT NULL,
                docker_image_id TEXT,
                happy_image TEXT NOT NULL,
                bcftools_image TEXT NOT NULL,
                scorer_name TEXT NOT NULL,
                scoring_version TEXT NOT NULL,
                paths_json TEXT NOT NULL,
                failure_reason TEXT,
                region TEXT,
                hardware_json TEXT NOT NULL,
                validator_would_record INTEGER,
                gate_passed INTEGER,
                cached INTEGER NOT NULL,
                gatk_threads INTEGER NOT NULL,
                memory_gb INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE (study, role, trial_number, sample_id, cache_key)
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_eval_cache ON evaluations(cache_key, status)"
        )
        self._conn.commit()

    def record(self, evaluation: Evaluation) -> Evaluation:
        """Insert an evaluation, or replace an earlier attempt at the same key.

        A completed score is kept when the new row is only a cache replay.
        A failed, timed-out, or dry-run row does not block a later attempt:
        those runs are not a cache hit. A fresh recompute of the same trial
        replaces the stored row, including ``--force-rerun``.
        """
        with self._lock:
            existing = self._find_exact(evaluation)
            if existing is not None:
                completed = existing.status in {"success", "discarded"}
                if completed and evaluation.cached:
                    return existing
                self._delete_exact(evaluation)
            self._insert(evaluation)
            self._conn.commit()
            return evaluation

    def find_completed(self, cache_key: str) -> Evaluation | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT * FROM evaluations
                WHERE cache_key = ? AND status IN ('success', 'discarded')
                ORDER BY id DESC LIMIT 1
                """,
                (cache_key,),
            ).fetchone()
            return _from_row(row) if row else None

    def list_study(self, study: str) -> list[Evaluation]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM evaluations WHERE study = ? ORDER BY id",
                (study,),
            ).fetchall()
            return [_from_row(row) for row in rows]

    def scoring_versions(self, study: str) -> set[str]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT DISTINCT scoring_version FROM evaluations
                WHERE study = ? AND status != 'dry_run'
                """,
                (study,),
            ).fetchall()
            return {row["scoring_version"] for row in rows}

    def _find_exact(self, evaluation: Evaluation) -> Evaluation | None:
        row = self._conn.execute(
            """
            SELECT * FROM evaluations
            WHERE study = ? AND role = ? AND trial_number = ? AND sample_id = ? AND cache_key = ?
            """,
            (
                evaluation.study,
                evaluation.role,
                evaluation.trial_number,
                evaluation.sample_id,
                evaluation.cache_key,
            ),
        ).fetchone()
        return _from_row(row) if row else None

    def _delete_exact(self, evaluation: Evaluation) -> None:
        self._conn.execute(
            """
            DELETE FROM evaluations
            WHERE study = ? AND role = ? AND trial_number = ? AND sample_id = ? AND cache_key = ?
            """,
            (
                evaluation.study,
                evaluation.role,
                evaluation.trial_number,
                evaluation.sample_id,
                evaluation.cache_key,
            ),
        )

    def _insert(self, evaluation: Evaluation) -> None:
        payload = evaluation.model_dump()
        values = {
            "cache_key": payload["cache_key"],
            "study": payload["study"],
            "trial_number": payload["trial_number"],
            "sample_id": payload["sample_id"],
            "role": payload["role"],
            "parameters_json": _dump(payload["parameters"]),
            "status": payload["status"],
            "score": payload["score"],
            "combined_final": payload["combined_final"],
            "submetrics_json": _dump(payload["submetrics"]),
            "runtime_seconds": payload["runtime_seconds"],
            "config_hash": payload["config_hash"],
            "input_hashes_json": _dump(payload["input_hashes"]),
            "minos_commit": payload["minos_commit"],
            "minos_dirty": int(bool(payload["minos_dirty"])),
            "gatk_image": payload["gatk_image"],
            "docker_image_id": payload["docker_image_id"],
            "happy_image": payload["happy_image"],
            "bcftools_image": payload["bcftools_image"],
            "scorer_name": payload["scorer_name"],
            "scoring_version": payload["scoring_version"],
            "paths_json": _dump(payload["paths"]),
            "failure_reason": payload["failure_reason"],
            "region": payload["region"],
            "hardware_json": _dump(payload["hardware"]),
            "validator_would_record": _bit(payload["validator_would_record"]),
            "gate_passed": _bit(payload["gate_passed"]),
            "cached": int(bool(payload["cached"])),
            "gatk_threads": payload["gatk_threads"],
            "memory_gb": payload["memory_gb"],
            "created_at": payload["created_at"],
        }
        columns = ", ".join(_COLUMNS)
        placeholders = ", ".join(f":{name}" for name in _COLUMNS)
        self._conn.execute(
            f"INSERT INTO evaluations ({columns}) VALUES ({placeholders})",
            values,
        )


def _bit(value: bool | None) -> int | None:
    if value is None:
        return None
    return int(bool(value))


def _dump(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _load(value: str | None) -> Any:
    if not value:
        return {}
    return json.loads(value)


def _from_row(row: sqlite3.Row) -> Evaluation:
    return Evaluation(
        cache_key=row["cache_key"],
        study=row["study"],
        trial_number=row["trial_number"],
        sample_id=row["sample_id"],
        role=row["role"],
        parameters=_load(row["parameters_json"]),
        status=row["status"],
        score=row["score"],
        combined_final=row["combined_final"],
        submetrics=_load(row["submetrics_json"]),
        runtime_seconds=row["runtime_seconds"],
        config_hash=row["config_hash"],
        input_hashes=_load(row["input_hashes_json"]),
        minos_commit=row["minos_commit"],
        minos_dirty=bool(row["minos_dirty"]),
        gatk_image=row["gatk_image"],
        docker_image_id=row["docker_image_id"],
        happy_image=row["happy_image"],
        bcftools_image=row["bcftools_image"],
        scorer_name=row["scorer_name"],
        scoring_version=row["scoring_version"],
        paths=_load(row["paths_json"]),
        failure_reason=row["failure_reason"],
        region=row["region"],
        hardware=_load(row["hardware_json"]),
        validator_would_record=(
            None if row["validator_would_record"] is None else bool(row["validator_would_record"])
        ),
        gate_passed=None if row["gate_passed"] is None else bool(row["gate_passed"]),
        cached=bool(row["cached"]),
        gatk_threads=row["gatk_threads"],
        memory_gb=row["memory_gb"],
        created_at=row["created_at"],
    )

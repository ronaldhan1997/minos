"""Run directory: frozen manifest, trial artifacts, and the success cache."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tuning.baseline import file_sha256, write_conf
from tuning.errors import TuningError
from tuning.models import EvaluationResult


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def snapshot_baseline(run_dir: Path, baseline_path: Path) -> str:
    digest = file_sha256(baseline_path)
    destination = run_dir / "baseline" / "gatk.original.conf"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        destination.write_bytes(baseline_path.read_bytes())
    elif file_sha256(destination) != digest:
        raise TuningError(
            f"{baseline_path} changed after the run snapshot. "
            "Start a new --run-dir. This tuner does not rewrite the miner config."
        )
    if file_sha256(baseline_path) != digest:
        raise TuningError(f"{baseline_path} hash changed while snapshotting")
    return digest


def assert_baseline_unchanged(baseline_path: Path, expected_sha256: str) -> None:
    current = file_sha256(baseline_path)
    if current != expected_sha256:
        raise TuningError(
            f"{baseline_path} sha256 changed from {expected_sha256} to {current}. "
            "Stop. The miner config must stay byte-for-byte identical."
        )


def load_sample_file(path: Path) -> list[str]:
    ids: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            ids.append(line)
    if not ids:
        raise TuningError(f"{path} has no sample ids")
    return ids


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def cache_hit(path: Path, cache_key: str) -> dict[str, Any] | None:
    for row in reversed(iter_jsonl(path)):
        if row.get("cache_key") == cache_key and row.get("status") == "success":
            return row
    return None


def result_from_cache(row: dict[str, Any]) -> EvaluationResult:
    result = EvaluationResult.model_validate(row["result"])
    result.reused = True
    return result


def write_trial_config(trial_dir: Path, values: dict[str, Any]) -> Path:
    path = trial_dir / "gatk.conf"
    write_conf(path, values)
    return path

"""Non-destructive migration of the old shared ``trials/`` layout.

Phase 1 and phase 2 used to write ``trials/{trial:06d}`` with independent
Optuna numbering, so a later phase could overwrite an earlier trial. This
module copies that tree into ``legacy_archive/`` and, only when SQLite and
JSONL agree on a single owner for every trial number, copies those records
into ``phase{n}/trials/``. It never deletes the original directory and never
guesses which overwritten file belonged to which phase.
"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import optuna

from tuning.storage import atomic_json, iter_jsonl, read_json

STATUS_NAME = "legacy_status.json"


def migrate_legacy_run(run_dir: Path) -> dict[str, Any]:
    """Copy a legacy trial tree aside. Return the status written to disk."""
    run_dir = Path(run_dir)
    status_path = run_dir / STATUS_NAME
    if status_path.exists():
        return read_json(status_path)
    legacy = run_dir / "trials"
    records = _legacy_records(legacy)
    if not records:
        payload = {
            "status": "absent",
            "reason": "no legacy trials/*/trial.json files",
            "copied": 0,
        }
        atomic_json(status_path, payload)
        return payload
    archive = _archive_legacy(run_dir, legacy)
    reasons = _ambiguity_reasons(run_dir, records)
    reasons.extend(_destination_conflicts(run_dir, records))
    if reasons:
        payload = {
            "status": "untrusted",
            "reason": " ".join(reasons),
            "archive": str(archive),
            "copied": 0,
            "note": "Original trials/ was left in place. Start a new --run-dir.",
        }
        atomic_json(status_path, payload)
        return payload
    copied = _copy_into_phase_dirs(run_dir, records)
    payload = {
        "status": "migrated",
        "reason": "SQLite and JSONL assign every legacy trial to one phase",
        "archive": str(archive),
        "copied": copied,
        "note": "Original trials/ was left in place.",
    }
    atomic_json(status_path, payload)
    return payload


def assert_legacy_resolved(run_dir: Path) -> None:
    """Refuse to resume a run whose old shared trial tree has not been classified."""
    from tuning.errors import TuningError

    legacy = run_dir / "trials"
    if not legacy.is_dir() or not any(legacy.glob("*/trial.json")):
        return
    status_path = run_dir / STATUS_NAME
    if not status_path.exists():
        raise TuningError(
            f"{legacy} uses the shared trial layout. Run "
            "`python -m tuning.cli migrate-legacy --run-dir "
            f"{run_dir}` before resuming. Overlapping phase histories are marked "
            "untrusted and need a new --run-dir."
        )
    status = read_json(status_path).get("status")
    if status == "untrusted":
        raise TuningError(
            "Legacy trials cannot be attributed to one phase. "
            "Start a new --run-dir. trials/ was not deleted."
        )


def _legacy_records(legacy: Path) -> list[tuple[int, Path, dict[str, Any]]]:
    if not legacy.is_dir():
        return []
    found: list[tuple[int, Path, dict[str, Any]]] = []
    for path in sorted(legacy.glob("*/trial.json")):
        row = read_json(path)
        try:
            number = int(path.parent.name)
        except ValueError:
            number = -1
        found.append((number, path.parent, row))
    return found


def _archive_legacy(run_dir: Path, legacy: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = run_dir / "legacy_archive" / stamp
    suffix = 0
    while archive.exists():
        suffix += 1
        archive = run_dir / "legacy_archive" / f"{stamp}-{suffix}"
    archive.mkdir(parents=True)
    shutil.copytree(legacy, archive / "trials")
    for name in ("trial_results.jsonl", "optuna.sqlite3"):
        source = run_dir / name
        if source.is_file():
            shutil.copy2(source, archive / name)
    return archive


def _ambiguity_reasons(run_dir: Path, records: list[tuple[int, Path, dict[str, Any]]]) -> list[str]:
    reasons: list[str] = []
    db_path = run_dir / "optuna.sqlite3"
    jsonl_path = run_dir / "trial_results.jsonl"
    if not db_path.is_file():
        reasons.append("optuna.sqlite3 is missing, so phase ownership cannot be verified.")
    if not jsonl_path.is_file():
        reasons.append("trial_results.jsonl is missing, so phase ownership cannot be verified.")
    sqlite_numbers = _sqlite_trial_numbers(db_path)
    overlap = sqlite_numbers.get(1, set()) & sqlite_numbers.get(2, set())
    if overlap:
        shown = ", ".join(str(number) for number in sorted(overlap)[:8])
        reasons.append(
            f"SQLite phase1 and phase2 both contain trial numbers [{shown}]. "
            "The shared trials/ directory is not attributed."
        )
    jsonl_rows = iter_jsonl(jsonl_path)
    jsonl_phases: dict[int, set[int]] = {}
    jsonl_hashes: dict[tuple[int, int], set[str]] = {}
    for row in jsonl_rows:
        phase = row.get("phase")
        number = row.get("trial")
        if phase not in (1, 2) or not isinstance(number, int):
            continue
        jsonl_phases.setdefault(number, set()).add(phase)
        digest = row.get("config_sha256")
        if isinstance(digest, str):
            jsonl_hashes.setdefault((phase, number), set()).add(digest)
    jsonl_overlap = {number for number, phases in jsonl_phases.items() if len(phases) > 1}
    if jsonl_overlap:
        shown = ", ".join(str(number) for number in sorted(jsonl_overlap)[:8])
        reasons.append(f"JSONL attributes trial numbers [{shown}] to both phases.")
    for number, _directory, row in records:
        phase = row.get("phase")
        if number < 0:
            reasons.append("a legacy trial directory name is not an integer.")
            continue
        if row.get("trial") != number:
            reasons.append(f"trial directory {number:06d} does not match trial.json.")
        if phase not in (1, 2):
            reasons.append(f"trial {number} has no phase field.")
            continue
        if overlap or jsonl_overlap:
            continue
        if db_path.is_file() and number not in sqlite_numbers.get(phase, set()):
            reasons.append(f"trial {number} says phase {phase} but that SQLite study does not contain it.")
        if jsonl_path.is_file():
            hashes = jsonl_hashes.get((phase, number), set())
            file_hash = row.get("config_sha256")
            if phase not in jsonl_phases.get(number, set()):
                reasons.append(f"trial {number} has no JSONL row for phase {phase}.")
            elif isinstance(file_hash, str) and hashes and file_hash not in hashes:
                reasons.append(f"trial {number} config hash disagrees with JSONL.")
            elif len(hashes) > 1:
                reasons.append(f"trial {number} JSONL rows disagree on config hash.")
    return reasons


def _sqlite_trial_numbers(db_path: Path) -> dict[int, set[int]]:
    numbers = {1: set(), 2: set()}
    if not db_path.is_file():
        return numbers
    storage = "sqlite:///" + str(db_path.resolve())
    try:
        summaries = optuna.get_all_study_summaries(storage)
    except Exception:
        return numbers
    present = {item.study_name for item in summaries}
    for phase in (1, 2):
        name = f"phase{phase}"
        if name not in present:
            continue
        study = optuna.load_study(study_name=name, storage=storage)
        numbers[phase] = {trial.number for trial in study.trials}
    return numbers


def _destination_conflicts(run_dir: Path, records: list[tuple[int, Path, dict[str, Any]]]) -> list[str]:
    reasons: list[str] = []
    for number, directory, row in records:
        phase = row.get("phase")
        if phase not in (1, 2) or number < 0:
            continue
        destination = run_dir / f"phase{phase}" / "trials" / f"{number:06d}"
        existing = destination / "trial.json"
        source = directory / "trial.json"
        if destination.exists() and (not existing.is_file() or existing.read_bytes() != source.read_bytes()):
            reasons.append(f"{destination} already exists and differs from the legacy trial.")
    return reasons


def _copy_into_phase_dirs(run_dir: Path, records: list[tuple[int, Path, dict[str, Any]]]) -> int:
    copied = 0
    for number, directory, row in records:
        phase = row["phase"]
        destination = run_dir / f"phase{phase}" / "trials" / f"{number:06d}"
        if destination.exists():
            continue
        shutil.copytree(directory, destination)
        copied += 1
    return copied

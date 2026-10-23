"""Write a validator-compatible ``gatk.conf`` and a provenance sidecar.

``--force`` never skips parameter validation, a failed evaluation, or the
miner config at ``configs/gatk.conf``. It only covers two benign cases:
exporting the best training config when held-out validation has not been run,
and exporting the tuned leader when the held-out gate kept the baseline.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from tuner.config_generator import dump_conf, normalize_options
from tuner.config_validator import validate_gatk_options
from tuner.errors import ExportError
from tuner.storage import Store
from tuner.validation import HEURISTIC, Selection, load_meta, top_training_configs


def export_config(
    settings,
    study_name: str,
    output: Path,
    *,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    from tuner.settings import StudyPaths

    _refuse_active_miner_config(output, settings.repo_root)
    paths = StudyPaths.for_study(settings, study_name, dry_run=False)
    if not paths.evaluations_db.exists():
        raise ExportError(f"No evaluations for study {study_name}.")
    store = Store(paths.evaluations_db)
    try:
        rows = store.list_study(study_name)
    finally:
        store.close()
    versions = {row.scoring_version for row in rows if row.status != "dry_run"}
    if len(versions) > 1:
        raise ExportError(
            "Study mixes scoring versions "
            + ", ".join(sorted(versions))
            + ". Those scales are not exported together."
        )
    selection = None
    if paths.selection.exists():
        selection = Selection.model_validate(json.loads(paths.selection.read_text(encoding="utf-8")))
    parameters, reason, label = _choose(rows, selection, force=force)
    _require_success(rows, parameters)
    normalized = normalize_options(parameters)
    checked = validate_gatk_options(normalized)
    if not checked.valid:
        raise ExportError(
            "Export refused: official parameter validation failed: " + "; ".join(checked.errors)
        )
    meta = load_meta(paths.meta)
    provenance = {
        "study": study_name,
        "label": label,
        "reason": reason,
        "heuristic": HEURISTIC,
        "parameters": normalized,
        "flags": checked.flags,
        "scoring_version": next(iter(versions)) if versions else (meta.scoring_version if meta else None),
        "split": meta.model_dump() if meta else None,
        "selection": selection.model_dump() if selection else None,
        "evidence": [
            {
                "sample_id": row.sample_id,
                "role": row.role,
                "status": row.status,
                "score": row.score,
                "combined_final": row.combined_final,
                "scoring_version": row.scoring_version,
                "scorer_name": row.scorer_name,
                "config_hash": row.config_hash,
                "minos_commit": row.minos_commit,
                "minos_dirty": row.minos_dirty,
                "gatk_image": row.gatk_image,
                "docker_image_id": row.docker_image_id,
                "region": row.region,
            }
            for row in rows
            if row.parameters == normalized or row.config_hash == _hash(normalized)
        ],
        "next_step": (
            "Re-score this file locally with `python -m tuner.cli test`. "
            "Copy it onto configs/gatk.conf only after that score is acceptable. "
            "The tuner does not submit it."
        ),
    }
    header = [
        f"Minos GATK tuner export for study {study_name}",
        f"label: {label}",
        f"reason: {reason}",
        "Official parameter validation passed. Provenance is the sidecar JSON.",
    ]
    body = dump_conf(normalized, header_lines=header)
    # Parse the bytes we are about to write and validate them again.
    from tuner.config_generator import parse_conf

    rewritten = parse_conf(body, origin=str(output))
    again = validate_gatk_options(rewritten)
    if not again.valid:
        raise ExportError("Serialized config failed official validation: " + "; ".join(again.errors))
    sidecar = Path(str(output) + ".provenance.json")
    if dry_run:
        return {"config": body, "provenance": provenance, "output": str(output), "sidecar": str(sidecar), "dry_run": True}
    _atomic_text(output, body)
    _atomic_text(sidecar, json.dumps(provenance, indent=2, default=str) + "\n")
    return {"output": str(output), "sidecar": str(sidecar), "label": label, "reason": reason, "dry_run": False}


def _choose(rows, selection: Selection | None, *, force: bool) -> tuple[dict, str, str]:
    if selection and selection.selected_parameters:
        if force and selection.allow_force_tuned and selection.tuned_leader_parameters and selection.baseline_preferred:
            return (
                selection.tuned_leader_parameters,
                "force: exporting the tuned leader despite the held-out gate. " + selection.selected_reason,
                "tuned-leader",
            )
        if force and selection.allow_force_tuned and selection.tuned_leader_parameters and not selection.selected_parameters:
            return (
                selection.tuned_leader_parameters,
                "force: exporting the tuned leader. " + selection.selected_reason,
                "tuned-leader",
            )
        return selection.selected_parameters, selection.selected_reason, selection.selected_label or "selected"
    if selection and not selection.selected_parameters:
        if force and selection.allow_force_tuned and selection.tuned_leader_complete and selection.tuned_leader_parameters:
            return (
                selection.tuned_leader_parameters,
                "force: exporting the tuned leader. " + selection.selected_reason,
                "tuned-leader",
            )
        raise ExportError(selection.selected_reason + " --force does not apply to this result.")
    best = top_training_configs(rows, 1)
    if not best:
        raise ExportError(
            "No successful training evaluation to export. Run tune, then validate."
        )
    if not force:
        raise ExportError(
            "No held-out selection file. Run validate, or pass --force to export the "
            "best successful training config without a held-out comparison."
        )
    return best[0].parameters, "force: best successful training config, no held-out comparison", "training-best"


def _require_success(rows, parameters: dict) -> None:
    from tuner.config_generator import config_hash

    hashed = config_hash(normalize_options(parameters))
    matched = [row for row in rows if row.config_hash == hashed]
    if not any(row.status == "success" and row.score is not None for row in matched):
        raise ExportError(
            "Export refused: the selected config has no successful official score in this study."
        )


def _hash(parameters: dict) -> str:
    from tuner.config_generator import config_hash

    return config_hash(parameters)


def _refuse_active_miner_config(output: Path, repo_root: Path) -> None:
    active = (repo_root / "configs" / "gatk.conf").resolve()
    if output.resolve() == active:
        raise ExportError(
            "Refusing to overwrite the miner config configs/gatk.conf. "
            "Write to another path, re-score it with test, and copy it only after that."
        )


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=path.suffix or ".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise

"""Read and canonicalize configs/gatk.conf without rewriting it."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from templates.tool_params import GATK_QUALITY_PARAMS
from tuning.errors import TuningError
from utils.config_loader import _parse_value

FIXED_VALUES = {
    "emit_ref_confidence": "NONE",
    "sample_ploidy": 2,
    "contamination_fraction_to_filter": 0.0,
}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_baseline(path: Path) -> dict[str, Any]:
    """Parse the 25 quality keys. Duplicate keys are rejected."""
    if not path.is_file():
        raise TuningError(f"Baseline config not found: {path}")
    values: dict[str, Any] = {}
    seen: set[str] = set()
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if key in seen:
            raise TuningError(f"{path}:{line_number}: duplicate key {key}")
        seen.add(key)
        if key not in GATK_QUALITY_PARAMS:
            raise TuningError(f"{path}:{line_number}: {key} is not a GATK quality parameter")
        values[key] = _parse_value(raw_value)
    missing = [name for name in GATK_QUALITY_PARAMS if name not in values]
    if missing:
        raise TuningError(f"{path} is missing {', '.join(missing)}")
    extra = [name for name in values if name not in GATK_QUALITY_PARAMS]
    if extra:
        raise TuningError(f"{path} has unknown keys {', '.join(extra)}")
    return values


def format_value(name: str, value: Any) -> str:
    spec = GATK_QUALITY_PARAMS[name]
    kind = spec["type"]
    if kind == "bool":
        if not isinstance(value, bool):
            raise TuningError(f"{name} must be a boolean")
        return "true" if value else "false"
    if kind == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            raise TuningError(f"{name} must be an integer")
        return str(value)
    if kind == "enum":
        text = str(value)
        if text not in spec["allowed_values"]:
            raise TuningError(f"{name} value {text!r} is not allowed")
        return text
    if kind == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TuningError(f"{name} must be a float")
        number = float(value)
        if number != number or number in {float("inf"), float("-inf")}:
            raise TuningError(f"{name} is not finite")
        text = format(number, ".12g")
        if "." not in text and "e" not in text.lower():
            text += ".0"
        return text
    raise TuningError(f"{name} has unsupported type {kind}")


def canonical_text(values: dict[str, Any]) -> str:
    lines = [f"{name}={format_value(name, values[name])}" for name in GATK_QUALITY_PARAMS]
    return "\n".join(lines) + "\n"


def config_sha256(values: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_text(values).encode("utf-8")).hexdigest()


def write_conf(path: Path, values: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(canonical_text(values), encoding="utf-8")
    temporary.replace(path)


def type_counts(values: dict[str, Any]) -> dict[str, int]:
    counts = {"int": 0, "float": 0, "bool": 0, "enum": 0}
    for name in GATK_QUALITY_PARAMS:
        counts[GATK_QUALITY_PARAMS[name]["type"]] += 1
        _ = values[name]
    return counts

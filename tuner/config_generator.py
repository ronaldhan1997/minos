"""Parse and write GATK ``.conf`` files in the miner's key=value format.

Parsing uses ``utils.config_loader._parse_value`` so a value becomes the same
Python type the miner would submit. Serialization follows the whitelist order
in ``GATK_QUALITY_PARAMS`` and is the input to the config hash.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from templates.tool_params import GATK_QUALITY_PARAMS
from utils.config_loader import _parse_value


class ConfSyntaxError(ValueError):
    """The file is not a key=value config."""


def load_conf(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return parse_conf(path.read_text(encoding="utf-8-sig"), origin=str(path))


def parse_conf(text: str, *, origin: str = "<config>") -> dict[str, Any]:
    options: dict[str, Any] = {}
    for line_num, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ConfSyntaxError(f"{origin}:{line_num}: expected key=value, got {raw!r}")
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip()
        if not key:
            raise ConfSyntaxError(f"{origin}:{line_num}: empty key")
        if key in options:
            raise ConfSyntaxError(
                f"{origin}:{line_num}: duplicate key {key!r}. "
                "The miner keeps the last value; the tuner rejects duplicates "
                "so the hashed config has one value."
            )
        options[key] = _parse_value(val)
    return options


def format_value(key: str, value: Any) -> str:
    spec = GATK_QUALITY_PARAMS.get(key)
    if isinstance(value, bool):
        return "true" if value else "false"
    if spec and spec["type"] == "int":
        return str(int(value))
    if spec and spec["type"] == "float":
        return format(float(value), ".12g")
    if isinstance(value, float):
        return format(value, ".12g")
    if isinstance(value, int):
        return str(value)
    return str(value)


def canonical_conf(options: dict[str, Any]) -> str:
    """Deterministic body. Whitelist keys first, then any other keys."""
    lines: list[str] = []
    for key in GATK_QUALITY_PARAMS:
        if key in options:
            lines.append(f"{key}={format_value(key, options[key])}")
    for key in sorted(set(options) - set(GATK_QUALITY_PARAMS)):
        lines.append(f"{key}={format_value(key, options[key])}")
    return "\n".join(lines) + ("\n" if lines else "")


def normalize_options(options: dict[str, Any]) -> dict[str, Any]:
    """Round-trip through the canonical text so 30 and 30.0 hash together."""
    return parse_conf(canonical_conf(options), origin="<canonical>")


def config_hash(options: dict[str, Any]) -> str:
    body = canonical_conf(normalize_options(options))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def merge_options(baseline: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Keep every baseline value and replace only the trial keys."""
    merged = dict(baseline)
    merged.update(overrides)
    return normalize_options(merged)


def dump_conf(options: dict[str, Any], header_lines: list[str] | None = None) -> str:
    lines = []
    for line in header_lines or []:
        text = line if line.startswith("#") or line == "" else f"# {line}"
        lines.append(text)
    if lines and lines[-1] != "":
        lines.append("")
    body = canonical_conf(normalize_options(options))
    return "\n".join(lines) + ("\n" if lines else "") + body

"""Search space checked against ``templates.tool_params.GATK_QUALITY_PARAMS``.

A bound outside the official range is rejected here. The official validator
rejects the whole config for one bad value, and it does not clamp.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from templates.tool_params import GATK_QUALITY_PARAMS
from tuner.errors import TunerError


class ParamSpec(BaseModel):
    name: str
    type: str
    low: float | None = None
    high: float | None = None
    log: bool = False
    choices: list[Any] | None = None
    upstream_min: float | None = None
    upstream_max: float | None = None
    upstream_default: Any = None
    upstream_choices: list[Any] | None = None


class SearchSpace(BaseModel):
    version: str
    profile: str
    parameters: dict[str, ParamSpec] = Field(default_factory=dict)

    def names(self) -> list[str]:
        return list(self.parameters)


def load_search_space(path: Path, profile: str) -> SearchSpace:
    if profile not in {"core", "advanced", "full"}:
        raise TunerError(f"Unknown search-space profile {profile!r}")
    if not path.exists():
        raise TunerError(f"Search space file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "core" not in raw:
        raise TunerError(f"{path} must contain a 'core' mapping")
    core = _section(raw, "core", path)
    if profile == "core":
        merged = core["parameters"]
        version = core["version"]
    elif profile == "full":
        if "full" not in raw:
            raise TunerError(f"{path} has no 'full' section")
        full = _section(raw, "full", path)
        merged = full["parameters"]
        version = full["version"]
        expected = set(GATK_QUALITY_PARAMS) - {"emit_ref_confidence"}
        got = set(merged)
        if got != expected:
            missing = ", ".join(sorted(expected - got)) or "(none)"
            extra = ", ".join(sorted(got - expected)) or "(none)"
            raise TunerError(
                f"{path}: full must list every quality parameter except "
                f"emit_ref_confidence. Missing: {missing}. Extra: {extra}."
            )
    else:
        if "advanced" not in raw:
            raise TunerError(f"{path} has no 'advanced' section")
        advanced = _section(raw, "advanced", path)
        overlap = set(core["parameters"]) & set(advanced["parameters"])
        if overlap:
            raise TunerError(
                "advanced search space repeats core parameters: "
                + ", ".join(sorted(overlap))
            )
        merged = {**core["parameters"], **advanced["parameters"]}
        version = advanced["version"]
    parameters = {
        name: _checked_param(name, body)
        for name, body in merged.items()
    }
    return SearchSpace(version=str(version), profile=profile, parameters=parameters)


def _section(raw: dict, name: str, path: Path) -> dict:
    section = raw[name]
    if not isinstance(section, dict):
        raise TunerError(f"{path}: {name} must be a mapping")
    if "version" not in section or "parameters" not in section:
        raise TunerError(f"{path}: {name} needs version and parameters")
    if not isinstance(section["parameters"], dict) or not section["parameters"]:
        raise TunerError(f"{path}: {name}.parameters must be a non-empty mapping")
    return section


def _checked_param(name: str, body: dict) -> ParamSpec:
    if name not in GATK_QUALITY_PARAMS:
        raise TunerError(
            f"Parameter {name!r} is not in the GATK quality whitelist. "
            "Remove it from the search space."
        )
    if not isinstance(body, dict) or "type" not in body:
        raise TunerError(f"Search-space entry {name} needs a type")
    upstream = GATK_QUALITY_PARAMS[name]
    declared = body["type"]
    if declared != upstream["type"]:
        raise TunerError(
            f"{name}: search space type {declared} does not match upstream "
            f"type {upstream['type']}"
        )
    spec = ParamSpec(
        name=name,
        type=declared,
        upstream_default=upstream.get("default"),
        upstream_choices=list(upstream.get("allowed_values", [])) or None,
        upstream_min=upstream.get("min"),
        upstream_max=upstream.get("max"),
    )
    if declared in {"int", "float"}:
        if "low" not in body or "high" not in body:
            raise TunerError(f"{name} needs low and high")
        low = _number(name, body["low"], declared)
        high = _number(name, body["high"], declared)
        if low > high:
            raise TunerError(f"{name}: low {low} is above high {high}")
        umin = upstream["min"]
        umax = upstream["max"]
        if low < umin or high > umax:
            raise TunerError(
                f"{name}: search range [{low}, {high}] is outside the official "
                f"range [{umin}, {umax}]"
            )
        spec.low = low
        spec.high = high
        spec.log = bool(body.get("log", False))
        if spec.log and not (low > 0 and high > 0):
            raise TunerError(f"{name}: a log range needs both bounds above 0")
    elif declared == "enum":
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise TunerError(f"{name} needs a non-empty choices list")
        allowed = list(upstream["allowed_values"])
        unknown = [choice for choice in choices if choice not in allowed]
        if unknown:
            raise TunerError(
                f"{name}: {unknown} not in official choices {allowed}"
            )
        spec.choices = list(choices)
    elif declared == "bool":
        spec.choices = [False, True]
    else:
        raise TunerError(f"{name}: unsupported type {declared}")
    return spec


def _number(name: str, value: Any, kind: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TunerError(f"{name}: bound {value!r} is not a number")
    if kind == "int" and int(value) != value:
        raise TunerError(f"{name}: integer bound {value} is not a whole number")
    return float(value)


def baseline_point(space: SearchSpace, options: dict[str, Any]) -> dict[str, Any] | None:
    """Baseline values for the tuned keys, when every one sits inside the space.

    Returns None when a baseline value is outside the box. The full baseline
    config is still evaluated; it is just not enqueued as an Optuna trial.
    """
    point: dict[str, Any] = {}
    for name, spec in space.parameters.items():
        if name not in options:
            return None
        value = options[name]
        if spec.type == "int":
            if isinstance(value, bool) or not isinstance(value, int):
                return None
            if value < spec.low or value > spec.high:
                return None
            point[name] = int(value)
        elif spec.type == "float":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return None
            numeric = float(value)
            if numeric < spec.low or numeric > spec.high:
                return None
            point[name] = numeric
        elif spec.type == "enum":
            if value not in (spec.choices or []):
                return None
            point[name] = value
        elif spec.type == "bool":
            if not isinstance(value, bool):
                return None
            point[name] = bool(value)
        else:
            return None
    return point


def suggest(trial: Any, space: SearchSpace) -> dict[str, Any]:
    """Draw one point. Call this on every trial, including enqueued ones."""
    params: dict[str, Any] = {}
    for name, spec in space.parameters.items():
        if spec.type == "int":
            params[name] = int(trial.suggest_int(name, int(spec.low), int(spec.high)))
        elif spec.type == "float":
            params[name] = float(
                trial.suggest_float(name, float(spec.low), float(spec.high), log=spec.log)
            )
        elif spec.type == "enum":
            params[name] = trial.suggest_categorical(name, list(spec.choices or []))
        elif spec.type == "bool":
            params[name] = bool(trial.suggest_categorical(name, [False, True]))
        else:
            raise TunerError(f"Cannot suggest {name}")
    return params

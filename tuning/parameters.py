"""Versioned search bounds for phase 1 (8 parameters) and phase 2 (15).

Phase 1 tunes eight high-importance fields. Phase 2 keeps those eight and
opens seven more. Every other quality key stays at the baseline. Three of
those are fixed by the scoring setup: ``emit_ref_confidence=NONE``,
``sample_ploidy=2``, and ``contamination_fraction_to_filter=0.0``. Every
trial config still writes all 25 keys.

``SEARCH_SPACE_VERSION`` changes when a range changes. Old run manifests
refuse to resume across that boundary.
"""

from __future__ import annotations

from typing import Any

from templates.tool_params import GATK_QUALITY_PARAMS, validate_and_build_flags
from tuning.baseline import FIXED_VALUES
from tuning.errors import TuningError

# name -> (low, high, step or None, log)
_INT = "int"
_FLOAT = "float"

# Practical boxes inside the official ranges. Defaults from configs/gatk.conf sit inside each box.
IMPORTANT_SPEC: dict[str, dict[str, Any]] = {
    "min_base_quality_score": {"low": 5, "high": 30},
    "min_mapping_quality_score": {"low": 10, "high": 40},
    "base_quality_score_threshold": {"low": 10, "high": 30},
    "standard_min_confidence_threshold_for_calling": {"low": 10.0, "high": 50.0, "step": 1.0},
    "pcr_indel_model": {"choices": ["NONE", "HOSTILE", "AGGRESSIVE", "CONSERVATIVE"]},
    "min_pruning": {"low": 1, "high": 10},
    "active_probability_threshold": {"low": 5e-4, "high": 1e-2, "log": True},
    "max_reads_per_alignment_start": {"low": 25, "high": 200, "step": 5},
}

PHASE1_SPEC = IMPORTANT_SPEC
# Initial phase-2 hypotheses. Ranges sit inside the upstream GATK whitelist.
PHASE2_EXTRA: dict[str, dict[str, Any]] = {
    "min_dangling_branch_length": {"low": 2, "high": 10},
    "max_num_haplotypes_in_population": {"low": 64, "high": 256},
    "assembly_region_padding": {"low": 50, "high": 200},
    "pair_hmm_gap_continuation_penalty": {"low": 5, "high": 20},
    "phred_scaled_global_read_mismapping_rate": {"low": 30, "high": 60},
    "heterozygosity": {"low": 1e-4, "high": 1e-2, "log": True},
    "indel_heterozygosity": {"low": 1e-5, "high": 1e-3, "log": True},
}
PHASE2_SPEC = {**PHASE1_SPEC, **PHASE2_EXTRA}
SEARCH_SPACE_VERSION = "phase2-15-v1"

FIXED_KEYS = tuple(FIXED_VALUES)


def phase_spec(phase: int) -> dict[str, dict[str, Any]]:
    if phase not in (1, 2):
        raise TuningError(f"Phase must be 1 or 2, got {phase}")
    return dict(PHASE1_SPEC if phase == 1 else PHASE2_SPEC)


def tunable_names(phase: int) -> list[str]:
    return list(phase_spec(phase))


def fixed_names(phase: int) -> list[str]:
    tunable = set(tunable_names(phase))
    return [name for name in GATK_QUALITY_PARAMS if name not in tunable]


def assert_ranges_inside_upstream() -> None:
    for name, body in {**PHASE1_SPEC, **PHASE2_EXTRA}.items():
        upstream = GATK_QUALITY_PARAMS[name]
        if "choices" in body:
            unknown = [choice for choice in body["choices"] if choice not in upstream["allowed_values"]]
            if unknown:
                raise TuningError(f"{name}: {unknown} is outside the official enum")
            continue
        if upstream["type"] == "bool":
            continue
        low, high = body["low"], body["high"]
        if low < upstream["min"] or high > upstream["max"]:
            raise TuningError(
                f"{name}: phase range [{low}, {high}] is outside official "
                f"[{upstream['min']}, {upstream['max']}]"
            )


assert_ranges_inside_upstream()


def merge_candidate(baseline: dict[str, Any], suggested: dict[str, Any]) -> dict[str, Any]:
    """Full 25-key config. ``baseline`` is not mutated."""
    merged = dict(baseline)
    merged.update(suggested)
    return merged


def validate_candidate(values: dict[str, Any], phase: int) -> list[str]:
    """Official whitelist plus the phase's fixed fields and region-size rule."""
    errors: list[str] = []
    if set(values) != set(GATK_QUALITY_PARAMS):
        missing = sorted(set(GATK_QUALITY_PARAMS) - set(values))
        extra = sorted(set(values) - set(GATK_QUALITY_PARAMS))
        if missing:
            errors.append("missing " + ", ".join(missing))
        if extra:
            errors.append("unknown " + ", ".join(extra))
        return errors
    for name, expected in FIXED_VALUES.items():
        if values.get(name) != expected:
            errors.append(f"{name} must stay {expected!r}")
    official = validate_and_build_flags("gatk", values)
    if not official["valid"]:
        errors.append(str(official.get("error")))
    low = values.get("min_assembly_region_size")
    high = values.get("max_assembly_region_size")
    if isinstance(low, int) and isinstance(high, int) and low > high:
        errors.append(
            f"min_assembly_region_size {low} is greater than max_assembly_region_size {high}"
        )
    tunable = set(tunable_names(phase))
    for name, body in phase_spec(phase).items():
        if name not in tunable:
            continue
        value = values[name]
        if "low" in body and isinstance(value, (int, float)) and not isinstance(value, bool):
            if value < body["low"] or value > body["high"]:
                errors.append(f"{name}={value} is outside the phase {phase} range")
        if "choices" in body and value not in body["choices"]:
            errors.append(f"{name}={value!r} is outside the phase {phase} choices")
    return errors


def suggest(trial: Any, phase: int) -> dict[str, Any]:
    """Draw the phase's tunable keys. Call this on enqueued trials too."""
    drawn: dict[str, Any] = {}
    for name, body in phase_spec(phase).items():
        kind = GATK_QUALITY_PARAMS[name]["type"]
        if kind == "int":
            drawn[name] = int(trial.suggest_int(name, int(body["low"]), int(body["high"]), step=int(body.get("step", 1))))
        elif kind == "float":
            kwargs: dict[str, Any] = {}
            if body.get("step") is not None:
                kwargs["step"] = float(body["step"])
            if body.get("log"):
                kwargs["log"] = True
            drawn[name] = float(trial.suggest_float(name, float(body["low"]), float(body["high"]), **kwargs))
        elif kind == "enum":
            drawn[name] = trial.suggest_categorical(name, list(body["choices"]))
        elif kind == "bool":
            drawn[name] = bool(trial.suggest_categorical(name, [False, True]))
        else:
            raise TuningError(f"Cannot suggest {name}")
    return drawn


def enqueue_point(baseline: dict[str, Any], phase: int) -> dict[str, Any] | None:
    """Baseline values for the tunable keys when each one sits on the phase grid."""
    point: dict[str, Any] = {}
    for name, body in phase_spec(phase).items():
        value = baseline.get(name)
        kind = GATK_QUALITY_PARAMS[name]["type"]
        if kind == "int":
            if isinstance(value, bool) or not isinstance(value, int):
                return None
            if value < body["low"] or value > body["high"]:
                return None
            step = int(body.get("step", 1))
            if (value - int(body["low"])) % step != 0:
                return None
            point[name] = int(value)
        elif kind == "float":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return None
            number = float(value)
            if number < float(body["low"]) or number > float(body["high"]):
                return None
            point[name] = number
        elif kind == "enum":
            if value not in body["choices"]:
                return None
            point[name] = value
        elif kind == "bool":
            if not isinstance(value, bool):
                return None
            point[name] = value
        else:
            return None
    return point

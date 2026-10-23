"""Config IO, official validation, and search-space bounds."""

from __future__ import annotations

from pathlib import Path

import pytest

from templates.tool_params import validate_and_build_flags
from tuner.config_generator import (
    ConfSyntaxError,
    canonical_conf,
    config_hash,
    load_conf,
    merge_options,
    normalize_options,
    parse_conf,
)
from tuner.config_validator import validate_gatk_options
from tuner.parameters import baseline_point, load_search_space
from minos_layout import minos_root
from tuner.errors import TunerError

ROOT = Path(__file__).resolve().parents[2]
MINOS = minos_root()


def test_baseline_matches_shipped_gatk_and_passes_official_validator():
    baseline = normalize_options(load_conf(ROOT / "configs" / "baseline.conf"))
    shipped = normalize_options(load_conf(MINOS / "configs" / "gatk.conf"))
    assert config_hash(baseline) == config_hash(shipped)
    checked = validate_gatk_options(baseline)
    assert checked.valid, checked.errors
    again = validate_and_build_flags("gatk", baseline)
    assert again["valid"]


def test_hash_is_order_independent_and_integer_floats_match():
    left = {"min_base_quality_score": 10, "pcr_indel_model": "NONE"}
    right = {"pcr_indel_model": "NONE", "min_base_quality_score": 10}
    assert config_hash(left) == config_hash(right)
    as_float = {"standard_min_confidence_threshold_for_calling": 30.0}
    as_int = {"standard_min_confidence_threshold_for_calling": 30}
    assert config_hash(as_float) == config_hash(as_int)
    assert canonical_conf(as_float) == canonical_conf(as_int)


def test_duplicate_keys_are_rejected():
    with pytest.raises(ConfSyntaxError):
        parse_conf("min_pruning=2\nmin_pruning=3\n")


def test_official_validator_rejects_bad_values():
    unknown = validate_gatk_options({"not_a_gatk_param": 1})
    assert not unknown.valid
    out_of_range = validate_gatk_options({"min_base_quality_score": 51})
    assert not out_of_range.valid
    wrong_type = validate_gatk_options({"min_pruning": True})
    assert not wrong_type.valid
    bad_enum = validate_gatk_options({"pcr_indel_model": "SOMETIMES"})
    assert not bad_enum.valid
    nan = validate_gatk_options({"standard_min_confidence_threshold_for_calling": float("nan")})
    assert not nan.valid


def test_merge_keeps_untuned_baseline_values():
    baseline = normalize_options(load_conf(ROOT / "configs" / "baseline.conf"))
    merged = merge_options(baseline, {"min_pruning": 3, "pcr_indel_model": "NONE"})
    assert merged["min_pruning"] == 3
    assert merged["pcr_indel_model"] == "NONE"
    assert merged["sample_ploidy"] == baseline["sample_ploidy"]
    assert merged["min_base_quality_score"] == baseline["min_base_quality_score"]


def test_full_space_searches_every_quality_key_except_output_mode():
    from templates.tool_params import GATK_QUALITY_PARAMS

    space = load_search_space(ROOT / "configs" / "search_space.yaml", "full")
    assert space.version == "full-v1"
    assert set(space.parameters) == set(GATK_QUALITY_PARAMS) - {"emit_ref_confidence"}
    options = normalize_options(load_conf(ROOT / "configs" / "baseline.conf"))
    assert baseline_point(space, options) is not None
    assert "emit_ref_confidence" not in space.parameters


def test_core_space_accepts_baseline_and_rejects_widened_bounds(tmp_path):
    space = load_search_space(ROOT / "configs" / "search_space.yaml", "core")
    options = normalize_options(load_conf(ROOT / "configs" / "baseline.conf"))
    point = baseline_point(space, options)
    assert point is not None
    assert point["pcr_indel_model"] in {"NONE", "CONSERVATIVE"}
    advanced = load_search_space(ROOT / "configs" / "search_space.yaml", "advanced")
    assert "min_dangling_branch_length" in advanced.parameters
    assert "min_pruning" in advanced.parameters
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "core:\n  version: bad\n  parameters:\n"
        "    min_base_quality_score:\n      type: int\n      low: -1\n      high: 10\n",
        encoding="utf-8",
    )
    with pytest.raises(TunerError):
        load_search_space(bad, "core")

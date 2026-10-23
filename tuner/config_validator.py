"""Official GATK parameter check.

``validate_and_build_flags`` is the validator the template runs before it
builds a container command. One error rejects the whole config.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from templates.tool_params import validate_and_build_flags


class ValidationResult(BaseModel):
    valid: bool
    errors: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)


def validate_gatk_options(options: dict[str, Any]) -> ValidationResult:
    result = validate_and_build_flags("gatk", options)
    flags = [str(flag) for flag in result.get("flags", [])]
    return ValidationResult(
        valid=bool(result.get("valid")),
        errors=list(result.get("errors") or []),
        flags=flags,
    )

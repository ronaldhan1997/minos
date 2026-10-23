"""Run official ``templates.gatk.variant_call`` in an isolated directory.

The template already uses an argv list, a timeout, and container cleanup.
This module does not build a second GATK command line.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class GatkResult(BaseModel):
    success: bool
    variant_count: int = 0
    runtime_seconds: float | None = None
    error: str | None = None
    vcf_path: str | None = None
    log_path: str | None = None
    timed_out: bool = False


def interpret_gatk_result(
    result: dict[str, Any],
    vcf: Path,
    log_path: Path,
    elapsed: float,
) -> GatkResult:
    """Map the template's result dict onto a tuner status.

    The template reports a timeout as ``Timeout after {seconds}s`` and a
    nonzero container exit as ``rc=<code> ...``. Neither is a score.
    """
    if result.get("success") and vcf.exists():
        metadata = result.get("metadata") or {}
        runtime = metadata.get("runtime_seconds", elapsed)
        return GatkResult(
            success=True,
            variant_count=int(result.get("variant_count") or 0),
            runtime_seconds=float(runtime),
            vcf_path=str(vcf),
            log_path=str(log_path),
        )
    error = str(result.get("error") or "GATK failed")
    lowered = error.lower()
    timed_out = lowered.startswith("timeout") or "timeoutexpired" in lowered
    return GatkResult(
        success=False,
        error=error,
        timed_out=timed_out,
        runtime_seconds=elapsed,
        vcf_path=str(vcf) if vcf.exists() else None,
        log_path=str(log_path),
    )


def run_gatk(sample, options: dict[str, Any], work_dir: Path, settings) -> GatkResult:
    """Call the official HaplotypeCaller template once."""
    from templates.gatk import variant_call

    work_dir.mkdir(parents=True, exist_ok=True)
    vcf = work_dir / "output.vcf.gz"
    log_path = work_dir / "gatk.log"
    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.addHandler(handler)
    started = time.time()
    try:
        result = variant_call(
            bam_path=Path(sample.bam),
            reference_path=Path(sample.reference),
            output_vcf_path=vcf,
            region=sample.region,
            config={
                "gatk_options": options,
                "timeout": settings.timeout_seconds,
                "threads": settings.gatk_threads,
                "memory_gb": settings.memory_gb,
                "ref_build": "GRCh38",
            },
        )
    finally:
        root.removeHandler(handler)
        handler.close()
    interpreted = interpret_gatk_result(result, vcf, log_path, time.time() - started)
    logger.info(
        "gatk success=%s variants=%s seconds=%s sample=%s error=%s",
        interpreted.success,
        interpreted.variant_count,
        interpreted.runtime_seconds,
        sample.sample_id,
        interpreted.error,
    )
    return interpreted

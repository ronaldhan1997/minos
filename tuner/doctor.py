"""Report whether this machine can run a practice GATK evaluation."""

from __future__ import annotations

import subprocess
import sys
from typing import Callable

from pydantic import BaseModel, Field

from tuner.upstream import INSPECTED_COMMIT, bcftools_image, gatk_image, git_identity, happy_image, resolve_scoring_version, scorer_name_for


class Check(BaseModel):
    name: str
    ok: bool
    severity: str
    detail: str


class DoctorReport(BaseModel):
    ready: bool
    checks: list[Check] = Field(default_factory=list)
    scoring_version: str | None = None
    scorer_name: str | None = None
    commit: str | None = None
    dirty: bool | None = None


def probe_command(argv: list[str], timeout: int = 15) -> tuple[bool, str]:
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
    except FileNotFoundError:
        return False, f"{argv[0]} is not installed"
    except subprocess.TimeoutExpired:
        return False, f"{' '.join(argv)} timed out"
    except OSError as exc:
        return False, str(exc)
    text = ((proc.stdout or "") + (proc.stderr or "")).strip()
    if proc.returncode != 0:
        return False, text.splitlines()[-1] if text else f"exit {proc.returncode}"
    return True, text.splitlines()[0] if text else "ok"


def probe_docker() -> tuple[bool, str]:
    ok, detail = probe_command(["docker", "--version"])
    if not ok:
        return ok, detail
    running, info = probe_command(["docker", "info"], timeout=20)
    if not running:
        return False, "Docker is installed but the daemon is not responding. " + info
    return True, detail


def probe_image(image: str) -> tuple[bool, str]:
    ok, detail = probe_command(["docker", "image", "inspect", "--format", "{{.Id}}", image], timeout=30)
    if not ok:
        return False, f"{image} is not present locally. Pull it before tune or test. {detail}"
    return True, f"{image} -> {detail}"


def run_doctor(settings, *, probe_docker_fn: Callable | None = None, probe_image_fn: Callable | None = None) -> DoctorReport:
    docker_fn = probe_docker_fn or probe_docker
    image_fn = probe_image_fn or probe_image
    checks: list[Check] = []
    py = sys.version.split()[0]
    checks.append(
        Check(
            name="python",
            ok=sys.version_info >= (3, 11),
            severity="error",
            detail=f"{py}. The tuner is written for Python 3.11+.",
        )
    )
    docker_ok, docker_detail = docker_fn()
    checks.append(Check(name="docker", ok=docker_ok, severity="error", detail=docker_detail))
    try:
        image = gatk_image(settings.repo_root)
        present, detail = image_fn(image) if docker_ok else (False, "skipped because Docker is unavailable")
        checks.append(Check(name="gatk_image", ok=present, severity="error", detail=detail))
    except Exception as exc:  # noqa: BLE001
        checks.append(Check(name="gatk_image", ok=False, severity="error", detail=str(exc)))
    for name, loader in (("happy_image", happy_image), ("bcftools_image", bcftools_image)):
        try:
            ref = loader()
            present, detail = image_fn(ref) if docker_ok else (False, "skipped because Docker is unavailable")
            checks.append(Check(name=name, ok=present, severity="error", detail=detail))
        except Exception as exc:  # noqa: BLE001
            checks.append(Check(name=name, ok=False, severity="error", detail=str(exc)))
    try:
        from templates.tool_params import validate_and_build_flags
        from utils.scoring import AdvancedScorer, HappyScorer

        checks.append(
            Check(
                name="official_scorer",
                ok=callable(getattr(AdvancedScorer, "compute_advanced_score", None))
                and callable(getattr(AdvancedScorer, "compute_score_v2", None))
                and HappyScorer is not None
                and validate_and_build_flags is not None,
                severity="error",
                detail="Imported HappyScorer, AdvancedScorer, and validate_and_build_flags.",
            )
        )
    except Exception as exc:  # noqa: BLE001
        checks.append(Check(name="official_scorer", ok=False, severity="error", detail=str(exc)))
    try:
        import pysam  # noqa: F401

        checks.append(
            Check(
                name="pysam",
                ok=True,
                severity="error",
                detail=f"pysam {pysam.__version__}. Scoring v2 parses the hap.py VCF with this module.",
            )
        )
    except Exception as exc:  # noqa: BLE001
        checks.append(
            Check(
                name="pysam",
                ok=False,
                severity="error",
                detail=f"pysam is required to parse the hap.py VCF for scoring v2: {exc}",
            )
        )
    try:
        import optuna  # noqa: F401

        checks.append(Check(name="optuna", ok=True, severity="error", detail=f"optuna {optuna.__version__}"))
    except Exception as exc:  # noqa: BLE001
        checks.append(Check(name="optuna", ok=False, severity="error", detail=str(exc)))
    commit, dirty = git_identity(settings.repo_root)
    if commit == "unknown":
        checks.append(Check(name="minos_commit", ok=False, severity="warning", detail="git commit could not be read"))
    elif commit != INSPECTED_COMMIT:
        checks.append(
            Check(
                name="minos_commit",
                ok=True,
                severity="warning",
                detail=f"HEAD {commit} differs from the inspected commit {INSPECTED_COMMIT}",
            )
        )
    else:
        checks.append(Check(name="minos_commit", ok=True, severity="info", detail=commit))
    checks.append(
        Check(
            name="working_tree",
            ok=not dirty,
            severity="warning" if dirty else "info",
            detail="dirty" if dirty else "clean",
        )
    )
    version = None
    scorer = None
    try:
        version = resolve_scoring_version(settings.scoring_version)
        scorer = scorer_name_for(version)
        checks.append(
            Check(
                name="scoring_version",
                ok=True,
                severity="info",
                detail=(
                    f"{version} ({scorer}). Resolved from the last recorded network version, "
                    "or v1 if this machine has never recorded one. v1 and v2 are different scales."
                ),
            )
        )
    except Exception as exc:  # noqa: BLE001
        checks.append(Check(name="scoring_version", ok=False, severity="error", detail=str(exc)))
    _reference_checks(settings, checks)
    ready = all(item.ok for item in checks if item.severity == "error")
    return DoctorReport(ready=ready, checks=checks, scoring_version=version, scorer_name=scorer, commit=commit, dirty=dirty)


def _reference_checks(settings, checks: list[Check]) -> None:
    present = []
    missing = []
    for chrom in ("chr18", "chr19", "chr20", "chr21", "chr22"):
        base = settings.reference_dir() / chrom
        fasta = base / f"{chrom}.fa"
        fai = base / f"{chrom}.fa.fai"
        sdf = base / f"{chrom}.sdf"
        if fasta.exists() and fai.exists() and sdf.is_dir():
            present.append(chrom)
        else:
            missing.append(chrom)
    checks.append(
        Check(
            name="practice_references",
            ok=not missing,
            severity="warning" if missing else "info",
            detail=f"ready {present or 'none'}; missing fasta/fai/sdf for {missing or 'none'}",
        )
    )

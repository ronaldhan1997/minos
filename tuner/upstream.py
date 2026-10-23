"""Pinned Minos identity: commit, container images, and scoring version.

Scores are only comparable when the commit, image, and scoring version match.
The version string comes from ``utils.scoring_version``. The numeric score
comes from ``utils.scoring.AdvancedScorer``.
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
from pathlib import Path

from pydantic import BaseModel, Field

from tuner.errors import TunerError

# Commit inspected while this tuner was written. doctor warns when HEAD differs.
INSPECTED_COMMIT = "4b3c0944b517d09be9b54eb8fa3158ae700b5d5f"

_GATK_IMAGE_RE = re.compile(r'"(broadinstitute/gatk:[^"]+)"')


class Provenance(BaseModel):
    minos_commit: str
    minos_dirty: bool
    gatk_image: str
    docker_image_id: str | None = None
    happy_image: str
    bcftools_image: str
    scoring_version: str
    scorer_name: str
    hardware: dict = Field(default_factory=dict)
    gatk_threads: int
    memory_gb: int


def git_identity(repo_root: Path) -> tuple[str, bool]:
    """Return ``(commit, dirty)``. A missing git directory is ``("unknown", True)``."""
    try:
        sha = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            timeout=15,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            text=True,
            timeout=15,
            stderr=subprocess.DEVNULL,
        )
        return sha, bool(status.strip())
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return "unknown", True


def gatk_image(repo_root: Path) -> str:
    """Image string literal in ``templates/gatk.py``.

    The template hardcodes the image inside the docker argv. Reading it from
    that file keeps the tuner on the same image the official caller uses.
    """
    path = repo_root / "templates" / "gatk.py"
    text = path.read_text(encoding="utf-8")
    match = _GATK_IMAGE_RE.search(text)
    if not match:
        raise TunerError(f"Could not find the GATK image literal in {path}")
    return match.group(1)


def happy_image() -> str:
    from utils.scoring import HAPPY_DOCKER_IMAGE

    return HAPPY_DOCKER_IMAGE


def bcftools_image() -> str:
    from utils.scoring import BCFTOOLS_DOCKER_IMAGE

    return BCFTOOLS_DOCKER_IMAGE


def docker_image_id(image: str) -> str | None:
    """Local image id, or None when docker cannot see the image."""
    try:
        proc = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", image],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None
    if proc.returncode != 0:
        return None
    value = (proc.stdout or "").strip()
    return value or None


def hardware_summary() -> dict:
    memory_gb = None
    try:
        memory_gb = round(
            os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / (1024**3),
            2,
        )
    except (ValueError, OSError, AttributeError):
        memory_gb = None
    return {
        "cpu_count": os.cpu_count(),
        "memory_gb": memory_gb,
        "platform": platform.platform(),
        "python": platform.python_version(),
    }


def resolve_scoring_version(explicit: str | None = None) -> str:
    """Official scoring version.

    An explicit ``v1`` or ``v2`` is used as given. Otherwise the tuner follows
    ``utils.scoring_version.resolve`` with no network fetch: the last version
    this machine recorded, or v1 when none was recorded. v1 and v2 are
    different scales.
    """
    from utils import scoring_version as scoring_version_mod

    if explicit is not None:
        normalised = scoring_version_mod.normalise(explicit)
        if normalised is None:
            raise TunerError(
                f"Unsupported scoring version {explicit!r}. Use v1 or v2."
            )
        return normalised
    return scoring_version_mod.resolve(None)


def scorer_name_for(version: str) -> str:
    from utils.scoring_version import scorer_name

    return scorer_name(version)


def collect_provenance(settings, *, scoring_version: str | None = None) -> Provenance:
    commit, dirty = git_identity(settings.repo_root)
    version = scoring_version or resolve_scoring_version(settings.scoring_version)
    image = gatk_image(settings.repo_root)
    return Provenance(
        minos_commit=commit,
        minos_dirty=dirty,
        gatk_image=image,
        docker_image_id=docker_image_id(image),
        happy_image=happy_image(),
        bcftools_image=bcftools_image(),
        scoring_version=version,
        scorer_name=scorer_name_for(version),
        hardware=hardware_summary(),
        gatk_threads=settings.gatk_threads,
        memory_gb=settings.memory_gb,
    )

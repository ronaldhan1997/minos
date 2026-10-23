"""Paths and runtime limits for a tuner process."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel

from minos_layout import minos_root, project_root
from tuner.errors import TunerError


def _repo_root() -> Path:
    """Minos checkout: git identity, templates, datasets, and configs/gatk.conf."""
    return minos_root()


class Settings(BaseModel):
    """Resolved paths and GATK resource limits.

    ``workers`` is the number of simultaneous GATK jobs. It defaults to 1.
    ``gatk_threads`` is the thread count inside one container. It also defaults
    to 1 so a repeated run has a better chance of matching the previous VCF.
    """

    repo_root: Path
    experiments_dir: Path
    datasets_dir: Path
    baseline_config: Path
    search_space_path: Path
    sample_id_file: Path | None = None
    gatk_threads: int = 1
    memory_gb: int = 4
    timeout_seconds: int = 3600
    workers: int = 1
    seed: int = 17
    space_profile: str = "core"
    scoring_version: str | None = None
    allow_oversubscribe: bool = False

    def practice_dir(self) -> Path:
        return self.datasets_dir / "practice"

    def catalog_path(self) -> Path:
        return self.practice_dir() / "catalog.json"

    def reference_dir(self) -> Path:
        return self.datasets_dir / "reference"

    def study_dir(self, name: str, *, dry_run: bool = False) -> Path:
        root = self.experiments_dir / name
        if dry_run:
            return root / "dry-run"
        return root


class StudyPaths(BaseModel):
    root: Path
    optuna_db: Path
    evaluations_db: Path
    meta: Path
    selection: Path
    work: Path
    log: Path

    @classmethod
    def for_study(cls, settings: Settings, name: str, *, dry_run: bool = False) -> "StudyPaths":
        root = settings.study_dir(name, dry_run=dry_run)
        return cls(
            root=root,
            optuna_db=root / "optuna.sqlite3",
            evaluations_db=root / "evaluations.sqlite3",
            meta=root / "study.json",
            selection=root / "selection.json",
            work=root / "work",
            log=root / "tuner.log",
        )


def load_settings(
    *,
    experiments: Path | None = None,
    data_dir: Path | None = None,
    baseline: Path | None = None,
    search_space: Path | None = None,
    sample_id_file: Path | None = None,
    threads: int | None = None,
    memory_gb: int | None = None,
    timeout_seconds: int | None = None,
    workers: int | None = None,
    seed: int | None = None,
    space_profile: str | None = None,
    scoring_version: str | None = None,
    allow_oversubscribe: bool = False,
) -> Settings:
    root = _repo_root()
    home = project_root()
    exp = experiments or Path(os.environ.get("MINOS_TUNER_EXPERIMENTS", home / "experiments"))
    data = data_dir or Path(os.environ.get("MINOS_TUNER_DATA", root / "datasets"))
    base = baseline or home / "configs" / "baseline.conf"
    space = search_space or home / "configs" / "search_space.yaml"
    if sample_id_file is None:
        default_ids = home / "configs" / "sample_id.md"
        sample_id_file = default_ids if default_ids.exists() else None
    settings = Settings(
        repo_root=root,
        experiments_dir=Path(exp).resolve(),
        datasets_dir=Path(data).resolve(),
        baseline_config=Path(base).resolve(),
        search_space_path=Path(space).resolve(),
        sample_id_file=Path(sample_id_file).resolve() if sample_id_file else None,
        gatk_threads=threads if threads is not None else 1,
        memory_gb=memory_gb if memory_gb is not None else 4,
        timeout_seconds=timeout_seconds if timeout_seconds is not None else 3600,
        workers=workers if workers is not None else 1,
        seed=seed if seed is not None else 17,
        space_profile=space_profile or "core",
        scoring_version=scoring_version,
        allow_oversubscribe=allow_oversubscribe,
    )
    _check_limits(settings)
    return settings


def _check_limits(settings: Settings) -> None:
    if settings.gatk_threads < 1:
        raise TunerError("GATK threads must be >= 1")
    if settings.memory_gb < 1:
        raise TunerError("memory_gb must be >= 1")
    if settings.timeout_seconds < 1:
        raise TunerError("timeout must be >= 1 second")
    if settings.workers < 1:
        raise TunerError("workers must be >= 1")
    if settings.space_profile not in {"core", "advanced", "full"}:
        raise TunerError("search-space profile must be 'core', 'advanced', or 'full'")

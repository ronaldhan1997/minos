"""Command line for the local GATK practice tuner.

The commands are ``doctor``, ``prepare``, ``test``, ``tune``, ``validate``,
``report``, and ``export``. None of them submit a live round.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

import typer

from tuner import __version__
from tuner.catalog import assert_catalog_coverage, assert_selected_ids
from tuner.config_generator import load_conf, normalize_options
from tuner.config_validator import validate_gatk_options
from tuner.datasets import (
    apply_region_override,
    classify_tokens,
    discover_samples,
    prepare_datasets,
    read_sample_ids,
    resolve_sample,
)
from tuner.doctor import run_doctor
from tuner.errors import ExportError, PrepareError, StudyConflict, TunerError
from tuner.exporter import export_config
from tuner.parameters import load_search_space
from tuner.reporting import write_report
from tuner.settings import StudyPaths, load_settings
from tuner.validation import (
    HEURISTIC,
    baseline_hash_from_file,
    build_selection,
    ensure_split,
    load_meta,
    top_training_configs,
)

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Local Minos GATK HaplotypeCaller tuner.")
_STUDY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
logger = logging.getLogger("tuner")


def _setup_logging(verbose: bool, log_path: Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def _csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _check_study(name: str) -> str:
    if not _STUDY.match(name):
        raise TunerError(
            f"Study name {name!r} must be letters, numbers, dots, underscores, or hyphens."
        )
    return name


def _settings_from(
    ctx: typer.Context,
    *,
    threads: int | None = None,
    memory_gb: int | None = None,
    timeout_seconds: int | None = None,
    workers: int | None = None,
    seed: int | None = None,
    space_profile: str | None = None,
    scoring_version: str | None = None,
    allow_oversubscribe: bool = False,
    baseline: Path | None = None,
):
    obj = ctx.obj or {}
    return load_settings(
        experiments=obj.get("experiments"),
        data_dir=obj.get("data_dir"),
        baseline=baseline,
        threads=threads,
        memory_gb=memory_gb,
        timeout_seconds=timeout_seconds,
        workers=workers,
        seed=seed,
        space_profile=space_profile,
        scoring_version=scoring_version,
        allow_oversubscribe=allow_oversubscribe,
    )


@app.callback()
def main(
    ctx: typer.Context,
    verbose: bool = typer.Option(False, "--verbose", help="Debug logging."),
    experiments: Optional[Path] = typer.Option(None, "--experiments", help="Directory for studies and reports."),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", help="Root that contains practice/ and reference/."),
) -> None:
    """Tune GATK against local Minos practice data. No wallet and no live submission."""
    ctx.obj = {"verbose": verbose, "experiments": experiments, "data_dir": data_dir}
    _setup_logging(verbose)


@app.command()
def doctor(ctx: typer.Context, json_output: bool = typer.Option(False, "--json")) -> None:
    """Show Docker, GATK, hap.py, the Minos commit, and the scoring version."""
    settings = _settings_from(ctx)
    report = run_doctor(settings)
    if json_output:
        typer.echo(json.dumps(report.model_dump(), indent=2))
    else:
        typer.echo(f"minos-gatk-tuner {__version__}")
        for check in report.checks:
            mark = "ok" if check.ok else check.severity
            typer.echo(f"[{mark}] {check.name}: {check.detail}")
        typer.echo(f"ready: {report.ready}")
    if not report.ready:
        raise typer.Exit(code=1)


@app.command()
def prepare(
    ctx: typer.Context,
    samples: Optional[str] = typer.Option(
        None,
        "--samples",
        help="Comma-separated chromosome filters (chr20) and/or practice sample UUIDs.",
    ),
    manifest: Optional[Path] = typer.Option(None, "--manifest", help="JSON file with sample_id and region."),
    dry_run: bool = typer.Option(False, "--dry-run"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Index local practice BAMs, or download a missing UUID when PLATFORM_URL is set."""
    settings = _settings_from(ctx)
    try:
        result = prepare_datasets(settings, _csv(samples), dry_run=dry_run, manifest_path=manifest)
    except PrepareError as exc:
        typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    payload = result.model_dump()
    if json_output:
        typer.echo(json.dumps(payload, indent=2, default=str))
    else:
        for message in result.messages:
            typer.echo(message)
        if not result.samples:
            typer.echo("No practice samples matched.")
        for sample in result.samples:
            state = "ready" if sample.ready else "blocked"
            typer.echo(
                f"{state}  {sample.sample_id}  {sample.chromosome or '-'}  {sample.region or '-'}  "
                f"region_source={sample.region_source or '-'}"
            )
            for blocker in sample.blockers:
                typer.echo(f"       {blocker}")
    requested = bool(samples)
    blocked = any(not sample.ready for sample in result.samples)
    if requested and (not result.samples or blocked):
        raise typer.Exit(code=1)
    if not requested and result.ready_count == 0:
        raise typer.Exit(code=1)


@app.command("test")
def test_config(
    ctx: typer.Context,
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
    sample: str = typer.Option(..., "--sample", help="Practice sample UUID."),
    region: Optional[str] = typer.Option(None, "--region", help="Override the region when none was recorded."),
    dry_run: bool = typer.Option(False, "--dry-run"),
    force_rerun: bool = typer.Option(False, "--force-rerun"),
    json_output: bool = typer.Option(False, "--json"),
    threads: int = typer.Option(1, "--threads"),
    memory_gb: int = typer.Option(4, "--memory-gb"),
    timeout_seconds: int = typer.Option(3600, "--timeout"),
    scoring_version: Optional[str] = typer.Option(None, "--scoring-version"),
) -> None:
    """Run one config through official GATK and AdvancedScorer."""
    settings = _settings_from(
        ctx,
        threads=threads,
        memory_gb=memory_gb,
        timeout_seconds=timeout_seconds,
        scoring_version=scoring_version,
        baseline=config,
    )
    try:
        staged = resolve_sample(settings, sample)
        if region:
            staged = apply_region_override(settings, staged, region)
        if not staged.ready:
            raise PrepareError("; ".join(staged.blockers))
        options = normalize_options(load_conf(config))
        checked = validate_gatk_options(options)
        if not checked.valid:
            raise TunerError("invalid GATK parameters: " + "; ".join(checked.errors))
        evaluation = _run_one(
            settings,
            study_name="manual",
            sample=staged,
            options=options,
            role="test",
            trial_number=0,
            dry_run=dry_run,
            force_rerun=force_rerun,
        )
    except (PrepareError, TunerError, StudyConflict) as exc:
        typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    if json_output:
        typer.echo(evaluation.model_dump_json(indent=2))
    else:
        typer.echo(
            f"status={evaluation.status} score={evaluation.score} "
            f"combined_final={evaluation.combined_final} "
            f"scoring_version={evaluation.scoring_version} scorer={evaluation.scorer_name} "
            f"commit={evaluation.minos_commit} dirty={evaluation.minos_dirty}"
        )
        if evaluation.failure_reason:
            typer.echo(evaluation.failure_reason)
    if evaluation.status == "dry_run":
        raise typer.Exit(code=0)
    if evaluation.status != "success":
        raise typer.Exit(code=1)


@app.command()
def tune(
    ctx: typer.Context,
    trials: int = typer.Option(..., "--trials", min=1),
    workers: int = typer.Option(1, "--workers"),
    study: str = typer.Option(..., "--study"),
    samples: Optional[str] = typer.Option(
        None,
        "--samples",
        help="Sample UUIDs to tune, or chromosome filters. Explicit UUIDs are the whole study.",
    ),
    space: str = typer.Option(
        "full",
        "--space",
        help="full searches every quality key except emit_ref_confidence. core is 6 keys. advanced adds 3.",
    ),
    seed: int = typer.Option(17, "--seed"),
    threads: int = typer.Option(1, "--threads"),
    memory_gb: int = typer.Option(4, "--memory-gb"),
    timeout_seconds: int = typer.Option(3600, "--timeout"),
    scoring_version: Optional[str] = typer.Option(None, "--scoring-version"),
    allow_oversubscribe: bool = typer.Option(False, "--allow-oversubscribe"),
    allow_partial_catalog: bool = typer.Option(
        False,
        "--allow-partial-catalog",
        help="Search the runnable staged samples even when configs/sample_id.md lists more ids.",
    ),
    dry_run: bool = typer.Option(False, "--dry-run"),
    config: Optional[Path] = typer.Option(None, "--config", help="Baseline conf. Defaults to configs/baseline.conf."),
) -> None:
    """Search GATK parameters. The objective is the official score on training samples."""
    settings = _settings_from(
        ctx,
        threads=threads,
        memory_gb=memory_gb,
        timeout_seconds=timeout_seconds,
        workers=workers,
        seed=seed,
        space_profile=space,
        scoring_version=scoring_version,
        allow_oversubscribe=allow_oversubscribe,
        baseline=config,
    )
    try:
        _check_study(study)
        paths = StudyPaths.for_study(settings, study, dry_run=False)
        if ctx.obj.get("verbose"):
            _setup_logging(True, paths.log)
        ready = [item for item in discover_samples(settings, hash_files=not dry_run) if item.ready]
        tokens = _csv(samples)
        selected_ids: set[str] = set()
        chromosome_filter = False
        if tokens:
            chromosomes, sample_ids, rejected = classify_tokens(tokens)
            if rejected:
                raise TunerError(
                    "Not a chromosome filter and not a sample UUID: " + ", ".join(rejected)
                )
            selected_ids = set(sample_ids)
            chromosome_filter = bool(chromosomes)
            ready = [
                item
                for item in ready
                if item.sample_id.lower() in sample_ids or item.chromosome in chromosomes
            ]
        ready_ids = [item.sample_id for item in ready]
        if selected_ids and not chromosome_filter:
            coverage = assert_selected_ids(sorted(selected_ids), ready_ids)
        else:
            coverage = assert_catalog_coverage(
                read_sample_ids(settings.sample_id_file),
                ready_ids,
                allow_partial=allow_partial_catalog,
            )
        from tuner.upstream import collect_provenance

        provenance = collect_provenance(settings)
        search = load_search_space(settings.search_space_path, settings.space_profile)
        meta = ensure_split(
            meta_path=paths.meta,
            study_name=study,
            ready_ids=[item.sample_id for item in ready],
            space=search,
            settings=settings,
            scoring_version=provenance.scoring_version,
            commit=provenance.minos_commit,
            chromosomes={item.sample_id: item.chromosome for item in ready if item.chromosome},
        )
        by_id = {item.sample_id: item for item in discover_samples(settings, hash_files=not dry_run)}
        missing = [sample_id for sample_id in meta.train_sample_ids if sample_id not in by_id or not by_id[sample_id].ready]
        if missing:
            raise TunerError(
                "Training sample(s) are not runnable: " + ", ".join(missing)
                + ". Fix them with prepare, or use a new --study name."
            )
        train = [by_id[sample_id] for sample_id in meta.train_sample_ids]
        baseline = normalize_options(load_conf(settings.baseline_config))
        checked = validate_gatk_options(baseline)
        if not checked.valid:
            raise TunerError("baseline config is invalid: " + "; ".join(checked.errors))
        typer.echo(coverage)
        typer.echo(meta.split_description)
        typer.echo(f"train: {', '.join(meta.train_sample_ids)}")
        typer.echo(f"held-out (not used by Optuna): {', '.join(meta.val_sample_ids) or '(none)'}")
        typer.echo(
            f"workers={settings.workers} gatk_threads={settings.gatk_threads} "
            f"memory_gb={settings.memory_gb} estimated_ram_gb={settings.workers * settings.memory_gb} "
            f"scoring_version={provenance.scoring_version} scorer={provenance.scorer_name} "
            f"commit={provenance.minos_commit} dirty={provenance.minos_dirty}"
        )
        from tuner.optimizer import run_study

        result = run_study(
            settings=settings,
            study_name=study,
            n_trials=trials,
            space=search,
            train_samples=train,
            baseline_options=baseline,
            provenance=provenance,
            dry_run=dry_run,
        )
    except (TunerError, StudyConflict, PrepareError, FileNotFoundError, ValueError) as exc:
        typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(
        f"best_adv_score={result.get('best_adv_score')} "
        f"baseline_adv_score={result.get('baseline_adv_score')} "
        f"(leaderboard ADV. SCORE = official score / 100; 0.85 is an official mean of 85)"
    )
    capped = result.get("penalty_capped_samples") or []
    if capped:
        typer.echo(
            "baseline training samples at the 45-point overcall cap: " + ", ".join(capped)
        )
    typer.echo(json.dumps(result, default=str))


@app.command()
def validate(
    ctx: typer.Context,
    study: str = typer.Option(..., "--study"),
    top: int = typer.Option(5, "--top", min=1),
    samples: Optional[str] = typer.Option(
        None,
        "--samples",
        help="Held-out sample UUIDs. Defaults to the split stored by tune.",
    ),
    dry_run: bool = typer.Option(False, "--dry-run"),
    threads: int = typer.Option(1, "--threads"),
    memory_gb: int = typer.Option(4, "--memory-gb"),
    timeout_seconds: int = typer.Option(3600, "--timeout"),
    scoring_version: Optional[str] = typer.Option(None, "--scoring-version"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Score the baseline and the top training configs on held-out samples."""
    settings = _settings_from(
        ctx,
        threads=threads,
        memory_gb=memory_gb,
        timeout_seconds=timeout_seconds,
        scoring_version=scoring_version,
    )
    try:
        selection = _run_validate(settings, _check_study(study), top, _csv(samples), dry_run)
    except (TunerError, StudyConflict, PrepareError, FileNotFoundError) as exc:
        typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    if json_output:
        typer.echo(selection.model_dump_json(indent=2))
    else:
        typer.echo(HEURISTIC)
        typer.echo(f"selected: {selection.selected_label} {selection.selected_config_hash}")
        typer.echo(selection.selected_reason)
        for row in selection.leaderboard:
            typer.echo(
                f"{row['label']} complete={row['complete']} mean={row['mean_score']} "
                f"min={row['min_score']} std={row['std']} success_rate={row['success_rate']} "
                f"robustness={row['robustness']} per_sample={row['per_sample']}"
            )
    if dry_run:
        raise typer.Exit(code=0)
    if selection.selected_config_hash is None:
        raise typer.Exit(code=1)


@app.command()
def report(
    ctx: typer.Context,
    study: str = typer.Option(..., "--study"),
    output: Path = typer.Option(..., "--output"),
) -> None:
    """Write a CSV trial table and a JSON summary beside it."""
    settings = _settings_from(ctx)
    try:
        summary = write_report(settings, _check_study(study), output)
    except (FileNotFoundError, TunerError) as exc:
        typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(f"csv: {summary['csv']}")
    typer.echo(f"json: {summary['json']}")
    if summary.get("warning"):
        typer.echo(f"WARNING: {summary['warning']}")


@app.command()
def export(
    ctx: typer.Context,
    study: str = typer.Option(..., "--study"),
    output: Path = typer.Option(..., "--output"),
    force: bool = typer.Option(
        False,
        "--force",
        help="Export without held-out evidence, or export the tuned leader when the gate kept baseline. Does not skip parameter validation.",
    ),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Write a validated gatk.conf and a provenance sidecar."""
    settings = _settings_from(ctx)
    try:
        result = export_config(settings, _check_study(study), output, force=force, dry_run=dry_run)
    except (ExportError, TunerError, FileNotFoundError) as exc:
        typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(json.dumps({key: result[key] for key in result if key != "config"}, indent=2, default=str))


def _run_one(settings, *, study_name, sample, options, role, trial_number, dry_run, force_rerun):
    from tuner.objective import evaluate_candidate
    from tuner.storage import Store
    from tuner.upstream import collect_provenance

    paths = StudyPaths.for_study(settings, study_name, dry_run=dry_run)
    paths.root.mkdir(parents=True, exist_ok=True)
    provenance = collect_provenance(settings)
    store = Store(paths.evaluations_db)
    try:
        versions = store.scoring_versions(study_name)
        others = versions - {provenance.scoring_version}
        if others:
            raise StudyConflict(
                f"{study_name} already used scoring version(s) {sorted(others)}; "
                f"this run resolved {provenance.scoring_version}."
            )
        return evaluate_candidate(
            store=store,
            study=study_name,
            sample=sample,
            options=options,
            settings=settings,
            provenance=provenance,
            role=role,
            trial_number=trial_number,
            dry_run=dry_run,
            force_rerun=force_rerun,
        )
    finally:
        store.close()


def _run_validate(settings, study_name: str, top: int, sample_tokens: list[str], dry_run: bool):
    from tuner.objective import evaluate_candidate
    from tuner.storage import Store
    from tuner.upstream import collect_provenance

    paths = StudyPaths.for_study(settings, study_name, dry_run=False)
    meta = load_meta(paths.meta)
    if meta is None:
        raise TunerError(f"Study {study_name} has no split. Run tune before validate.")
    if not paths.evaluations_db.exists() and not dry_run:
        raise TunerError(f"Study {study_name} has no evaluations.")
    provenance = collect_provenance(settings)
    if provenance.scoring_version != meta.scoring_version:
        raise StudyConflict(
            f"Study is locked to {meta.scoring_version}; this process resolved {provenance.scoring_version}."
        )
    explicit = [token.lower() for token in sample_tokens]
    chromosomes, sample_ids, rejected = classify_tokens(explicit) if explicit else (set(), set(), [])
    if rejected or chromosomes:
        raise TunerError(
            "validate --samples takes practice sample UUIDs. Chromosome names are filters for prepare and tune."
        )
    train = set(meta.train_sample_ids)
    if sample_ids & train:
        raise TunerError(
            "These samples were used as Optuna training data and cannot be the held-out set: "
            + ", ".join(sorted(sample_ids & train))
        )
    expected = sorted(sample_ids) if sample_ids else list(meta.val_sample_ids)
    if not expected:
        raise TunerError("The study has no held-out samples. Tune with at least 2 ready samples.")
    by_id = {item.sample_id: item for item in discover_samples(settings, hash_files=not dry_run)}
    staged = []
    for sample_id in expected:
        sample = by_id.get(sample_id)
        if sample is None or not sample.ready:
            raise PrepareError(f"Held-out sample {sample_id} is not runnable.")
        staged.append(sample)
    baseline, baseline_hash = baseline_hash_from_file(settings.baseline_config)
    eval_paths = StudyPaths.for_study(settings, study_name, dry_run=dry_run)
    eval_paths.root.mkdir(parents=True, exist_ok=True)
    db = eval_paths.evaluations_db if dry_run else paths.evaluations_db
    store = Store(db)
    try:
        if not dry_run:
            rows = store.list_study(study_name)
            chosen = top_training_configs(rows, top)
        else:
            chosen = []
        jobs = [(-1, baseline)]
        for row in chosen:
            if row.config_hash == baseline_hash:
                continue
            jobs.append((row.trial_number, row.parameters))
        for trial_number, options in jobs:
            for sample in staged:
                evaluate_candidate(
                    store=store,
                    study=study_name,
                    sample=sample,
                    options=options,
                    settings=settings,
                    provenance=provenance,
                    role="heldout",
                    trial_number=trial_number,
                    dry_run=dry_run,
                )
        heldout = [row for row in store.list_study(study_name) if row.role == "heldout"]
    finally:
        store.close()
    selection = build_selection(
        study=study_name,
        scoring_version=provenance.scoring_version,
        expected_samples=expected,
        heldout_rows=heldout,
        baseline_parameters=baseline,
        baseline_hash=baseline_hash,
    )
    if not dry_run:
        paths.selection.write_text(selection.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return selection


if __name__ == "__main__":
    app()

"""Operator commands for the practice-mode GATK tuner.

The command you want is::

    python -m tuning.cli tune
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import urllib.request
from pathlib import Path

from minos_layout import minos_root, project_root
from tuning.errors import ScoringVersionAbort, TuningError

UUID = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
NETWORK_CONFIG_URL = "https://api.theminos.ai/scoring/network-config"
DEFAULT_SAMPLE = "2ea4a48e-ba7f-43d8-86e7-81ae5a99196f"
DEFAULT_TRIALS = 40


def repo_root_from(start: Path | None = None) -> Path:
    """Minos checkout used for ``start-miner.sh`` and ``configs/gatk.conf``."""
    if start is None and os.environ.get("MINOS_REPO_ROOT"):
        return minos_root()
    current = (start or Path(__file__).resolve()).parent
    for candidate in [current, *current.parents]:
        if (candidate / "start-miner.sh").is_file() and (candidate / "templates").is_dir():
            return candidate
    try:
        return minos_root()
    except RuntimeError as exc:
        raise TuningError(str(exc)) from exc


def fetch_scoring_version(url: str = NETWORK_CONFIG_URL, timeout: float = 10) -> str:
    from utils.scoring_version import normalise

    with urllib.request.urlopen(url, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    version = normalise(payload.get("scoring_version") if isinstance(payload, dict) else None)
    if version is None:
        raise TuningError(f"{url} did not contain scoring_version v1 or v2")
    return version


def resolve_scoring_version(explicit: str | None) -> tuple[str, str]:
    from utils.scoring_version import normalise

    if explicit:
        version = normalise(explicit)
        if version is None:
            raise TuningError("--scoring-version must be v1 or v2")
        return version, "cli"
    try:
        return fetch_scoring_version(), "network"
    except TuningError:
        raise
    except Exception as exc:
        raise TuningError(
            "Could not read /scoring/network-config. Pass --scoring-version after confirming "
            "the live version. This tuner does not fall back to a cached version. "
            f"({exc.__class__.__name__}: {exc})"
        ) from exc


def preflight(repo: Path, baseline: Path, train_file: Path | None, sample_id: str | None = None) -> dict:
    checks = []

    def add(name: str, ok: bool, detail: str, severity: str = "error") -> None:
        checks.append({"name": name, "ok": ok, "severity": severity, "detail": detail})

    add("python", sys.version_info >= (3, 11), sys.version.split()[0])
    add("start-miner.sh", (repo / "start-miner.sh").is_file(), str(repo / "start-miner.sh"))
    try:
        import optuna

        add("optuna", True, optuna.__version__)
    except ImportError as exc:
        add("optuna", False, str(exc))
    docker = shutil.which("docker")
    if not docker:
        add("docker", False, "docker is not on PATH", "warning")
    else:
        import subprocess

        probe = subprocess.run(["docker", "info"], capture_output=True, text=True)
        add("docker", probe.returncode == 0, "docker info" if probe.returncode == 0 else probe.stderr[-300:], "warning")
    try:
        from tuning.baseline import FIXED_VALUES, load_baseline, type_counts

        values = load_baseline(baseline)
        counts = type_counts(values)
        add("baseline_25", len(values) == 25, str(counts))
        fixed_ok = all(values.get(name) == expected for name, expected in FIXED_VALUES.items())
        add("fixed_fields", fixed_ok, str({name: values.get(name) for name in FIXED_VALUES}))
    except Exception as exc:
        add("baseline_25", False, str(exc))
    if sample_id is not None:
        add("sample_id", bool(UUID.match(sample_id)), sample_id)
        add(
            "single_sample",
            True,
            "one training id: results from optimize are single-sample exploratory",
            "warning",
        )
    if train_file is not None:
        from tuning.storage import load_sample_file

        try:
            ids = load_sample_file(train_file)
            bad = [item for item in ids if not UUID.match(item)]
            add("train_sample_ids", not bad, f"{len(ids)} ids" if not bad else f"not uuids: {bad[:3]}")
            if len(ids) < 2:
                add(
                    "single_sample",
                    True,
                    "one training id: results from optimize are single-sample exploratory",
                    "warning",
                )
        except Exception as exc:
            add("train_sample_ids", False, str(exc))
    add("gatk_not_launched", True, "preflight does not start HaplotypeCaller")
    ready = all(item["ok"] or item["severity"] == "warning" for item in checks)
    return {"ready": ready, "checks": checks}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m tuning.cli")
    parser.add_argument("--repo-root", type=Path, default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    tune = sub.add_parser("tune", help="Search the 8 high-importance GATK parameters.")
    tune.add_argument(
        "--samples",
        default=None,
        help="Training UUIDs, comma-separated, or a path to a text file. Default: one built-in sample.",
    )
    tune.add_argument("--run-dir", type=Path, default=None)
    tune.add_argument("--baseline", type=Path, default=None)
    tune.add_argument("--scoring-version", default=None)
    tune.add_argument("--validation-samples", type=Path, default=None)
    tune.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    tune.add_argument("--seed", type=int, default=42)
    tune.add_argument("--timeout-seconds", type=int, default=3600)
    tune.add_argument("--dry-run", action="store_true")

    pre = sub.add_parser("preflight", help="Check the repo, baseline, and sample file. Does not run GATK.")
    pre.add_argument("--baseline", type=Path, default=None)
    pre.add_argument("--train-samples", type=Path, default=None)
    pre.add_argument("--sample-id", default=None, help="One training UUID. Does not start GATK.")

    opt = sub.add_parser("optimize", help="Search the 8 high-importance parameters.")
    opt.add_argument("--phase", type=int, required=True, choices=(1, 2))
    opt.add_argument("--trials", type=int, required=True)
    opt.add_argument("--train-samples", type=Path, default=None)
    opt.add_argument("--run-dir", type=Path, required=True)
    opt.add_argument("--baseline", type=Path, default=None)
    opt.add_argument("--scoring-version", default=None)
    opt.add_argument("--seed", type=int, default=None)
    opt.add_argument("--robustness-penalty", type=float, default=None)
    opt.add_argument("--workers", type=int, default=1)
    opt.add_argument("--timeout-seconds", type=int, default=3600)
    opt.add_argument("--resume", action="store_true")
    opt.add_argument("--dry-run", action="store_true")

    val = sub.add_parser("validate", help="Score frozen finalists on held-out samples.")
    val.add_argument("--run-dir", type=Path, required=True)
    val.add_argument("--validation-samples", type=Path, required=True)
    val.add_argument("--baseline", type=Path, default=None)
    val.add_argument("--finalists", type=int, default=None)
    val.add_argument("--timeout-seconds", type=int, default=3600)
    val.add_argument("--exploratory", action="store_true")

    rep = sub.add_parser("report", help="Rewrite the phase report from stored trials.")
    rep.add_argument("--run-dir", type=Path, required=True)
    rep.add_argument("--phase", type=int, default=None, choices=(1, 2))

    exp = sub.add_parser("export", help="Write a promoted config. Never writes configs/gatk.conf.")
    exp.add_argument("--run-dir", type=Path, required=True)
    exp.add_argument("--output", type=Path, required=True)
    exp.add_argument("--exploratory", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        repo = Path(args.repo_root).resolve() if args.repo_root else repo_root_from()
        baseline = _baseline(repo, getattr(args, "baseline", None))
        if args.command == "tune":
            return _tune(args, repo, baseline)
        if args.command == "preflight":
            payload = preflight(repo, baseline, args.train_samples, args.sample_id)
            _print(payload)
            return 0 if payload["ready"] else 1
        if args.command == "optimize":
            return _optimize(args, repo, baseline)
        if args.command == "validate":
            return _validate(args, repo, baseline)
        if args.command == "report":
            return _report(args)
        if args.command == "export":
            from tuning.validate import export_promoted

            path = export_promoted(args.run_dir, args.output, exploratory=args.exploratory)
            _print({"exported": str(path)})
            return 0
    except (TuningError, ScoringVersionAbort) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 2


def _train_ids(samples: str | None) -> list[str]:
    if not samples:
        return [DEFAULT_SAMPLE]
    path = Path(samples)
    if path.is_file():
        from tuning.storage import load_sample_file

        return load_sample_file(path)
    return [part.strip() for part in samples.split(",") if part.strip()]


def _tune(args, repo: Path, baseline: Path) -> int:
    """Preflight, then one study over the 8 high-importance parameters."""
    run_dir = args.run_dir if args.run_dir is not None else project_root() / "tuning_runs" / "my_run"
    samples = _train_ids(args.samples)
    bad = [item for item in samples if not UUID.match(item)]
    if bad:
        raise TuningError(f"sample ids must be UUIDs, got {bad[:3]}")
    run_dir.mkdir(parents=True, exist_ok=True)
    train_file = run_dir / "train_samples.txt"
    train_file.write_text("\n".join(samples) + "\n", encoding="utf-8")
    sample_id = samples[0] if len(samples) == 1 else None
    check = preflight(repo, baseline, train_file, sample_id)
    if not check["ready"]:
        _print(check)
        return 1
    if args.trials < 1:
        raise TuningError("--trials must be >= 1")
    from tuning.parameters import tunable_names

    names = ", ".join(tunable_names(1))
    print(f"[tune] searching 8 parameters: {names}", flush=True)
    print(f"[tune] {args.trials} trials. The other 17 parameters stay at the baseline.", flush=True)
    if args.dry_run:
        print(
            f"dry-run: would run {args.trials} trials on {len(samples)} sample(s). GATK is not started.",
            file=sys.stderr,
        )
    code = _optimize(
        argparse.Namespace(
            phase=1,
            trials=args.trials,
            train_samples=train_file,
            run_dir=run_dir,
            scoring_version=args.scoring_version,
            seed=args.seed,
            robustness_penalty=None,
            workers=1,
            timeout_seconds=args.timeout_seconds,
            resume=True,
            dry_run=args.dry_run,
        ),
        repo,
        baseline,
    )
    if code != 0 or args.dry_run:
        return code
    report_code = _report(argparse.Namespace(run_dir=run_dir, phase=None))
    if report_code != 0 or args.validation_samples is None:
        if args.validation_samples is None:
            print(
                "Held-out check skipped. Pass --validation-samples FILE with at least "
                "3 other sample ids when you want a certified config.",
                file=sys.stderr,
            )
        return report_code
    code = _validate(
        argparse.Namespace(
            run_dir=run_dir,
            validation_samples=args.validation_samples,
            timeout_seconds=args.timeout_seconds,
            exploratory=False,
            finalists=5,
        ),
        repo,
        baseline,
    )
    if code != 0:
        return code
    from tuning.validate import export_promoted

    destination = run_dir / "final" / "best_validated_gatk.conf"
    path = export_promoted(run_dir, destination, exploratory=False)
    _print({"exported": str(path)})
    return 0


def _baseline(repo: Path, given: Path | None) -> Path:
    return given if given is not None else repo / "configs" / "gatk.conf"


def _optimize(args, repo: Path, baseline: Path) -> int:
    from tuning.optimize import dry_run_trial, run_phase
    from tuning.storage import load_sample_file, read_json

    manifest_path = args.run_dir / "manifest.json"
    existing = read_json(manifest_path) if manifest_path.exists() else None
    if args.train_samples is not None:
        samples = load_sample_file(args.train_samples)
    elif existing is not None:
        samples = list(existing["train_samples"])
    else:
        raise TuningError("--train-samples is required to start a run")
    bad = [item for item in samples if not UUID.match(item)]
    if bad:
        raise TuningError(f"train sample ids must be UUIDs, got {bad[:3]}")
    seed = args.seed if args.seed is not None else (existing["seed"] if existing else 42)
    penalty = (
        args.robustness_penalty
        if args.robustness_penalty is not None
        else (existing["robustness_penalty"] if existing else 0.25)
    )
    if args.dry_run:
        payload = dry_run_trial(
            repo_root=repo,
            run_dir=args.run_dir,
            baseline_path=baseline,
            phase=args.phase,
            sample_id=samples[0],
        )
        _print(payload)
        return 0
    if args.scoring_version is None and existing is not None:
        version, source = existing["scoring_version"], "manifest"
    else:
        version, source = resolve_scoring_version(args.scoring_version)
    if len(samples) < 2:
        print(
            "warning: one training sample. This optimize run is single-sample exploratory "
            "and does not support a generalization claim.",
            file=sys.stderr,
        )
    summary = run_phase(
        repo_root=repo,
        run_dir=args.run_dir,
        baseline_path=baseline,
        phase=args.phase,
        trials=args.trials,
        train_samples=samples,
        scoring_version=version,
        seed=seed,
        robustness_penalty=penalty,
        workers=args.workers,
        timeout_seconds=args.timeout_seconds,
    )
    summary["scoring_version_source"] = source
    _print(summary)
    return 0


def _validate(args, repo: Path, baseline: Path) -> int:
    from tuning.storage import load_sample_file
    from tuning.validate import run_validation

    samples = load_sample_file(args.validation_samples)
    report = run_validation(
        repo_root=repo,
        run_dir=args.run_dir,
        baseline_path=baseline,
        validation_samples=samples,
        timeout_seconds=args.timeout_seconds,
        runner=None,
        exploratory=args.exploratory,
        finalist_limit=args.finalists,
    )
    _print({"decision": report["decision"], "chosen": report["chosen"]})
    return 0 if report["decision"] == "PROMOTED" else 2


def _report(args) -> int:
    from tuning.reports import trial_records, write_phase_report
    from tuning.storage import read_json

    manifest = read_json(args.run_dir / "manifest.json")
    phases = [args.phase] if args.phase is not None else [1, 2]
    summaries = []
    for phase in phases:
        if args.phase is None and not trial_records(args.run_dir, phase):
            continue
        summaries.append(write_phase_report(args.run_dir, phase, manifest))
    if not summaries:
        raise TuningError(f"{args.run_dir} has no phase trials to report")
    _print({"phases": summaries})
    return 0


def _print(payload: dict) -> None:
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    raise SystemExit(main())

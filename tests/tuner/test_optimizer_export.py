"""Optuna objective, held-out gate, and export safety."""

from __future__ import annotations

from pathlib import Path

import optuna
from optuna.trial import TrialState

from tuner.config_generator import config_hash, load_conf, normalize_options
from tuner.errors import ExportError, StudyConflict, TunerError
from tuner.exporter import export_config
from tuner.optimizer import assert_worker_budget, run_study, startup_trials
from tuner.parameters import load_search_space
from tuner.settings import StudyPaths
from tuner.storage import Evaluation, Store
from tuner.validation import build_selection, ensure_split, split_samples

ROOT = Path(__file__).resolve().parents[2]


def test_split_holds_out_one_sample_per_chromosome():
    ids = [f"id-{index}" for index in range(6)]
    chromosomes = {
        "id-0": "chr20",
        "id-1": "chr20",
        "id-2": "chr21",
        "id-3": "chr21",
        "id-4": "chr22",
        "id-5": "chr22",
    }
    train, heldout, description = split_samples(ids, chromosomes)
    assert heldout == ["id-1", "id-3", "id-5"]
    assert train == ["id-0", "id-2", "id-4"]
    assert "chr20" in description
    assert split_samples(ids)[1] != heldout


def test_explicit_ids_are_the_study_and_must_be_runnable():
    from tuner.catalog import assert_selected_ids

    chosen = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaa1"
    assert assert_selected_ids([chosen], [chosen]) == "selected 1 sample id for this study"
    with __import__("pytest").raises(TunerError, match="not runnable"):
        assert_selected_ids([chosen, "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"], [chosen])


def test_catalog_refuses_a_partial_sample_list():
    from tuner.catalog import assert_catalog_coverage

    listed = [f"aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaa{index}" for index in range(10)]
    ready = listed[:2]
    with __import__("pytest").raises(TunerError, match="catalog 10 ids"):
        assert_catalog_coverage(listed, ready, allow_partial=False)
    summary = assert_catalog_coverage(listed, ready, allow_partial=True)
    assert "runnable 2" in summary


def test_leaderboard_summary_uses_the_zero_to_one_scale():
    from tuner.reporting import leaderboard_summary

    row = _held("sample-a", 86.0, "hash", {"min_pruning": 2}, -1, role="baseline")
    row.submetrics = {"overcall_penalty": 45.0}
    summary = leaderboard_summary([row], ["sample-a"])
    assert summary["baseline_mean"] == 86.0
    assert summary["baseline_adv_score"] == 0.86
    assert summary["penalty_capped_samples"] == ["sample-a"]


def test_split_holds_out_three_when_possible_and_is_stable():
    ids = [f"id-{index}" for index in range(5)]
    train, heldout, description = split_samples(ids)
    assert len(heldout) == 3
    assert set(train).isdisjoint(heldout)
    assert "3" in description
    assert split_samples(ids)[0] == train


def test_optuna_uses_only_training_samples_and_resumes(tuner_home, provenance):
    samples = sorted(
        __import__("tuner.datasets", fromlist=["discover_samples"]).discover_samples(tuner_home),
        key=lambda item: item.sample_id,
    )
    train = samples[:2]
    heldout_id = samples[2].sample_id
    space = load_search_space(ROOT / "configs" / "search_space.yaml", "core")
    baseline = normalize_options(load_conf(ROOT / "configs" / "baseline.conf"))
    seen = []

    def evaluate_fn(**kwargs):
        seen.append(kwargs["sample"].sample_id)
        if kwargs["sample"].sample_id == train[1].sample_id:
            return _store_eval(kwargs, status="failed", score=None)
        return _store_eval(kwargs, status="success", score=80.0)

    first = run_study(
        settings=tuner_home,
        study_name="opt",
        n_trials=3,
        space=space,
        train_samples=train,
        baseline_options=baseline,
        provenance=provenance,
        evaluate_fn=evaluate_fn,
    )
    assert heldout_id not in seen
    assert first["best_value"] is None
    paths = StudyPaths.for_study(tuner_home, "opt")
    study = optuna.load_study(study_name="opt", storage="sqlite:///" + str(paths.optuna_db.resolve()))
    assert all(trial.state == TrialState.FAIL for trial in study.trials)
    stored = Store(paths.evaluations_db)
    rows = stored.list_study("opt")
    stored.close()
    assert any(row.score == 80.0 for row in rows)
    assert any(row.status == "failed" and row.score is None for row in rows)
    before = len(seen)
    run_study(
        settings=tuner_home,
        study_name="opt",
        n_trials=3,
        space=space,
        train_samples=train,
        baseline_options=baseline,
        provenance=provenance,
        evaluate_fn=evaluate_fn,
    )
    assert len(seen) == before
    with __import__("pytest").raises(StudyConflict):
        run_study(
            settings=tuner_home,
            study_name="opt",
            n_trials=1,
            space=space,
            train_samples=train,
            baseline_options=baseline,
            provenance=provenance.model_copy(update={"scoring_version": "v2", "scorer_name": "AdvancedV2"}),
            evaluate_fn=evaluate_fn,
        )


def test_successful_objective_is_the_official_mean(tuner_home, provenance):
    from tuner.datasets import discover_samples

    sample = discover_samples(tuner_home)[0]
    space = load_search_space(ROOT / "configs" / "search_space.yaml", "core")
    baseline = normalize_options(load_conf(ROOT / "configs" / "baseline.conf"))

    def evaluate_fn(**kwargs):
        score = 10.0 + float(kwargs["options"]["min_pruning"])
        return _store_eval(kwargs, status="success", score=score)

    result = run_study(
        settings=tuner_home,
        study_name="good",
        n_trials=2,
        space=space,
        train_samples=[sample],
        baseline_options=baseline,
        provenance=provenance,
        evaluate_fn=evaluate_fn,
    )
    assert result["best_value"] is not None
    assert result["best_value"] >= 11.0


def test_worker_budget_and_startup():
    assert startup_trials(50) == 10
    assert startup_trials(3) == 3
    assert_worker_budget(1, 1, cpu=2, allow=False)
    with __import__("pytest").raises(TunerError):
        assert_worker_budget(4, 2, cpu=4, allow=False)


def test_heldout_gate_prefers_baseline_when_a_sample_regresses():
    baseline_params = {"min_pruning": 2}
    baseline_hash = config_hash(baseline_params)
    steady = {"min_pruning": 3}
    steady_hash = config_hash(steady)
    wild = {"min_pruning": 4}
    wild_hash = config_hash(wild)
    rows = [
        _held("s1", 50, baseline_hash, baseline_params, -1),
        _held("s2", 50, baseline_hash, baseline_params, -1),
        _held("s1", 60, steady_hash, steady, 1),
        _held("s2", 60, steady_hash, steady, 1),
        _held("s1", 90, wild_hash, wild, 2),
        _held("s2", 10, wild_hash, wild, 2),
    ]
    selection = build_selection(
        study="s",
        scoring_version="v1",
        expected_samples=["s1", "s2"],
        heldout_rows=rows,
        baseline_parameters=baseline_params,
        baseline_hash=baseline_hash,
    )
    assert selection.selected_config_hash == steady_hash
    assert selection.baseline_preferred is False
    only_wild = [row for row in rows if row.config_hash != steady_hash]
    gated = build_selection(
        study="s",
        scoring_version="v1",
        expected_samples=["s1", "s2"],
        heldout_rows=only_wild,
        baseline_parameters=baseline_params,
        baseline_hash=baseline_hash,
    )
    assert gated.baseline_preferred
    assert gated.selected_config_hash == baseline_hash
    assert gated.allow_force_tuned


def test_export_validates_and_refuses_the_miner_config(tuner_home, tmp_path):
    params = normalize_options(load_conf(ROOT / "configs" / "baseline.conf"))
    hashed = config_hash(params)
    paths = StudyPaths.for_study(tuner_home, "exportme")
    store = Store(paths.evaluations_db)
    store.record(_held("s1", 70, hashed, params, -1, study="exportme", role="train", status="success"))
    store.close()
    selection_path = paths.selection
    from tuner.validation import Selection

    selection = Selection(
        study="exportme",
        scoring_version="v1",
        selected_label="baseline",
        selected_config_hash=hashed,
        selected_parameters=params,
        selected_reason="kept",
        baseline_preferred=True,
        tuned_leader_hash=hashed,
        tuned_leader_parameters=params,
        tuned_leader_complete=True,
        allow_force_tuned=True,
    )
    selection_path.write_text(selection.model_dump_json(indent=2), encoding="utf-8")
    output = tmp_path / "best" / "gatk.conf"
    result = export_config(tuner_home, "exportme", output)
    text = output.read_text(encoding="utf-8")
    from tuner.config_generator import parse_conf
    from tuner.config_validator import validate_gatk_options

    parsed = parse_conf(text)
    assert validate_gatk_options(parsed).valid
    assert Path(result["sidecar"]).exists()
    assert not list(output.parent.glob(".tmp-*"))
    repo = tmp_path / "repo"
    (repo / "configs").mkdir(parents=True)
    active = repo / "configs" / "gatk.conf"
    active.write_text("original\n", encoding="utf-8")
    copied = tuner_home.model_copy(update={"repo_root": repo})
    with __import__("pytest").raises(ExportError):
        export_config(copied, "exportme", active, force=True)
    assert active.read_text(encoding="utf-8") == "original\n"
    bad = tmp_path / "bad.conf"
    store = Store(paths.evaluations_db)
    store.record(
        _held(
            "s2",
            10,
            config_hash({"not_a_gatk_param": 1}),
            {"not_a_gatk_param": 1},
            3,
            study="exportme",
            role="train",
            status="success",
        )
    )
    store.close()
    selection.selected_parameters = {"not_a_gatk_param": 1}
    selection.selected_config_hash = config_hash({"not_a_gatk_param": 1})
    selection.allow_force_tuned = False
    selection_path.write_text(selection.model_dump_json(indent=2), encoding="utf-8")
    with __import__("pytest").raises(ExportError):
        export_config(tuner_home, "exportme", bad, force=True)
    assert not bad.exists()


def test_export_force_is_required_without_heldout_selection(tuner_home, tmp_path):
    params = normalize_options({"min_base_quality_score": 12, "pcr_indel_model": "NONE"})
    paths = StudyPaths.for_study(tuner_home, "trainonly")
    store = Store(paths.evaluations_db)
    store.record(_held("s1", 60, config_hash(params), params, 1, study="trainonly", role="train"))
    store.close()
    output = tmp_path / "gatk.conf"
    with __import__("pytest").raises(ExportError):
        export_config(tuner_home, "trainonly", output, force=False)
    assert not output.exists()
    export_config(tuner_home, "trainonly", output, force=True)
    assert output.exists()


def test_split_conflict_on_profile_change(tuner_home, tmp_path):
    space = load_search_space(ROOT / "configs" / "search_space.yaml", "core")
    meta_path = tmp_path / "study.json"
    ensure_split(
        meta_path=meta_path,
        study_name="s",
        ready_ids=["b", "a"],
        space=space,
        settings=tuner_home,
        scoring_version="v1",
        commit="abc",
    )
    again = ensure_split(
        meta_path=meta_path,
        study_name="s",
        ready_ids=["a", "b", "c"],
        space=space,
        settings=tuner_home,
        scoring_version="v1",
        commit="abc",
    )
    assert again.train_sample_ids == ["a"]
    advanced = load_search_space(ROOT / "configs" / "search_space.yaml", "advanced")
    with __import__("pytest").raises(StudyConflict):
        ensure_split(
            meta_path=meta_path,
            study_name="s",
            ready_ids=["a", "b"],
            space=advanced,
            settings=tuner_home,
            scoring_version="v1",
            commit="abc",
        )


def _store_eval(kwargs, *, status, score):
    options = normalize_options(kwargs["options"])
    hashed = config_hash(options)
    provenance = kwargs["provenance"]
    evaluation = Evaluation(
        cache_key=f"{kwargs['role']}:{kwargs['trial_number']}:{kwargs['sample'].sample_id}:{hashed}:{status}",
        study=kwargs["study"],
        trial_number=kwargs["trial_number"],
        sample_id=kwargs["sample"].sample_id,
        role=kwargs["role"],
        parameters=options,
        status=status,
        score=score,
        combined_final=None if score is None else score / 100.0,
        config_hash=hashed,
        minos_commit=provenance.minos_commit,
        gatk_image=provenance.gatk_image,
        scorer_name=provenance.scorer_name,
        scoring_version=provenance.scoring_version,
        validator_would_record=status == "success",
    )
    return kwargs["store"].record(evaluation)


def _held(sample_id, score, hashed, params, trial, study="s", role="heldout", status="success"):
    return Evaluation(
        cache_key=f"{role}:{trial}:{sample_id}:{hashed}",
        study=study,
        trial_number=trial,
        sample_id=sample_id,
        role=role,
        parameters=params,
        status=status,
        score=score,
        combined_final=score / 100.0,
        config_hash=hashed,
        minos_commit="abc",
        gatk_image="gatk",
        scorer_name="Advanced",
        scoring_version="v1",
        validator_would_record=True,
    )

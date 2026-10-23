"""Dataset checks, SQLite cache, and official score delegation."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from tuner.config_generator import config_hash, normalize_options
from tuner.datasets import (
    classify_tokens,
    compatibility_errors,
    discover_samples,
    prepare_datasets,
    public_metadata,
    region_from_challenge_bed,
    resolve_sample,
)
from tuner.errors import PrepareError
from tuner.objective import evaluate_candidate
from tuner.runner import GatkResult, interpret_gatk_result
from tuner.scorer import ScoreRun, project_metrics, score_from_official_metrics, score_vcf
from tuner.storage import Evaluation, Store, evaluation_cache_key
from minos_layout import minos_root
from utils.scoring import AdvancedScorer

ROOT = Path(__file__).resolve().parents[2]
MINOS = minos_root()


def test_chromosome_token_is_not_a_sample_id():
    chromosomes, sample_ids, rejected = classify_tokens(["chr20", "chr18", "not an id"])
    assert chromosomes == {"chr20", "chr18"}
    assert sample_ids == set()
    assert rejected == ["not an id"]


def test_region_mismatch_and_challenge_bed(tmp_path):
    errors = compatibility_errors(
        region="chr21:10-20",
        bam_contigs=["chr20"],
        fai_contigs=["chr20"],
        truth_contigs=["chr20"],
        fasta_header=">chr20",
        bam_exists=True,
        bai_exists=True,
        truth_exists=True,
        truth_index_exists=True,
        reference_exists=True,
        fai_exists=True,
        sdf_exists=True,
        mutations_exists=True,
    )
    assert any("chr21" in item and "BAM" in item for item in errors)
    bed_dir = tmp_path / "score_run"
    bed_dir.mkdir()
    (bed_dir / "challenge_region_chr20_13522546_15723106_output.vcf.bed").write_text(
        "chr20\t13522545\t15723106\n",
        encoding="utf-8",
    )
    assert region_from_challenge_bed(tmp_path) == "chr20:13522546-15723106"


def test_prepare_names_unstaged_catalog_ids(tuner_home, tmp_path, monkeypatch):
    monkeypatch.delenv("PLATFORM_URL", raising=False)
    catalog = tmp_path / "ids.md"
    catalog.write_text(
        "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaa1\n"
        "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb\n",
        encoding="utf-8",
    )
    settings = tuner_home.model_copy(update={"sample_id_file": catalog})
    result = prepare_datasets(settings, None)
    assert result.ready_count == 3
    assert any("1 of 2" in message for message in result.messages)


def test_prepare_filters_chromosomes_and_rejects_garbage(tuner_home):
    bad = prepare_datasets(tuner_home, ["chr18"])
    assert bad.ready_count == 0
    assert any("chromosome filter" in message for message in bad.messages)
    good = prepare_datasets(tuner_home, ["chr20"])
    assert good.ready_count == 3
    with pytest.raises(PrepareError):
        prepare_datasets(tuner_home, ["chr20-window"])
    with pytest.raises(PrepareError):
        resolve_sample(tuner_home, "chr20")


def test_discover_blocks_contig_mismatch(tuner_home):
    samples = discover_samples(tuner_home, hash_files=False)
    target = next(item for item in samples if item.sample_id.endswith("1"))
    target.region = "chr21:10-20"
    from tuner.datasets import _fill_paths, _load_json

    target.chromosome = "chr21"
    target.blockers = []
    _fill_paths(
        tuner_home,
        target,
        Path(target.directory),
        hash_files=False,
        sidecar=_load_json(Path(target.directory) / "sample.json"),
    )
    assert any("BAM" in item for item in target.blockers)
    assert not target.ready


def test_sqlite_resume_does_not_duplicate(tmp_path):
    store = Store(tmp_path / "eval.sqlite3")
    evaluation = _evaluation(cache="same")
    store.record(evaluation)
    store.record(evaluation)
    assert len(store.list_study("study")) == 1
    store.close()
    reopened = Store(tmp_path / "eval.sqlite3")
    assert len(reopened.list_study("study")) == 1
    assert reopened.list_study("study")[0].score == 80.0
    reopened.close()


def test_failed_row_is_replaced_by_a_later_score(tmp_path):
    store = Store(tmp_path / "eval.sqlite3")
    failed = _evaluation(cache="same")
    failed.status = "failed"
    failed.score = None
    failed.combined_final = None
    failed.failure_reason = "v2 could not be computed"
    failed.validator_would_record = None
    store.record(failed)
    scored = _evaluation(cache="same")
    stored = store.record(scored)
    rows = store.list_study("study")
    assert len(rows) == 1
    assert stored.status == "success"
    assert stored.score == 80.0
    assert rows[0].failure_reason is None
    store.close()


def test_cache_misses_when_version_or_bam_changes(tuner_home, provenance):
    sample = resolve_sample(tuner_home, "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaa1")
    options = normalize_options({"min_base_quality_score": 10, "pcr_indel_model": "NONE"})
    store = Store(tuner_home.experiments_dir / "cache.sqlite3")
    calls = {"gatk": 0}

    def runner(sample, options, work, settings):
        calls["gatk"] += 1
        work.mkdir(parents=True, exist_ok=True)
        vcf = work / "output.vcf.gz"
        vcf.write_bytes(b"vcf")
        return GatkResult(success=True, variant_count=1, runtime_seconds=1.0, vcf_path=str(vcf))

    def scorer(sample, vcf, work, provenance):
        return ScoreRun(
            ok=True,
            advanced_score=80.0,
            combined_final=0.8,
            scoring_version=provenance.scoring_version,
            scorer_name=provenance.scorer_name,
            validator_would_record=True,
            projected={"f1_snp": 0.9},
        )

    first = evaluate_candidate(
        store=store,
        study="study",
        sample=sample,
        options=options,
        settings=tuner_home,
        provenance=provenance,
        role="train",
        trial_number=1,
        gatk_runner=runner,
        score_fn=scorer,
    )
    second = evaluate_candidate(
        store=store,
        study="study",
        sample=sample,
        options=options,
        settings=tuner_home,
        provenance=provenance,
        role="train",
        trial_number=2,
        gatk_runner=runner,
        score_fn=scorer,
    )
    assert first.score == 80.0
    assert second.cached
    assert calls["gatk"] == 1
    other = provenance.model_copy(update={"scoring_version": "v2", "scorer_name": "AdvancedV2"})
    evaluate_candidate(
        store=store,
        study="study",
        sample=sample,
        options=options,
        settings=tuner_home,
        provenance=other,
        role="train",
        trial_number=3,
        gatk_runner=runner,
        score_fn=scorer,
    )
    bam = Path(sample.bam)
    bam.write_bytes(bam.read_bytes() + b"x")
    evaluate_candidate(
        store=store,
        study="study",
        sample=sample,
        options=options,
        settings=tuner_home,
        provenance=provenance,
        role="train",
        trial_number=4,
        gatk_runner=runner,
        score_fn=scorer,
    )
    assert calls["gatk"] == 3
    store.close()


def test_timeout_and_nonzero_exit_are_not_scores(tuner_home, provenance):
    sample = resolve_sample(tuner_home, "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaa1")
    options = normalize_options({"min_base_quality_score": 10})
    store = Store(tuner_home.experiments_dir / "fail.sqlite3")
    scored = {"n": 0}

    def timeout_runner(sample, options, work, settings):
        return interpret_gatk_result(
            {"success": False, "error": "Timeout after 12s"},
            work / "output.vcf.gz",
            work / "gatk.log",
            12.0,
        )

    def boom_scorer(*args, **kwargs):
        scored["n"] += 1
        raise AssertionError("scorer should not run")

    timed = evaluate_candidate(
        store=store,
        study="study",
        sample=sample,
        options=options,
        settings=tuner_home,
        provenance=provenance,
        role="train",
        trial_number=1,
        gatk_runner=timeout_runner,
        score_fn=boom_scorer,
    )
    assert timed.status == "timeout"
    assert timed.score is None

    def nonzero_runner(sample, options, work, settings):
        return interpret_gatk_result(
            {"success": False, "error": "rc=1 HaplotypeCaller failed"},
            work / "missing.vcf.gz",
            work / "gatk.log",
            2.0,
        )

    failed = evaluate_candidate(
        store=store,
        study="study",
        sample=sample,
        options=options,
        settings=tuner_home,
        provenance=provenance,
        role="train",
        trial_number=2,
        gatk_runner=nonzero_runner,
        score_fn=boom_scorer,
    )
    assert failed.status == "failed"
    assert failed.score is None
    assert scored["n"] == 0
    invalid = evaluate_candidate(
        store=store,
        study="study",
        sample=sample,
        options={"nope": 1},
        settings=tuner_home,
        provenance=provenance,
        role="train",
        trial_number=3,
        gatk_runner=nonzero_runner,
        score_fn=boom_scorer,
    )
    assert invalid.status == "failed"
    assert "nope" in (invalid.failure_reason or "")
    store.close()


def test_official_score_is_delegated_and_missing_metrics_stay_null(tmp_path):
    metrics = {
        "f1_snp": 0.9,
        "truth_total_snp": 10,
        "truth_total_indel": 4,
        "recall_snp": 0.9,
        "recall_indel": 0.5,
        "precision_snp": 0.9,
        "precision_indel": 0.5,
        "query_total_snp": 10,
        "query_total_indel": 4,
        "fp_snp": 1,
        "fp_indel": 1,
    }
    projected = project_metrics(metrics)
    assert projected["f1_indel"] is None
    assert projected["f1_snp"] == 0.9
    run = score_from_official_metrics(
        metrics,
        scoring_version="v1",
        scorer_name="Advanced",
        truth_vcf=None,
        work_dir=tmp_path,
        commit="abc",
    )
    assert run.advanced_score == AdvancedScorer.compute_advanced_score(metrics)
    assert run.projected["f1_indel"] is None
    empty = score_from_official_metrics(
        None,
        scoring_version="v1",
        scorer_name="Advanced",
        truth_vcf=None,
        work_dir=tmp_path,
        commit="abc",
    )
    assert empty.advanced_score is None
    assert empty.ok is False


def test_v2_does_not_fall_back_when_it_cannot_score(tmp_path):
    run = score_from_official_metrics(
        {"f1_snp": 0.5, "truth_total_snp": 1, "truth_total_indel": 1},
        scoring_version="v2",
        scorer_name="AdvancedV2",
        truth_vcf=None,
        work_dir=tmp_path,
        commit="abc",
    )
    assert run.advanced_score is None
    assert "v1" in (run.failure_reason or "")


def test_score_vcf_preserves_a_null_from_happy(monkeypatch, tuner_home, provenance, tmp_path):
    sample = resolve_sample(tuner_home, "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaa1")

    class FakeHappy:
        def score_vcf(self, **kwargs):
            return None

    monkeypatch.setattr("utils.scoring.HappyScorer", FakeHappy)
    run = score_vcf(sample, tmp_path / "output.vcf.gz", tmp_path, provenance)
    assert run.advanced_score is None
    assert run.ok is False


def test_real_happy_summary_has_official_columns():
    summary = (
        MINOS
        / "datasets"
        / "practice"
        / "round_2aeeddc1d86288f3"
        / "score_run"
        / "happy_output.vcf.summary.csv"
    )
    if not summary.exists():
        pytest.skip("local practice summary is not staged")
    with summary.open(encoding="utf-8") as handle:
        header = csv.DictReader(handle).fieldnames
    from tuner.scorer import HAPPY_SUMMARY_COLUMNS

    assert header is not None
    for column in HAPPY_SUMMARY_COLUMNS:
        assert column in header


def test_public_metadata_drops_presigned_urls():
    cleaned = public_metadata({"sample_id": "abc", "bam_presigned_url": "https://secret", "region": "chr20:1-2"})
    assert "bam_presigned_url" not in cleaned
    assert cleaned["region"] == "chr20:1-2"


def _evaluation(cache: str) -> Evaluation:
    return Evaluation(
        cache_key=cache,
        study="study",
        trial_number=1,
        sample_id="sample",
        role="train",
        parameters={"min_pruning": 2},
        status="success",
        score=80.0,
        combined_final=0.8,
        config_hash=config_hash({"min_pruning": 2}),
        minos_commit="abc",
        gatk_image="gatk",
        scorer_name="Advanced",
        scoring_version="v1",
        validator_would_record=True,
    )


def test_cache_key_changes_with_checksum():
    base = {"bam": "a", "scoring_version": "v2"}
    assert evaluation_cache_key(base) != evaluation_cache_key({**base, "bam": "b"})
    assert evaluation_cache_key(base) != evaluation_cache_key({**base, "scoring_version": "v1"})

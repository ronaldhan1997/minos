"""Fixtures for the GATK tuner. They do not read practice BAMs."""

from __future__ import annotations

import gzip
import struct
from pathlib import Path

import pytest

from tuner.settings import load_settings
from tuner.upstream import Provenance

ROOT = Path(__file__).resolve().parents[2]


def write_bam(path: Path, contig: str = "chr20") -> None:
    text = f"@HD\tVN:1.6\tSO:coordinate\n@SQ\tSN:{contig}\tLN:1000\n".encode()
    payload = b"BAM\x01" + struct.pack("<I", len(text)) + text
    path.write_bytes(gzip.compress(payload))


def write_vcf(path: Path, contig: str = "chr20") -> None:
    body = (
        "##fileformat=VCFv4.2\n"
        f"##contig=<ID={contig},length=1000>\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
    )
    path.write_bytes(gzip.compress(body.encode()))


def stage_sample(practice: Path, sample_id: str, region: str, contig: str = "chr20") -> None:
    directory = practice / sample_id
    directory.mkdir(parents=True)
    write_bam(directory / "input.bam", contig)
    (directory / "input.bam.bai").write_bytes(b"bai")
    write_vcf(directory / "truth.vcf.gz", contig)
    (directory / "truth.vcf.gz.tbi").write_bytes(b"tbi")
    write_vcf(directory / "mutations.vcf.gz", contig)
    (directory / "sample.json").write_text(
        '{"sample_id": "%s", "region": "%s", "region_source": "fixture"}\n' % (sample_id, region),
        encoding="utf-8",
    )


def stage_reference(data: Path, contig: str = "chr20") -> None:
    folder = data / "reference" / contig
    folder.mkdir(parents=True)
    (folder / f"{contig}.fa").write_text(f">{contig}\nACGT\n", encoding="utf-8")
    (folder / f"{contig}.fa.fai").write_text(f"{contig}\t1000\t6\t4\t5\n", encoding="utf-8")
    sdf = folder / f"{contig}.sdf"
    sdf.mkdir()
    (sdf / "done").write_text("ok\n", encoding="utf-8")


@pytest.fixture
def tuner_home(tmp_path):
    data = tmp_path / "datasets"
    practice = data / "practice"
    ids = [
        ("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaa1", "chr20:10-40"),
        ("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaa2", "chr20:50-80"),
        ("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaa3", "chr20:90-120"),
    ]
    for sample_id, region in ids:
        stage_sample(practice, sample_id, region)
    stage_reference(data)
    settings = load_settings(
        experiments=tmp_path / "experiments",
        data_dir=data,
        baseline=ROOT / "configs" / "baseline.conf",
        search_space=ROOT / "configs" / "search_space.yaml",
        sample_id_file=tmp_path / "no-ids.md",
    )
    return settings


@pytest.fixture
def provenance():
    return Provenance(
        minos_commit="abc123",
        minos_dirty=False,
        gatk_image="broadinstitute/gatk:4.5.0.0",
        docker_image_id="sha256:test",
        happy_image="happy@sha256:test",
        bcftools_image="bcftools:test",
        scoring_version="v1",
        scorer_name="Advanced",
        hardware={"cpu_count": 2},
        gatk_threads=1,
        memory_gb=4,
    )

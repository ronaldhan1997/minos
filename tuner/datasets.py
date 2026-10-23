"""Discover official practice samples staged on disk.

Sample identifiers are the platform UUIDs. Chromosome names such as ``chr20``
are filters, not identifiers. Files live under
``datasets/practice/<safe_round_dir_name(sample_id)>/``, which is the directory
``neurons.miner._download_practice_files`` writes.

A region is taken from a manifest, from ``sample.json``, or from a challenge
BED left by an official practice score. The read span of the BAM is not used
as a region.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import struct
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from neurons import CHROMOSOME_PATTERN
from templates.tool_params import validate_region
from tuner.errors import PrepareError
from utils.path_utils import safe_round_dir_name

UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
_CONTIG_RE = re.compile(r"ID=([^,>]+)")
_BUILD_BAD_RE = re.compile(r"GRCh37|hg19", re.IGNORECASE)


class FileStamp(BaseModel):
    path: str
    sha256: str
    size: int
    mtime_ns: int


class PracticeSample(BaseModel):
    sample_id: str
    directory: str
    chromosome: str | None = None
    region: str | None = None
    bam: str | None = None
    bai: str | None = None
    truth: str | None = None
    truth_index: str | None = None
    mutations: str | None = None
    mutations_index: str | None = None
    reference: str | None = None
    reference_index: str | None = None
    sdf: str | None = None
    stamps: dict[str, FileStamp] = Field(default_factory=dict)
    ready: bool = False
    blockers: list[str] = Field(default_factory=list)
    region_source: str | None = None
    reference_header: str | None = None


class PrepareResult(BaseModel):
    samples: list[PracticeSample]
    messages: list[str] = Field(default_factory=list)
    ready_count: int = 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stamp_file(path: Path, previous: FileStamp | None = None) -> FileStamp:
    stat = path.stat()
    if (
        previous
        and previous.size == stat.st_size
        and previous.mtime_ns == stat.st_mtime_ns
        and previous.path == str(path)
    ):
        return previous
    return FileStamp(
        path=str(path),
        sha256=sha256_file(path),
        size=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
    )


def hash_tree(path: Path) -> str:
    digest = hashlib.sha256()
    files = [item for item in path.rglob("*") if item.is_file()]
    for item in sorted(files):
        rel = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(rel)
        digest.update(str(item.stat().st_size).encode("ascii"))
        digest.update(sha256_file(item).encode("ascii"))
    return digest.hexdigest()


def read_sample_ids(path: Path | None) -> list[str]:
    if path is None or not path.exists():
        return []
    found = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        token = raw.strip()
        if UUID_RE.match(token):
            found.append(token.lower())
    return found


def classify_tokens(tokens: list[str]) -> tuple[set[str], set[str], list[str]]:
    """Split a CLI list into chromosome filters, sample ids, and rejected tokens."""
    chromosomes: set[str] = set()
    sample_ids: set[str] = set()
    rejected: list[str] = []
    for token in tokens:
        if CHROMOSOME_PATTERN.match(token):
            chromosomes.add(token)
        elif UUID_RE.match(token):
            sample_ids.add(token.lower())
        else:
            rejected.append(token)
    return chromosomes, sample_ids, rejected


def read_bam_contigs(path: Path) -> list[str]:
    """Contig names from the BAM header. BGZF is readable as gzip."""
    with gzip.open(path, "rb") as handle:
        magic = handle.read(4)
        if magic != b"BAM\x01":
            raise PrepareError(f"{path} is not a BAM file (bad magic)")
        raw_len = handle.read(4)
        if len(raw_len) != 4:
            raise PrepareError(f"{path} BAM header length is truncated")
        length = struct.unpack("<I", raw_len)[0]
        if length > 10_000_000:
            raise PrepareError(f"{path} BAM header is unexpectedly large")
        text = handle.read(length).decode("utf-8", "replace")
    contigs = []
    for line in text.splitlines():
        if not line.startswith("@SQ"):
            continue
        for field in line.split("\t"):
            if field.startswith("SN:"):
                contigs.append(field[3:])
    return contigs


def read_vcf_contigs(path: Path) -> list[str]:
    opener = gzip.open if str(path).endswith(".gz") else open
    contigs: list[str] = []
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        for count, line in enumerate(handle):
            if count > 20000:
                break
            if not line.startswith("#"):
                break
            if line.startswith("##contig="):
                match = _CONTIG_RE.search(line)
                if match:
                    contigs.append(match.group(1))
    return contigs


def read_fai_contigs(path: Path) -> list[str]:
    contigs = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            contigs.append(line.split("\t", 1)[0])
    return contigs


def read_fasta_header(path: Path) -> str:
    with path.open("rt", encoding="utf-8", errors="replace") as handle:
        return handle.readline().strip()


def region_from_challenge_bed(directory: Path) -> str | None:
    """Invert ``generate_challenge_region_bed`` from an official score_run."""
    beds = sorted(
        (directory / "score_run").glob("challenge_region_*.bed"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    for bed in beds:
        for line in bed.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) < 3 or parts[0].startswith("#"):
                continue
            chrom, start_s, end_s = parts[0], parts[1], parts[2]
            region = f"{chrom}:{int(start_s) + 1}-{int(end_s)}"
            if validate_region(region)["valid"]:
                return region
    return None


def compatibility_errors(
    *,
    region: str | None,
    bam_contigs: list[str] | None,
    fai_contigs: list[str] | None,
    truth_contigs: list[str] | None,
    fasta_header: str | None,
    bam_exists: bool,
    bai_exists: bool,
    truth_exists: bool,
    truth_index_exists: bool,
    reference_exists: bool,
    fai_exists: bool,
    sdf_exists: bool,
    mutations_exists: bool,
) -> list[str]:
    """Pre-flight checks. These run before a container is started."""
    errors: list[str] = []
    if not region:
        errors.append(
            "region is unknown. Pass a manifest with region, or score the sample "
            "once through official practice mode so a challenge BED is recorded. "
            "The BAM read span is not used as the region."
        )
    else:
        checked = validate_region(region)
        if not checked["valid"]:
            errors.append(checked["error"] or "invalid region")
        else:
            chrom = region.split(":", 1)[0]
            if fai_contigs is not None and chrom not in fai_contigs:
                errors.append(
                    f"region contig {chrom} is not in the reference index "
                    f"{fai_contigs[:8]}"
                )
            if bam_contigs is not None and chrom not in bam_contigs:
                errors.append(
                    f"region contig {chrom} is not in the BAM header {bam_contigs}"
                )
            if truth_contigs and chrom not in truth_contigs:
                errors.append(
                    f"region contig {chrom} is not in the truth VCF contigs "
                    f"{truth_contigs[:8]}"
                )
    if fasta_header and _BUILD_BAD_RE.search(fasta_header):
        errors.append(
            f"reference header looks like GRCh37/hg19 ({fasta_header!r}); "
            "Minos practice data uses the chr-prefixed GRCh38 layout"
        )
    if not bam_exists:
        errors.append("BAM is missing")
    if not bai_exists:
        errors.append("BAM index (.bai) is missing")
    if not truth_exists:
        errors.append("truth VCF is missing")
    if not truth_index_exists:
        errors.append("truth VCF index (.tbi or .csi) is missing")
    if not reference_exists:
        errors.append("reference FASTA is missing")
    if not fai_exists:
        errors.append("reference index (.fai) is missing")
    if not sdf_exists:
        errors.append(
            "RTG SDF directory is missing. hap.py vcfeval refuses to score without it"
        )
    if not mutations_exists:
        errors.append(
            "mutations VCF is missing. Practice scoring and scoring v2 need it"
        )
    return errors


def _index_beside(path: Path, suffixes: tuple[str, ...]) -> Path | None:
    for suffix in suffixes:
        candidate = Path(str(path) + suffix)
        if candidate.exists():
            return candidate
    return None


def _load_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _apply_manifest(sample: PracticeSample, entry: dict) -> None:
    region = entry.get("region")
    chromosome = entry.get("chromosome")
    if chromosome:
        if not CHROMOSOME_PATTERN.match(str(chromosome)):
            raise PrepareError(f"{sample.sample_id}: chromosome {chromosome!r} is not allowlisted")
        sample.chromosome = str(chromosome)
    if region:
        checked = validate_region(str(region))
        if not checked["valid"]:
            raise PrepareError(f"{sample.sample_id}: {checked['error']}")
        sample.region = str(region)
        sample.region_source = "manifest"
        sample.chromosome = sample.region.split(":", 1)[0]


def discover_samples(
    settings,
    *,
    hash_files: bool = True,
    manifest: dict[str, dict] | None = None,
    extra_ids: list[str] | None = None,
) -> list[PracticeSample]:
    practice = settings.practice_dir()
    known_ids = read_sample_ids(settings.sample_id_file)
    for sample_id in extra_ids or []:
        if sample_id.lower() not in known_ids:
            known_ids.append(sample_id.lower())
    id_by_dir = {safe_round_dir_name(sample_id): sample_id for sample_id in known_ids}
    if not practice.exists():
        return []
    samples: list[PracticeSample] = []
    for directory in sorted(item for item in practice.iterdir() if item.is_dir()):
        bam = directory / "input.bam"
        truth = directory / "truth.vcf.gz"
        if not bam.exists() and not truth.exists():
            continue
        sidecar = _load_json(directory / "sample.json")
        sample_id = sidecar.get("sample_id") or id_by_dir.get(directory.name)
        if not sample_id:
            sample_id = f"unidentified:{directory.name}"
        sample = PracticeSample(sample_id=str(sample_id), directory=str(directory))
        if manifest and sample.sample_id in manifest:
            _apply_manifest(sample, manifest[sample.sample_id])
        elif sidecar.get("region"):
            checked = validate_region(str(sidecar["region"]))
            if checked["valid"]:
                sample.region = str(sidecar["region"])
                sample.region_source = sidecar.get("region_source") or "sample.json"
                sample.chromosome = sample.region.split(":", 1)[0]
        if sample.region is None:
            inferred = region_from_challenge_bed(directory)
            if inferred:
                sample.region = inferred
                sample.region_source = "official challenge BED"
                sample.chromosome = inferred.split(":", 1)[0]
        _fill_paths(settings, sample, directory, hash_files=hash_files, sidecar=sidecar)
        samples.append(sample)
    return samples


def _fill_paths(settings, sample: PracticeSample, directory: Path, *, hash_files: bool, sidecar: dict) -> None:
    sample.reference = None
    sample.reference_index = None
    sample.sdf = None
    sample.reference_header = None
    bam = directory / "input.bam"
    truth = directory / "truth.vcf.gz"
    if not truth.exists():
        alt = directory / "truth.vcf"
        truth = alt if alt.exists() else truth
    mutations = directory / "mutations.vcf.gz"
    sample.bam = str(bam) if bam.exists() else None
    bai = _index_beside(bam, (".bai",)) if bam.exists() else None
    sample.bai = str(bai) if bai else None
    sample.truth = str(truth) if truth.exists() else None
    truth_index = None
    if truth.exists():
        truth_index = _index_beside(truth, (".tbi", ".csi"))
    sample.truth_index = str(truth_index) if truth_index else None
    sample.mutations = str(mutations) if mutations.exists() else None
    mut_index = _index_beside(mutations, (".tbi", ".csi")) if mutations.exists() else None
    sample.mutations_index = str(mut_index) if mut_index else None

    chrom = sample.chromosome
    if chrom is None and truth.exists():
        try:
            contigs = read_vcf_contigs(truth)
        except OSError:
            contigs = []
        if len(contigs) == 1 and CHROMOSOME_PATTERN.match(contigs[0]):
            chrom = contigs[0]
            sample.chromosome = chrom
    reference = None
    fai = None
    sdf = None
    if chrom and CHROMOSOME_PATTERN.match(chrom):
        reference = settings.reference_dir() / chrom / f"{chrom}.fa"
        fai_candidate = Path(str(reference) + ".fai")
        sdf_candidate = settings.reference_dir() / chrom / f"{chrom}.sdf"
        if reference.exists():
            sample.reference = str(reference)
            try:
                sample.reference_header = read_fasta_header(reference)
            except OSError:
                sample.reference_header = None
        if fai_candidate.exists():
            sample.reference_index = str(fai_candidate)
            fai = fai_candidate
        if sdf_candidate.is_dir():
            sample.sdf = str(sdf_candidate)
            sdf = sdf_candidate

    bam_contigs = None
    fai_contigs = None
    truth_contigs = None
    try:
        if bam.exists():
            bam_contigs = read_bam_contigs(bam)
    except (OSError, PrepareError, EOFError, gzip.BadGzipFile) as exc:
        sample.blockers.append(f"BAM header: {exc}")
    try:
        if fai and fai.exists():
            fai_contigs = read_fai_contigs(fai)
    except OSError as exc:
        sample.blockers.append(f"reference index: {exc}")
    try:
        if truth.exists():
            truth_contigs = read_vcf_contigs(truth)
    except OSError as exc:
        sample.blockers.append(f"truth VCF: {exc}")

    sample.blockers.extend(
        compatibility_errors(
            region=sample.region,
            bam_contigs=bam_contigs,
            fai_contigs=fai_contigs,
            truth_contigs=truth_contigs,
            fasta_header=sample.reference_header,
            bam_exists=bam.exists(),
            bai_exists=bool(sample.bai),
            truth_exists=truth.exists(),
            truth_index_exists=bool(sample.truth_index),
            reference_exists=bool(sample.reference),
            fai_exists=bool(sample.reference_index),
            sdf_exists=sdf is not None and sdf.is_dir(),
            mutations_exists=mutations.exists(),
        )
    )
    # A header error is already in blockers; drop duplicates while keeping order.
    deduped = []
    for item in sample.blockers:
        if item not in deduped:
            deduped.append(item)
    sample.blockers = deduped
    sample.ready = not sample.blockers
    if hash_files and sample.ready:
        previous = {
            key: FileStamp.model_validate(value)
            for key, value in (sidecar.get("stamps") or {}).items()
            if isinstance(value, dict)
        }
        sample.stamps = _stamp_inputs(sample, previous)


def _stamp_inputs(sample: PracticeSample, previous: dict[str, FileStamp]) -> dict[str, FileStamp]:
    stamps: dict[str, FileStamp] = {}
    mapping = {
        "bam": sample.bam,
        "bai": sample.bai,
        "truth": sample.truth,
        "truth_index": sample.truth_index,
        "mutations": sample.mutations,
        "reference": sample.reference,
        "reference_index": sample.reference_index,
    }
    for key, value in mapping.items():
        if value:
            stamps[key] = stamp_file(Path(value), previous.get(key))
    if sample.sdf:
        sdf_path = Path(sample.sdf)
        prior = previous.get("sdf")
        current_marker = _tree_marker(sdf_path)
        if prior and prior.path == current_marker:
            stamps["sdf"] = prior
        else:
            stamps["sdf"] = FileStamp(
                path=current_marker,
                sha256=hash_tree(sdf_path),
                size=sum(item.stat().st_size for item in sdf_path.rglob("*") if item.is_file()),
                mtime_ns=sdf_path.stat().st_mtime_ns,
            )
    return stamps


def _tree_marker(path: Path) -> str:
    files = [item for item in path.rglob("*") if item.is_file()]
    newest = max((item.stat().st_mtime_ns for item in files), default=0)
    return f"{path}|{len(files)}|{newest}"


def input_hashes(sample: PracticeSample, *, refresh: bool = True) -> dict[str, str | None]:
    """Checksums that affect a score. Refresh when size or mtime changed."""
    if refresh and sample.ready:
        sample.stamps = _stamp_inputs(sample, sample.stamps)
    return {key: stamp.sha256 for key, stamp in sample.stamps.items()}


def load_manifest(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.get("samples") if isinstance(data, dict) else data
    if not isinstance(entries, list):
        raise PrepareError(f"{path} must be a list or an object with 'samples'")
    manifest = {}
    for entry in entries:
        if not isinstance(entry, dict) or "sample_id" not in entry:
            raise PrepareError(f"{path} has an entry without sample_id")
        manifest[str(entry["sample_id"]).lower()] = entry
    return manifest


def write_catalog(settings, samples: list[PracticeSample]) -> Path:
    path = settings.catalog_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "samples": [sample.model_dump() for sample in samples],
    }
    _atomic_json(path, payload)
    for sample in samples:
        if sample.sample_id.startswith("unidentified:"):
            continue
        sidecar = Path(sample.directory) / "sample.json"
        _atomic_json(sidecar, sample.model_dump())
    return path


def load_catalog(settings) -> list[PracticeSample]:
    path = settings.catalog_path()
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [PracticeSample.model_validate(item) for item in data.get("samples", [])]


def apply_region_override(settings, sample: PracticeSample, region: str) -> PracticeSample:
    """Set a region the operator supplied. The value is still whitelist-checked."""
    checked = validate_region(region)
    if not checked["valid"]:
        raise PrepareError(checked["error"] or f"invalid region {region!r}")
    sample.region = region
    sample.region_source = "cli"
    sample.chromosome = region.split(":", 1)[0]
    sample.blockers = []
    _fill_paths(
        settings,
        sample,
        Path(sample.directory),
        hash_files=True,
        sidecar=_load_json(Path(sample.directory) / "sample.json"),
    )
    return sample


def resolve_sample(settings, sample_id: str) -> PracticeSample:
    wanted = sample_id.strip().lower()
    if CHROMOSOME_PATTERN.match(wanted):
        raise PrepareError(
            f"{sample_id} is a chromosome name, not a practice sample id. "
            "Pass a sample UUID. prepare --samples chr20 filters by chromosome."
        )
    if not UUID_RE.match(wanted):
        raise PrepareError(
            f"{sample_id!r} is not a practice sample UUID and not a chromosome name."
        )
    samples = discover_samples(settings, hash_files=True, extra_ids=[wanted])
    match = next((item for item in samples if item.sample_id.lower() == wanted), None)
    if match is None:
        folder = safe_round_dir_name(wanted)
        raise PrepareError(
            f"Sample {wanted} is not staged. Expected files in "
            f"{settings.practice_dir() / folder}/ "
            "(input.bam, input.bam.bai, truth.vcf.gz, truth.vcf.gz.tbi, mutations.vcf.gz). "
            "Set PLATFORM_URL and rerun prepare to download it, or stage those files. "
            "This tuner does not submit a live round."
        )
    if not match.ready:
        raise PrepareError(
            f"Sample {wanted} is staged but not runnable: " + "; ".join(match.blockers)
        )
    return match


def prepare_datasets(
    settings,
    tokens: list[str] | None,
    *,
    dry_run: bool = False,
    manifest_path: Path | None = None,
) -> PrepareResult:
    manifest = load_manifest(manifest_path)
    chromosomes, sample_ids, rejected = classify_tokens(tokens or [])
    if rejected:
        raise PrepareError(
            "Not a chromosome (chr1-22, chrX, chrY, chrM) and not a sample UUID: "
            + ", ".join(rejected)
        )
    samples = discover_samples(
        settings,
        hash_files=not dry_run,
        manifest=manifest,
        extra_ids=sorted(sample_ids),
    )
    messages: list[str] = []
    if not sample_ids and not chromosomes:
        samples, catalog_notes = _stage_catalog(
            settings,
            samples,
            manifest=manifest,
            dry_run=dry_run,
        )
        messages.extend(catalog_notes)
    if sample_ids:
        staged = {item.sample_id.lower() for item in samples}
        missing = sorted(sample_ids - staged)
        if missing and not dry_run:
            fetched, notes = _fetch_missing(settings, missing)
            messages.extend(notes)
            if fetched:
                samples = discover_samples(
                    settings,
                    hash_files=True,
                    manifest=manifest,
                    extra_ids=sorted(sample_ids),
                )
        elif missing and dry_run:
            for sample_id in missing:
                messages.append(
                    f"dry-run: {sample_id} is not staged at "
                    f"{settings.practice_dir() / safe_round_dir_name(sample_id)}"
                )
    selected = _filter_samples(samples, chromosomes, sample_ids)
    if chromosomes:
        staged_chroms = sorted({item.chromosome for item in samples if item.chromosome})
        for chrom in sorted(chromosomes):
            count = sum(1 for item in samples if item.chromosome == chrom)
            if count == 0:
                messages.append(
                    f"{chrom} is a chromosome filter and matched no staged practice sample. "
                    f"Staged chromosomes: {staged_chroms or '(none)'}."
                )
            else:
                messages.append(f"{chrom} matched {count} staged sample(s).")
    if not dry_run:
        write_catalog(settings, samples)
    ready = [item for item in selected if item.ready]
    result = PrepareResult(samples=selected, messages=messages, ready_count=len(ready))
    return result


def _stage_catalog(settings, samples, *, manifest, dry_run: bool):
    """Download catalog ids that are not on disk when no filter was requested."""
    catalog_ids = read_sample_ids(settings.sample_id_file)
    if not catalog_ids:
        return samples, []
    staged = {item.sample_id.lower() for item in samples}
    missing = [sample_id for sample_id in catalog_ids if sample_id not in staged]
    if not missing:
        return samples, [f"catalog {len(catalog_ids)} ids, all staged"]
    if dry_run:
        return samples, [
            f"dry-run: {len(missing)} of {len(catalog_ids)} catalog ids are not staged"
        ]
    if not os.environ.get("PLATFORM_URL", "").strip():
        return samples, [
            f"{len(missing)} of {len(catalog_ids)} ids in the sample catalog are not staged. "
            "Set PLATFORM_URL and rerun prepare with no --samples to download them."
        ]
    fetched, notes = _fetch_missing(settings, missing)
    if fetched:
        samples = discover_samples(
            settings,
            hash_files=True,
            manifest=manifest,
            extra_ids=missing,
        )
    return samples, notes


def _filter_samples(
    samples: list[PracticeSample],
    chromosomes: set[str],
    sample_ids: set[str],
) -> list[PracticeSample]:
    if not chromosomes and not sample_ids:
        return samples
    chosen = []
    for sample in samples:
        if sample.sample_id.lower() in sample_ids:
            chosen.append(sample)
        elif sample.chromosome in chromosomes:
            chosen.append(sample)
    return chosen


def _fetch_missing(settings, sample_ids: list[str]) -> tuple[bool, list[str]]:
    """Download specific practice samples. Never calls a submit endpoint."""
    platform_url = os.environ.get("PLATFORM_URL", "").strip()
    if not platform_url:
        notes = []
        for sample_id in sample_ids:
            folder = settings.practice_dir() / safe_round_dir_name(sample_id)
            notes.append(
                f"{sample_id} is not staged. Expected directory: {folder}. "
                "Set PLATFORM_URL and rerun prepare to download the practice BAM, "
                "truth VCF, and mutations VCF. This tuner does not submit a live round."
            )
        return False, notes
    try:
        import asyncio
        import secrets

        from bittensor_wallet import Keypair

        from utils.platform_client import MinerPlatformClient, PlatformConfig, PlatformClientError
    except ImportError as exc:
        return False, [f"Cannot reach the practice API: {exc}"]

    async def _run() -> list[str]:
        from neurons.miner import _download_practice_files

        keypair = Keypair.create_from_uri(f"//practice-{secrets.token_hex(4)}")
        client = MinerPlatformClient(keypair, PlatformConfig(base_url=platform_url))
        notes: list[str] = []
        for sample_id in sample_ids:
            try:
                full = await client.get_practice_sample(sample_id)
            except PlatformClientError as exc:
                notes.append(f"{sample_id}: practice API error: {exc}")
                continue
            files = _download_practice_files(full)
            if not files:
                notes.append(f"{sample_id}: download failed")
                continue
            region = full.get("region")
            chromosome = full.get("chromosome")
            sidecar = {
                "sample_id": sample_id,
                "region": region,
                "chromosome": chromosome,
                "region_source": "platform",
            }
            # Drop presigned URLs. They are temporary credentials.
            meta = {
                key: value
                for key, value in full.items()
                if "url" not in key.lower()
            }
            sidecar["platform_metadata"] = meta
            path = Path(files["dir"]) / "sample.json"
            _atomic_json(path, sidecar)
            notes.append(f"{sample_id}: downloaded into {files['dir']}")
        return notes

    try:
        return True, asyncio.run(_run())
    except Exception as exc:  # noqa: BLE001 - surface the setup blocker
        return False, [f"Practice download failed: {exc}"]


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def public_metadata(sample: dict) -> dict:
    """Drop presigned URLs before anything is written to disk."""
    return {key: value for key, value in sample.items() if "url" not in str(key).lower()}

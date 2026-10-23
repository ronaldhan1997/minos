# Minos GATK tuner

Local search over GATK HaplotypeCaller settings for Minos practice samples.
Scores come from `HappyScorer` and `AdvancedScorer` in this repository.
The tuner does not register a hotkey, open a wallet, or submit a live round.

The Minos README stays in the parent checkout. Integration notes and the
inspected commit are in `docs/UPSTREAM_INTEGRATION.md`.

Run these commands from `minos_tuning/`. The default Minos checkout is the
parent directory. Override it with `MINOS_REPO_ROOT`.

## Install

Use Python 3.11 or newer. On this machine that is the Minos virtualenv
(`python3` on the host is 3.10).

```bash
../.venv/bin/python -m pip install -r requirements.txt
../.venv/bin/python -m tuner.cli doctor
```

`doctor` exits 1 when Docker, the GATK image, the hap.py image, or the official
scorer cannot be imported. A dirty git tree is a warning: the commit and the
dirty flag are stored with every score.

Pull the images the templates already name if `doctor` says they are missing:

```bash
docker pull broadinstitute/gatk:4.5.0.0
docker pull quay.io/biocontainers/bcftools:1.20--h8b25389_0
```

hap.py is pinned by digest in `utils/scoring.py`. `doctor` prints that reference.

## Data

Practice sample ids are UUIDs. `chr20` is a chromosome filter.

```bash
../.venv/bin/python -m tuner.cli prepare
../.venv/bin/python -m tuner.cli prepare --samples chr20,chr21,chr22
```

Staged files are reused from `datasets/practice/<safe_round_dir_name(sample_id)>/`.
A missing UUID is downloaded only when `PLATFORM_URL` is set. That path uses an
ephemeral practice key and the official download helper. It does not call the
live submit endpoint.

Each sample needs a BAM and `.bai`, truth VCF and index, mutations VCF, a
region, and `datasets/reference/<chrom>/<chrom>.fa` plus `.fai` and `<chrom>.sdf`.
The region comes from `sample.json`, a manifest, or the challenge BED left by
an official practice run:

```json
{"samples": [{"sample_id": "SAMPLE_UUID", "region": "chr20:13522546-15723106"}]}
```

```bash
../.venv/bin/python -m tuner.cli prepare --manifest manifest.json
```

A sample whose region was never recorded stays blocked. The read span of the
BAM is not substituted for that region.

Disk: each practice BAM here is about 60–200 MB. A study keeps a VCF and hap.py
output per trial under `experiments/<study>/work/`. A long study can use tens
of gigabytes. The SQLite logs are the part to keep if you delete `work/`.

## Commands

```bash
../.venv/bin/python -m tuner.cli doctor
../.venv/bin/python -m tuner.cli prepare --samples chr20,chr21,chr22
../.venv/bin/python -m tuner.cli test --config configs/baseline.conf --sample SAMPLE_ID
../.venv/bin/python -m tuner.cli tune --trials 50 --workers 1 --study gatk-v1
../.venv/bin/python -m tuner.cli validate --study gatk-v1 --top 5 --samples SAMPLE_A,SAMPLE_B
../.venv/bin/python -m tuner.cli report --study gatk-v1 --output experiments/report.csv
../.venv/bin/python -m tuner.cli export --study gatk-v1 --output configs/best/gatk.conf
```

`--dry-run` on `prepare`, `test`, `tune`, `validate`, and `export` checks
inputs and writes no official score. Dry-run trials live in
`experiments/<study>/dry-run/` so they are not resumed as real results.

Defaults are one worker, one GATK thread, 4 GB container memory, and a 3600s
timeout. Each extra worker holds another container of that size. The process
refuses `workers * threads` above the CPU count unless `--allow-oversubscribe`
is set. One thread is the setting used in the cache key for repeatability;
GATK with more threads can break ties differently.

`prepare` with no `--samples` reads `configs/sample_id.md`. When `PLATFORM_URL`
is set it downloads every listed id that is not already staged. A chromosome
filter does not do that download. `tune` with no sample UUIDs refuses to start
when that catalog still has ids with no local BAM. Pass the ids that should
participate and those ids are the whole study:

```bash
../.venv/bin/python -m tuner.cli tune --trials 20 --workers 1 --study gatk-one \
  --samples SAMPLE_UUID
```

One id means every trial scores that BAM only, and there is no held-out sample.
Two or more ids keep the usual holdout. `--allow-partial-catalog` is the other
way to search whatever is already staged. The finished `gatk-v2` study used
that partial path: 4 runnable ids out of 75.

The first `tune` writes `experiments/<study>/study.json`. With samples on two
or more chromosomes, the last id on each chromosome is held out. A single
chromosome keeps the lexicographic tail (three ids when at least five samples
are ready). Later runs of the same study keep that split. A different scoring
version or search-space version needs a new `--study` name. `core-v2` widened
the false-positive filters after `core-v1` ran into its own ceilings, so
`gatk-v2` does not resume onto the new box. v1 (`Advanced`) and v2
(`AdvancedV2`) are different scales. The tune summary prints `best_adv_score`,
which is the official 0–100 mean divided by 100. That is the leaderboard
ADV. SCORE. 0.85 there is an official mean of 85.

`--space advanced` adds `min_dangling_branch_length`,
`recover_all_dangling_branches`, and `pruning_lod_threshold` inside their
official ranges. Use a new study name. The report shows the official metrics
that came back; it does not claim an indel-F1 effect.

Optuna stores the study in SQLite. A second `tune` with the same name continues
until `--trials` finished trials exist. Failures, timeouts, and scores a
validator would discard are stored and marked failed in Optuna. They are not
given a stand-in number. An identical success is reused when the config, file
checksums, commit, image id, thread count, and scoring version all match.

`validate` scores the baseline and the top training configs on the held-out
ids. `mean - 0.5 * std` only orders those tuned configs. Export keeps the
baseline when the tuned leader is lower on any held-out sample, does not raise
the mean, or lowers the minimum. `--force` can export the training-best config
when validate has not been run, or the tuned leader when that gate kept the
baseline. `--force` does not skip parameter validation, does not export a
config with no successful official score, and does not overwrite
`configs/gatk.conf`.

## Example session

```bash
../.venv/bin/python -m tuner.cli doctor
../.venv/bin/python -m tuner.cli prepare --samples chr20
../.venv/bin/python -m tuner.cli tune --trials 3 --workers 1 --study gatk-v1 --dry-run --samples chr20
../.venv/bin/python -m tuner.cli tune --trials 20 --workers 1 --study gatk-v1 --samples chr20
../.venv/bin/python -m tuner.cli validate --study gatk-v1 --top 5
../.venv/bin/python -m tuner.cli report --study gatk-v1 --output experiments/gatk-v1-report.csv
../.venv/bin/python -m tuner.cli export --study gatk-v1 --output configs/best/gatk.conf
../.venv/bin/python -m tuner.cli test --config configs/best/gatk.conf --sample SAMPLE_ID
```

Replace `SAMPLE_ID` with a UUID from `prepare`. The public demo id in
`neurons/miner.py` is `d7b99f4a-061c-4202-8b12-2c1425bd4974`. A real GATK window
on these BAMs is a long job; start with a small `--trials` value.

## Tests

```bash
../.venv/bin/python -m pytest tests/tuner -m "not integration"
```

Docker and the staged practice BAMs:

```bash
MINOS_TUNER_INTEGRATION=1 ../.venv/bin/python -m pytest tests/tuner/test_integration_optional.py -m integration
```

One official baseline score (GATK, then hap.py, then AdvancedScorer):

```bash
MINOS_TUNER_INTEGRATION=1 MINOS_TUNER_RUN_GATK=1 \
  ../.venv/bin/python -m pytest tests/tuner/test_integration_optional.py -m integration
```

## Copying a config onto the miner

Export writes `configs/best/gatk.conf` and `configs/best/gatk.conf.provenance.json`.
Re-score that file with `test`. Copy it to `configs/gatk.conf` only after that
score is the one you want the miner to submit. Starting the miner is a separate
step; this package will not do it.

## Troubleshooting

- `doctor` says pysam is missing: `../.venv/bin/python -m pip install -r requirements.txt`.
  Scoring v2 reads the hap.py VCF through pysam. Without it the official score
  stays null; the tuner does not substitute the v1 number.
- `doctor` says the daemon is down: start Docker, then rerun `doctor`.
- A sample is blocked with no region: pass `--manifest` or run official
  `python neurons/miner.py --practice --sample_id SAMPLE_ID` once so the
  challenge BED exists. Then `prepare` again.
- `PLATFORM_URL` is unset and the UUID is not on disk: stage the BAM, index,
  truth VCF, and mutations VCF under the directory `prepare` prints.
- `Study ... already has scoring version`: the state file
  `~/.minos/scoring_version.json` changed. Keep the old study for the old scale
  and pass a new `--study` name.
- Timeouts are status `timeout` with a null score. Raise `--timeout` or
  `--memory-gb`. The default worker count stays 1.
- Export refuses `configs/gatk.conf` on purpose.

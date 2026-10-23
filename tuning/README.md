# Practice-mode GATK tuner

This package tunes GATK HaplotypeCaller by running Minos practice mode and
reading the `COMBINED_FINAL` line that practice prints. It does not submit a
live round, and it does not write Minos `configs/gatk.conf`.

Run these commands from `minos_tuning/`, or from the Minos checkout root.
`python -m tuning.cli` works in both places. Minos is the parent directory
unless `MINOS_REPO_ROOT` is set. A relative sample file is read from the
current directory, then from `minos_tuning/`.

The `tuner/` package is separate. It calls the GATK template directly. This
package calls `start-miner.sh` because that is the path whose printed score
is what a validator records.

## Install

Use the Minos virtualenv, which already has Optuna:

```bash
../.venv/bin/python -m pip install -r requirements.txt
../.venv/bin/python -m pytest tests/tuning -q
```

## What a trial is

`tune` is an exploratory search: **40 trials, 8 parameters, one sample**.
It is not the 100-trial phase 1 plus 400-trial phase 2 campaign. `pipeline`
prints that campaign and does not start GATK unless you pass `--execute`
and `--scoring-version`.

Phase 1 searches these 8 high-importance parameters. The other 17 stay at
the baseline, and all 25 keys are still written:

- `min_base_quality_score`
- `min_mapping_quality_score`
- `base_quality_score_threshold`
- `standard_min_confidence_threshold_for_calling`
- `pcr_indel_model`
- `min_pruning`
- `active_probability_threshold`
- `max_reads_per_alignment_start`

Phase 2 keeps those eight and also searches:

- `min_dangling_branch_length` (2–10, baseline 4)
- `max_num_haplotypes_in_population` (64–256, baseline 128)
- `assembly_region_padding` (50–200, baseline 100)
- `pair_hmm_gap_continuation_penalty` (5–20, baseline 10)
- `phred_scaled_global_read_mismapping_rate` (30–60, baseline 45)
- `heterozygosity` (1e-4–1e-2, log scale, baseline 0.001)
- `indel_heterozygosity` (1e-5–1e-3, log scale, baseline 0.000125)

Those seven ranges are exploration bounds inside the official GATK whitelist,
not a claim that the edges are optimal. Phase 2 is seeded with the best
distinct successful phase 1 configs. On those seeds the seven new fields stay
at the baseline. Failed and incomplete phase 1 records are not seeded.

These three stay at the baseline in every config:

- `emit_ref_confidence=NONE`
- `sample_ploidy=2`
- `contamination_fraction_to_filter=0.0`

The default budget is 40 trials. `--trials` is the total for the run. Running
the command again continues until that total exists. A replay of the baseline
parameter point is trial 1 and does count.

The objective, fixed before any trial, is:

```text
mean(combined_final) - 0.25 * population_std(combined_final)
```

Change the penalty with `--robustness-penalty`. A trial counts only when
every training sample prints a validator-recordable score. A failure is an
Optuna fail. It is not stored as zero.

`--workers` must be 1. Practice mode writes `score_run/` next to the sample
BAM, so two evaluations would share that directory. Every campaign in one
Minos checkout takes the same lock, `datasets/practice/.tuner-practice.lock`,
so two different `--run-dir` values cannot evaluate at the same time. A timed
out practice process is killed with its process group, and `minos-*` containers
that appeared during that run are removed before the lock is released.

Completed scores are reused only when the git commit is known and the tree is
clean. A dirty tree disables that cache, because the running code is not the
commit hash.

One training sample is allowed. The report marks it single-sample
exploratory. Phase 3 refuses to certify a result until at least three
validation ids are supplied, and those ids must not overlap training.

## Command

From this directory:

```bash
../.venv/bin/python -m tuning.cli tune
```

That checks the machine, then runs an **exploratory** 40-trial / 8-parameter
study on sample `2ea4a48e-ba7f-43d8-86e7-81ae5a99196f`. Results go to
`tuning_runs/my_run`. Running it again continues the same run. It does not
write `configs/gatk.conf`. Pass `--trials 80` to change the budget. A run
whose manifest was written by an older search space (`evaluator_version`
before `tuning-0.2.0`, or a different `phase2_spec_sha256`) cannot be resumed;
give it a new `--run-dir`.

To print the 100 + 400 campaign without starting GATK:

```bash
../.venv/bin/python -m tuning.cli pipeline \
  --run-dir tuning_runs/gatk_v2_01 \
  --train-samples tuning_train_samples.txt \
  --validation-samples tuning_validation_samples.txt
```

To train on sample ids you choose:

```bash
../.venv/bin/python -m tuning.cli tune --samples id1,id2
```

To score the best configs on held-out ids and export a promoted file, add a
text file with at least three other catalog UUIDs:

```bash
../.venv/bin/python -m tuning.cli tune --validation-samples tuning_validation_samples.txt
```

A one-sample run is exploratory. The export happens only when the held-out
gate passes.

## The same steps, separately

Put one sample UUID per line in the sample files. The initial training id
from the plan is `2ea4a48e-ba7f-43d8-86e7-81ae5a99196f`. Validation ids are
chosen by the operator and must not include any training id.

```bash
../.venv/bin/python -m tuning.cli preflight \
  --baseline configs/gatk.conf \
  --sample-id 2ea4a48e-ba7f-43d8-86e7-81ae5a99196f

../.venv/bin/python -m tuning.cli optimize \
  --phase 1 \
  --trials 100 \
  --baseline configs/gatk.conf \
  --train-samples tuning_train_samples.txt \
  --run-dir tuning_runs/my_run \
  --seed 42 \
  --workers 1

../.venv/bin/python -m tuning.cli optimize \
  --phase 2 \
  --trials 400 \
  --run-dir tuning_runs/my_run \
  --workers 1 \
  --resume

../.venv/bin/python -m tuning.cli validate \
  --run-dir tuning_runs/my_run \
  --validation-samples tuning_validation_samples.txt \
  --finalists 5

../.venv/bin/python -m tuning.cli report --run-dir tuning_runs/my_run

../.venv/bin/python -m tuning.cli export \
  --run-dir tuning_runs/my_run \
  --output tuning_runs/my_run/final/best_validated_gatk.conf
```

`tuning_train_samples.txt` starts with `2ea4a48e-ba7f-43d8-86e7-81ae5a99196f`.
Add more training UUIDs from the practice catalog on their own lines.
`tuning_validation_samples.txt` needs at least three other catalog UUIDs.
Do not repeat a training id there.

The first `optimize` reads `/scoring/network-config` and pins that version.
If the endpoint does not answer, the command stops. Re-run with
`--scoring-version` set to the version you confirmed. Phase 2 reuses the
pin stored in the run manifest.

`--dry-run` writes one validated config and prints the practice command. It
does not start GATK and does not consume the trial budget.

Omit `--scoring-version` only when `https://api.theminos.ai/scoring/network-config`
answers. If it does not, the command stops. A cached version is not used.

Export writes a file only after Phase 3 records `PROMOTED`. A failed gate
stays `NOT_PROMOTED` and is not sent back to Optuna. An uncertified copy
requires `--exploratory` and a filename containing `exploratory`.

## Outputs

`tuning_runs/` is gitignored. A run directory holds `manifest.json`,
one shared `optuna.sqlite3` (study names `phase1` and `phase2`),
`trial_results.jsonl`, and one artifact directory per phase:

```text
phase1/trials/000000/{gatk.conf,trial.json,...}
phase2/trials/000000/{gatk.conf,trial.json,...}
```

Phase 1 trial 0 and phase 2 trial 0 no longer share a path. The manifest
stores the sha256 of `configs/gatk.conf`. Every phase checks that hash and
stops if the miner config changed. `validate` checks it again before ranking.

## Existing runs

Runs created before phase-scoped trial directories kept every trial under
`trials/000000`. Phase 2 could overwrite phase 1's trial 0. Classify such a
directory before resuming it:

```bash
../.venv/bin/python -m tuning.cli migrate-legacy --run-dir tuning_runs/my_run
```

The command copies `trials/`, `trial_results.jsonl`, and `optuna.sqlite3`
into `legacy_archive/<timestamp>/` and leaves the original `trials/` tree in
place. It copies a trial into `phaseN/trials/` only when that trial number
belongs to one phase in both SQLite and JSONL. If both phases contain the
same trial number, the run is marked `untrusted` and nothing is merged.
Start a new `--run-dir` for an untrusted history. Do not delete `trials/`
yourself if you still need the archive.

A manifest written before `search_space_version` `phase2-15-v1` does not
resume. Phase 2 now tunes 15 parameters, so its scores are a different study.

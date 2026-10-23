# Practice-mode GATK tuner

This package tunes GATK HaplotypeCaller by running Minos practice mode and
reading the `COMBINED_FINAL` line that practice prints. It does not submit a
live round, and it does not write Minos `configs/gatk.conf`.

Run these commands from `minos_tuning/`. Minos is the parent directory unless
`MINOS_REPO_ROOT` is set.

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

`tune` searches these 8 high-importance parameters. The other 17 stay at the
baseline, and all 25 keys are still written:

- `min_base_quality_score`
- `min_mapping_quality_score`
- `base_quality_score_threshold`
- `standard_min_confidence_threshold_for_calling`
- `pcr_indel_model`
- `min_pruning`
- `active_probability_threshold`
- `max_reads_per_alignment_start`

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
BAM, so two evaluations would share that directory.

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

That checks the machine, then runs 40 trials of the 8 high-importance parameters
on sample `2ea4a48e-ba7f-43d8-86e7-81ae5a99196f`. Results go to `tuning_runs/my_run`.
Running it again continues the same run. It does not write `configs/gatk.conf`.
Pass `--trials 80` to change the budget. A run started with the old 15-parameter
search cannot be resumed; give it a new `--run-dir`.

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
`optuna.sqlite3`, `trial_results.jsonl`, one directory per trial, and the
phase report. The manifest stores the sha256 of `configs/gatk.conf`. Every
phase checks that hash and stops if the miner config changed.

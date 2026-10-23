# Example study layout

`tune` writes this tree under `experiments/<study>/`:

- `study.json` — training and held-out sample ids, scoring version, search-space version
- `evaluations.sqlite3` — one row per sample evaluation, including failures
- `optuna.sqlite3` — Optuna trial log
- `search_space.json` — the bounds that were actually searched
- `selection.json` — written by `validate`
- `work/` — per-trial VCF, GATK log, and hap.py output

`tune --dry-run` writes the trial database under `experiments/<study>/dry-run/` and does not invent scores. The split in `study.json` is shared with a later real run of the same study name.

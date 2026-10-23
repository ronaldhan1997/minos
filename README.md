# Minos GATK tuning

This project searches GATK HaplotypeCaller settings. It lives inside the Minos
checkout and calls that code. It does not register a hotkey, open a wallet, or
submit a live round.

```text
minos_subnet/                 miner, validator, datasets, configs/gatk.conf
minos_subnet/minos_tuning/    this project
```

`MINOS_REPO_ROOT` overrides the Minos path. The default is the parent
directory.

## Run

From this directory, using the Minos virtualenv:

```bash
../.venv/bin/python optuna_test/optimize.py
../.venv/bin/python -m tuner.cli doctor
../.venv/bin/python -m tuning.cli tune --dry-run
```

- `tuner/` scores with the official GATK template and AdvancedScorer. See `tuner/README.md`.
- `tuning/` scores by running `start-miner.sh --practice`. See `tuning/README.md`.
- `optuna_test/optimize.py` is the small Optuna example, `(x - 2) ** 2`.

Practice BAMs stay in `minos_subnet/datasets/`. Study logs stay here, under
`experiments/` and `tuning_runs/`.

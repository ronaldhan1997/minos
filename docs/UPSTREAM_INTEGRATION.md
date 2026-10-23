# Upstream integration

Tuner code in `tuner/` calls this repository. It does not reimplement
`AdvancedScorer`, hap.py parsing, or GATK flag validation.

## Pinned commit

Inspected commit: `4b3c0944b517d09be9b54eb8fa3158ae700b5d5f` (`main`).

`doctor` prints the current `HEAD` and warns when it differs, or when the
work tree is dirty. Every evaluation stores that commit, the dirty flag, the
GATK image id, and the scoring version. A change to any of those is a cache
miss. Scores from different scoring versions are not ranked together.

## What the official code actually does

| Concern | Official interface | Tuner behavior |
|---|---|---|
| Practice samples | UUIDs from `PlatformClient.list_practice_samples`. Files land in `datasets/practice/<safe_round_dir_name(id)>/` via `neurons.miner._download_practice_files`. | `prepare` indexes that directory. `chr20` is a chromosome filter, not a sample id. |
| Region | Platform metadata, then the challenge BED written by `generate_challenge_region_bed` (`start - 1` in the BED). | Recover the region from `sample.json`, a manifest, or that BED. The BAM read span is not treated as the region. |
| Config | `key=value` parsed by `utils.config_loader._parse_value`. | Same parser. Duplicate keys are rejected here; the miner keeps the last one. |
| Validation | `templates.tool_params.validate_and_build_flags("gatk", options)`. One bad key rejects the whole config. Values are not clamped. | Called before every run and again before export. |
| GATK | `templates.gatk.variant_call`. Image literal `broadinstitute/gatk:4.5.0.0`. Timeout, threads, and memory are template arguments, not quality parameters. | Called in `experiments/<study>/work/`. `configs/gatk.conf` is not modified. |
| Scoring | `HappyScorer.score_vcf` then `AdvancedScorer.compute_advanced_score` or `compute_score_v2`. v2 calls `parse_happy_vcf`, which imports pysam. | Same calls. hap.py `None` stays a null score. A missing pysam makes v2 null. v2 does not fall back to v1. |
| Version | `utils.scoring_version.resolve`. Labels are `Advanced` (v1) and `AdvancedV2` (v2). This machine's state file currently records `v2`. | No network fetch. Explicit `--scoring-version` or the state file, else v1. |
| Practice entrypoint | `neurons.miner._run_and_score` writes `datasets/practice/<id>/score_run/output.vcf.gz`. | Not called. It would overwrite the last practice run. The same scorer classes are used in an isolated directory. |
| Live submission | `submit_config` / chain commitment. | Not called. Download uses an ephemeral practice keypair only when `PLATFORM_URL` is set and a requested UUID is missing. |

## Differences from the prompt's assumed layout

- This project is `minos_subnet/minos_tuning/`. The tuner package is `tuner/`. `MINOS_REPO_ROOT` selects the checkout when the parent directory is not the Minos repo.
- Tuner dependencies are `requirements.txt` in this project. Minos keeps its own `setup.py`.
- The subnet `README.md` stays in the Minos checkout. Tuner install and the example session are in `tuner/README.md`.
- System `python3` on the build machine is 3.10.12. `python3.11` is 3.11.15 and is the interpreter to use. `doctor` treats anything older than 3.11 as not ready.
- There is no upstream `prepare` or `tune` command. Those names belong to this CLI.
- `neurons.miner._run_and_score` returns `combined_final` on a 0–1 scale. `AdvancedScorer` returns 0–100. The tuner stores both and Optuna maximizes the 0–100 official score, and only when every training sample would be recorded by a validator.
- Infrastructure keys (`threads`, `memory_gb`, `timeout`, `ref_build`, `num_threads`) are not legal quality parameters. They are tuner settings. The container default in the template is host CPUs and host RAM minus 1 GB, with a 1200s fallback inside `variant_call` when unset. This tuner defaults to 1 thread, 4 GB, and 3600 seconds so repeated runs are easier to compare and a practice window can finish. Threads are part of the cache key because Pair-HMM thread count can change ties.
- The core search box started as the prompt's hypothesis (`core-v1`) after checking it against `GATK_QUALITY_PARAMS` at this commit. A finished 20-trial study on that box reached the ceilings for confidence, base quality, and mapping quality while a training sample stayed at the 45-point overcall cap. `core-v2` raises those ceilings and `active_probability_threshold` / `min_pruning` toward the official maxima. `pcr_indel_model` is still limited to `NONE` and `CONSERVATIVE`. The advanced profile adds `min_dangling_branch_length`, `recover_all_dangling_branches`, and `pruning_lod_threshold` across their full official ranges. That is not a claim that the wider box reaches leaderboard ADV. SCORE 0.85.
- No intermediate official metric exists while HaplotypeCaller is running, so trials are not pruned.
- Held-out ids are the lexicographic tail (3 ids when at least 5 samples are ready). They are fixed in `study.json` on the first `tune` and are not passed to Optuna. `mean - 0.5 * std` only orders tuned candidates. The export gate uses official per-sample scores.

## Local practice files already on this machine

Five directories under `datasets/practice/` match ids in `configs/sample_id.md` via `safe_round_dir_name`. Four have an official challenge BED from a previous practice score. `cc0593ac-778c-4b81-b218-9ba165475947` (`round_2cbd02b9989fe6f7`) has reads on chr22 and no recorded region, so `prepare` reports it blocked until a manifest or the platform provides the region.

References and RTG SDFs for chr18–chr22 are under `datasets/reference/<chrom>/`. FASTA headers in this tree are `>chr20` with no explicit GRCh38 token. A header that says GRCh37 or hg19 is rejected. A chr-prefixed contig that matches the BAM, truth VCF, and `.fai` is accepted, and the run config still passes `ref_build=GRCh38` the way the miner does.

## Images

- GATK: `broadinstitute/gatk:4.5.0.0` (read from `templates/gatk.py`)
- hap.py: `utils.scoring.HAPPY_DOCKER_IMAGE` (digest-pinned)
- bcftools: `utils.scoring.BCFTOOLS_DOCKER_IMAGE` (truth slicing inside hap.py)

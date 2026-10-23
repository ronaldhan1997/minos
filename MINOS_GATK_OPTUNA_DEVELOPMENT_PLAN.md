# Minos SN107 — GATK HaplotypeCaller Optuna Tuning System

**Document type:** Cursor implementation plan / engineering specification  
**Status:** Ready for implementation; validate repository behavior before changing code  
**Project:** Minos (Bittensor Subnet 107)  
**Target tool:** GATK HaplotypeCaller  
**Optimization framework:** Optuna (Python)  
**Baseline configuration:** `configs/gatk.conf` (must remain unchanged)  
**Primary evaluation mode:** `bash start-miner.sh --practice`  
**Proposed budget:** Phase 1: **100 trials**; Phase 2: **400 additional trials**; Phase 3: **independent validation**

---

## 1. Executive summary

Build a reproducible, fault-tolerant, command-line tuning system that searches valid Minos GATK HaplotypeCaller configurations using Optuna, evaluates them using Minos's own practice mode, and exports the best **validated** configuration without editing the miner's active configuration.

The optimization target is the **actual validator-recordable score**, not a generic GATK metric. Minos's practice mode runs GATK, benchmarks its resulting VCF against truth using hap.py, and displays the selected AdvancedScorer result. The official result is the normalized `COMBINED_FINAL` score (generally in `(0, 1]` for recordable trials). **Scoring v1 and v2 are not comparable**: the tuner must pin and verify the active version for each study and fail closed when version verification is unavailable or inconsistent.

### Mandatory three-phase workflow

| Phase | Budget | Search scope | Primary outcome |
|---|---:|---|---|
| **1. Initial optimization** | **100 total trials** | 15 selected high-impact or plausibly high-impact parameters; other 10 values fixed to baseline | Identify promising settings and benchmark runtime/failure rate |
| **2. Expanded optimization** | **400 additional trials** | Expand to 22 tunable parameters, including interactions and broader/refined ranges; keep three safety/workflow parameters fixed | Select stable high-scoring candidates under the same scoring version |
| **3. Final validation** | No fixed Optuna trial budget; direct evaluation runs | No further parameter fitting; evaluate finalists and baseline on independent samples | Promote only a configuration with reproducible, meaningful out-of-sample performance |

**Budget interpretation:** 100 + 400 = **500 Optuna trials**, not 400 total. A separately executed baseline and final validation evaluations do **not** count toward those 500 optimization trials. Seed/replay trials within each study **do** count toward the specified phase budget.

### Non-negotiable protections

1. Do not modify or overwrite `configs/gatk.conf`, `.env`, deployed miner settings, or repository source files unless implementation of the *new tuner* requires new files.
2. Do not submit configurations to a live round, use a wallet, or initiate paid resubmissions. Use practice/offline evaluation only.
3. Never accept unsupported parameter names, wrong types, out-of-range values, missing scores, or validator-discarded results as successful trials.
4. Never use holdout samples to guide or refit Optuna.
5. Default to **one evaluation at a time** because Minos currently writes scoring artifacts into a fixed `score_run/` directory beside the sample BAM; concurrent evaluations of the same sample can conflict.
6. Store all trial settings, raw logs, scores, failure reasons, scoring-version metadata, source revision, and reproducible output artifacts.

---

## 2. Context and authoritative interfaces

### 2.1 Exact user-provided practice command

```bash
bash start-miner.sh --practice \
  --config configs/gatk.conf \
  --sample-id 2ea4a48e-ba7f-43d8-86e7-81ae5a99196f
```

The tuner must programmatically replace only the `--config` argument with a generated per-trial configuration path; the provided sample ID is the **initial training sample**, not an independent validation sample.

Example generated evaluation command:

```bash
bash start-miner.sh --practice \
  --config /absolute/path/to/tuning_runs/<run-id>/trials/000042/gatk.conf \
  --sample-id 2ea4a48e-ba7f-43d8-86e7-81ae5a99196f
```

Use `subprocess.Popen([...], cwd=<minos-repo-root>, shell=False)` or an equivalent safe subprocess API; do not interpolate arguments into a shell string. The `start-miner.sh` wrapper changes into its own repository directory, so generated configuration paths should be **absolute**.

### 2.2 Repository sources that Cursor must inspect before coding

| Purpose | Expected repository path | Required inspection |
|---|---|---|
| Practice CLI | `start-miner.sh` | Verify `--practice`, `--config`, `--sample-id` flag handling, setup steps, and process exit codes |
| Practice runner + printed score | `neurons/miner.py` | Inspect `run_practice`, `_run_and_score`, `_UI.result_panel`, scoring-version resolution, working/output directories |
| Permitted parameters, canonical types, bounds | `templates/tool_params.py` | Use `GATK_QUALITY_PARAMS` and `validate_and_build_flags()` as authoritative validation logic |
| Human-readable parameter documentation | `docs/parameter_ranges.md` | Cross-check all 25 provided values; use for explanation, not sole validation authority |
| Scoring algorithm and versions | `utils/scoring.py`, `docs/scoring.md` | Inspect v1/v2 computations and conditions where validators discard a result |
| Configuration parsing | `utils/config_loader.py` | Match Minos's `key=value`, boolean, integer, float and enum parsing |
| GATK execution | `templates/gatk.py` | Confirm option mapping and output expectations |
| Practice sample API | `utils/platform_client.py` | Discover available fully answered practice samples without hard-coding sample IDs |

**Repository references:**

- https://github.com/minos-protocol/minos_subnet
- https://github.com/minos-protocol/minos_subnet/blob/main/docs/parameter_ranges.md
- https://github.com/minos-protocol/minos_subnet/blob/main/docs/scoring.md
- https://github.com/minos-protocol/minos_subnet/blob/main/docs/tuning_guide.md
- https://github.com/minos-protocol/minos_subnet/blob/main/neurons/miner.py

At runtime, optionally compare the code-defined parameter whitelist with `GET https://api.theminos.ai/scoring/parameter-ranges`; treat the checked-out `templates/tool_params.py` used by the local validator-compatible code as the execution source of truth. Stop if the deployed network rules seem incompatible with the local checkout.

### 2.3 Scoring-version invariant

Before optimization, resolve `scoring_version` from `/scoring/network-config` and record:

- version (`v1` or `v2`), retrieval time, endpoint response/hash;
- Git commit and installed dependency versions;
- training/holdout sample manifests;
- base config SHA-256 and allowed-parameter schema hash.

At each practice evaluation, verify the output's reported scoring version where available and compare it with the study's pinned version. The Minos practice code may otherwise reuse a cached version, or fall back to v1 when connectivity is unavailable. **Do not silently compare scores obtained on different versions.** Abort/pause and require a new study if the version changes, rather than merging results.

**Score extraction policy:** Prefer a structured result from the existing local scoring code if one can be exposed through a small, well-tested, backward-compatible adapter. Otherwise parse the visible practice output. Parse **`COMBINED_FINAL`**, not the generic `ADVANCED SCORE` label alone. Strip ANSI sequences and Rich box artifacts; recognize both the plain and Rich output variants. Accept only if it explicitly indicates the result is validator-recordable. Reject `ZERO-INPUT`, `validator discards`, `out of range`, missing score, non-finite score, wrong version, and any failure exit status. Record any parsed SNP/indel precision, recall, F1, FP counts as diagnostic fields.

A `COMBINED_FINAL` number alone is insufficient proof of success: Minos can display a value for a discarded result.

---

## 3. Functional and non-functional requirements

### 3.1 Functional requirements

- Load baseline from `configs/gatk.conf`, independently snapshot its exact contents/hash, and parse all **25** known values.
- Generate a fresh `.conf` for each trial, including **all 25 parameters**, using baseline defaults for parameters not being tuned.
- Validate each candidate **before** launching GATK; validate both Minos range/type rules and cross-parameter constraints.
- Run a single practice evaluation or a configured training sample set; capture combined stdout/stderr and exit status.
- Extract a score with explicit success/discard/version classification; never fabricate missing metrics.
- Optimize a reproducible objective; keep raw per-sample scores in addition to the aggregate.
- Persist study, trial statuses, per-trial configs and output immediately, supporting restart/recovery.
- Report Phase 1 and Phase 2 results separately and generate a final comparison against an untouched baseline.
- Export the best candidate to a separate generated file. Never auto-deploy or overwrite the baseline.

### 3.2 Operational requirements

- Python 3.10+ and Optuna; use the Minos environment where possible.
- Linux/macOS command-line interface; Docker, GATK, hap.py and required reference/RTG SDF assets must be present.
- `n_jobs=1` by default; concurrency greater than one explicitly disabled until fully isolated GATK/scoring working directories are proven.
- A configurable outer subprocess timeout; failures safely terminate the process group and any owned child containers where possible.
- Disk and runtime budget awareness; keep config/log/metric artifacts while optionally deleting heavy per-trial VCFs only after diagnostics are preserved.
- Sanitized logs: no platform credentials, wallet secrets, environment dumps, or sensitive filesystem paths in public reports.
- Use stable random seed(s), deterministic configuration serialization, and atomic file writes.

---

## 4. Parameter registry and search space

### 4.1 All 25 Minos GATK parameters

The supplied `configs/gatk.conf` is the baseline. The following table specifies which parameters to optimize at each stage. **Proposed search bounds are engineering starting choices, not proven biological optima.** All must be checked against the current Minos whitelist.

#### Phase 1: Tune 15 parameters (100 trials)

| # | Parameter | Type | Baseline | Proposed Phase 1 search | Motivation |
|---:|---|---|---|---|---|
| 1 | `min_base_quality_score` | int | `10` | `5..30` | Base evidence filtering |
| 2 | `min_mapping_quality_score` | int | `20` | `10..40` | Read alignment confidence |
| 3 | `base_quality_score_threshold` | int | `18` | `10..30` | Base quality treatment |
| 4 | `standard_min_confidence_threshold_for_calling` | float | `30.0` | `10.0..50.0`, step `1.0` | Precision/recall balance |
| 5 | `pcr_indel_model` | categorical | `CONSERVATIVE` | `NONE`, `CONSERVATIVE` | Indel error assumptions; test PCR-free hypothesis |
| 6 | `min_pruning` | int | `2` | `1..5` | Assembly path sensitivity |
| 7 | `max_alternate_alleles` | int | `6` | `2..10` | Complex variant genotyping |
| 8 | `min_dangling_branch_length` | int | `4` | `1..12` | Branch recovery |
| 9 | `recover_all_dangling_branches` | bool | `false` | `false`, `true` | Assembly recovery |
| 10 | `max_num_haplotypes_in_population` | int | `128` | `64..256`, step `32` | Haplotype search capacity |
| 11 | `adaptive_pruning_initial_error_rate` | float | `0.001` | `1e-4..1e-2`, log scale | Pruning model assumption |
| 12 | `pruning_lod_threshold` | float | `2.302585` | `1.0..5.0` | Adaptive pruning strength |
| 13 | `active_probability_threshold` | float | `0.002` | `5e-4..1e-2`, log scale | Active-region detection |
| 14 | `pair_hmm_gap_continuation_penalty` | int | `10` | `5..20` | Indel likelihood calculation |
| 15 | `phred_scaled_global_read_mismapping_rate` | int | `45` | `25..60` | Read likelihood assumptions |

**Note:** Some flags can be inactive or interact conditionally, depending on the HaplotypeCaller code path. Cursor must inspect `templates/gatk.py` and GATK's actual option behavior; only tune effective parameters. If a flag is ignored in the current mode, either remove it from the active search space with an explicit note or make it conditional. Preserve the 100-trial phase budget.

#### Phase 2: Add 7 tunable parameters (400 additional trials)

| # | Parameter | Type | Baseline | Proposed Phase 2 search | Safeguard |
|---:|---|---|---|---|---|
| 16 | `min_assembly_region_size` | int | `50` | `25..100` | Must be <= max region size |
| 17 | `max_assembly_region_size` | int | `300` | `200..600` | Must be >= min region size |
| 18 | `assembly_region_padding` | int | `100` | `50..200` | Watch runtime/region edge effects |
| 19 | `heterozygosity` | float | `0.001` | `1e-4..1e-2`, log scale | Model prior; avoid single-sample overfit |
| 20 | `indel_heterozygosity` | float | `0.000125` | `1e-5..1e-3`, log scale | Indel prior; assess on diverse samples |
| 21 | `max_reads_per_alignment_start` | int | `50` | `25..200`, step `5` | Runtime/coverage tradeoff; `0` is allowed by Minos but excluded initially |
| 22 | `dont_use_soft_clipped_bases` | bool | `false` | `false`, `true` | Variant sensitivity near read ends |

Phase 2 also refines/revisits the original 15 parameters. The Phase 2 study must not assume a higher score from Phase 1 is directly comparable if the scoring version or evaluation sample set changed.

#### Keep fixed unless explicit data/workflow justification is established (3 parameters)

| # | Parameter | Fixed value | Reason |
|---:|---|---|---|
| 23 | `emit_ref_confidence` | `NONE` | Preserve standard practice-mode VCF output; GVCF/BP_RESOLUTION may change the pipeline's expected output semantics |
| 24 | `sample_ploidy` | `2` | Diploid baseline; only change with verified sample biology |
| 25 | `contamination_fraction_to_filter` | `0.0` | A contamination prior is not a generic performance knob; use data-derived evidence before adjusting |

**All 25 fields must still be written** to each trial config. The distinction is 15 tunable in Phase 1, 22 tunable in Phase 2, 3 fixed throughout. If a newly verified Minos constraint requires changing this plan, document why and keep the three-phase structure/budgets intact.

### 4.2 Serialization rules

- Exact `key=value` format with one field per line; no JSON as the primary practice configuration format.
- Boolean values rendered as lowercase `true`/`false`.
- Enum values rendered exactly as allowed uppercase strings.
- Integers must remain integers (`20`, **not** `20.0`).
- Floats serialized deterministically, with sufficient precision but no `NaN`/`inf`.
- Order fields consistently so candidate hashes are stable.
- Reject unknown/duplicate keys and non-finite numbers; reject parameter name injection or newline content.
- Preserve a separate, read-only snapshot of the full original baseline for reproducibility; never rewrite original comments in place.

### 4.3 Study-specific search-space separation

Use **two distinct Optuna studies** in the same SQLite database (or separate databases), one per phase. Phase 2 has new dimensions and possibly different ranges, so a separate study reduces ambiguity in search-space history and reporting.

Use seeded `TPESampler` with a configurable seed. `multivariate=True` (and optionally `group=True`) may be used after compatibility testing. For Phase 2, enqueue the top **5–10 distinct valid** Phase 1 configurations, filling newly enabled fields from baseline, and count those enqueued trials inside the 400. Avoid enqueuing literal duplicate configs without a documented reason. Keep baseline as an explicit control.

---

## 5. Objective function and evaluation policy

### 5.1 Evaluation data split

Use a manifest committed to the **experiment run**, not edited after viewing scores:

```yaml
train_samples:
  - 2ea4a48e-ba7f-43d8-86e7-81ae5a99196f
  # Add one or more different practice sample IDs BEFORE tuning.
validation_samples:
  # Required: independent practice sample IDs, never used by Optuna.
  # Populate from the live practice sample menu or practice API.
```

**Strong recommendation:** At least two or three distinct training samples and at least three independent validation samples, subject to available practice datasets and compute. Use different chromosomes/regions, mutation loads, and, where identifiable, different underlying donors. Do not claim biological independence based on sample IDs alone: inspect metadata; avoid overlapping samples or repeated copies of the same underlying BAM/truth data.

**Insufficient-data behavior:** The system can perform a single-sample smoke test or exploratory Phase 1 run using the supplied sample ID, but it must label the result **single-sample exploratory** and must not claim generalization. Phase 3 must refuse to certify any candidate until non-overlapping validation sample IDs are supplied.

### 5.2 Per-sample score

For candidate `c` and sample `s`, run practice and record `combined_final(c,s)` only when the output is validator-recordable and the score is finite and in `(0,1]`.

- `status=success`: valid recordable score, capture diagnostics.
- `status=invalid_config`: failed local Minos param validation; do not launch process.
- `status=execution_error`: process exit failure / Docker or GATK failure.
- `status=score_error`: missing or unreadable score / hap.py failure.
- `status=discarded`: validator would not record this score.
- `status=scoring_version_mismatch`: stop the study pending investigation.
- `status=timeout` / `status=interrupted`: controlled retry policy and persisted evidence.

Do not convert failed trials into fabricated scores. Use Optuna failed-state semantics for infrastructure failures; use a clearly documented treatment (e.g., a poor objective score or explicitly infeasible/invalid status) for genuinely poor biological configurations. Track both categories separately to avoid teaching the sampler that a Docker outage is an inferior hyperparameter setting.

### 5.3 Aggregate objective

Choose and freeze **one** Phase 1/2 objective before trials start. A reasonable initial choice is:

```text
sample_scores = [validator_recordable_combined_final for each training sample]
objective = mean(sample_scores) - 0.25 * population_std(sample_scores)
```

- Maximization direction: `maximize`.
- `0.25` is a **proposed** robustness coefficient; expose it as `--robustness-penalty`, default `0.25`, and store it in the study manifest.
- Report raw mean, minimum, median, standard deviation and all individual scores; do not hide poor samples behind the aggregate.
- A trial is valid for this objective **only if all configured training samples produce valid recordable results**. Reject or fail the trial otherwise, with precise diagnostics.
- If Phase 2 deliberately changes the training sample set or objective formula, create a new evaluation generation; re-evaluate the Phase 1 finalists and baseline under the new objective before comparing scores. The simplest first implementation is to keep the training set/objective identical across phases.

For cost control, a later enhancement may use two-stage evaluation or pruning, but do not apply naïve pruning that compares trials with different sample counts. Optuna's `MedianPruner` is only meaningful after consistent intermediate steps and enough comparable completed trials.

### 5.4 Score caching

Cache by a composite key such as:

```text
SHA256(
  canonical_25_parameter_config + sample_id + scoring_version +
  repository_commit + tool_image_identity + evaluator_version +
  reference/truth dataset identity
)
```

Only reuse successfully completed, version-matched evaluations. Do not cache timeouts or errors as stable quality outcomes. If dataset identities or code versions cannot be established, disable persistent cache reuse instead of risking stale scores.

---

## 6. Detailed Phase 1 — Initial optimization (100 trials)

**Objective:** Establish a trustworthy baseline and locate promising regions of the 15-parameter search space.

### Tasks

1. **Preflight** — verify Python/Optuna environment, Docker availability, Minos `--help`, baseline file readability, 25 parameters and their types, and existence/availability of at least the supplied practice sample ID.
2. **Freeze run manifest** — record source Git commit, baseline SHA-256, train/validation split, scoring version, random seed, objective formula, concurrency=1, timeout policy, and exact parameter ranges.
3. **Run untouched baseline** — execute the user's practice command for all configured training samples; retain full logs and scores. This is the control, executed outside Optuna's 100 trials.
4. **Verify parser** — confirm a real baseline run yields a correctly extracted `COMBINED_FINAL` and version. Compare parser result against the displayed terminal score. Reject a discarded-result fixture.
5. **Create Phase 1 study** — `direction="maximize"`; seeded TPE (`n_startup_trials` approximately `20–30` as a starting default); enqueue baseline or a known valid control as the **first trial** (counts toward 100).
6. **Run exactly 100 Optuna trials**, barring an explicit operator cancellation. For each trial: sample 15 parameters, merge with other baseline fields, pre-validate all 25, write per-trial config, sequentially run the practice command for each training sample, parse and persist results, compute aggregate objective, finalize the Optuna trial status.
7. **Monitor feasibility and runtime** — report trial count, valid/failed/discarded count, best-so-far score, per-sample score variation, median time per sample, and estimated cost to finish the configured budget. Do not invent a completion ETA until measured timings exist; report it as an estimate only.
8. **Create Phase 1 report** — show baseline vs best, top 10 unique configs, stability and failure rates, parameter importance (only if enough valid trials), and which ranges need Phase 2 exploration.

### Phase 1 deliverables

- SQLite Optuna study containing all 100 attempts/statuses.
- Baseline scores and immutable baseline snapshot.
- `phase1_best.conf`, `phase1_top10.csv`, `phase1_report.md`.
- Machine-readable `trial_results.jsonl` and `phase1_summary.json`.
- A shortlist of **5–10 distinct, valid** candidate configurations for Phase 2 seeding.

### Phase 1 exit criteria

- Baseline is reproducible, with a matching scoring version.
- The full set of 25 values is validated before every GATK invocation.
- Every completed Optuna trial is associated with config hash, sample IDs, logs, objective, and failure reason if applicable.
- A stopped study can be resumed without losing or silently duplicating completed trials.
- No original miner configuration has been changed.

---

## 7. Detailed Phase 2 — Expanded optimization (400 additional trials)

**Objective:** Refine the initial findings and explore interactions within a wider, biologically/workflow-safe search space.

### Tasks

1. **Analyze Phase 1** — compare best score against baseline, inspect per-sample consistency, detect influential parameters, unsuccessful regions, and probable parameter interactions. Treat importance statistics as suggestive rather than causal.
2. **Define and freeze Phase 2 search specification** — retain 15 primary parameters, add the seven from Section 4.1, and keep the three fixed fields. Narrow unpromising regions only when Phase 1 evidence supports doing so; preserve some exploration to avoid premature lock-in.
3. **Use a separate Phase 2 study** — seeded sampler, explicit phase metadata, same scoring version and primary objective. Enqueue top valid Phase 1 candidates (with the new fields filled from baseline) as counted trials.
4. **Implement condition-aware sampling/validation** — ensure `min_assembly_region_size <= max_assembly_region_size`, correct strict Minos types, and proper GATK output mode. Reject out-of-range values before launching any costly process.
5. **Run exactly 400 additional Optuna trials** (including seeds). Evaluate every candidate on the frozen training sample set with the same score calculation, while recording per-sample diagnostics and runtime.
6. **Deduplicate redundant configurations** — compare canonicalized full 25-param hashes; on true cache hits, use existing valid sample results when provenance exactly matches. If a candidate repeats, count the Optuna trial but avoid redundant GATK execution and log that it was reused.
7. **Manage reliability** — retry transient platform/download failures using bounded exponential backoff, but do not endlessly retry deterministic invalid configs. Pause on scoring-version changes, repeated platform failures, disk pressure, or widespread Docker errors.
8. **Generate Phase 2 report** — compare Phase 1/Phase 2 best against baseline on the *same sample set and scoring version*; summarize top 10 distinct configs, robustness, runtime, failure rate, parameter importance, and best-per-chromosome/sample behavior.
9. **Freeze Phase 3 finalists** — pre-register a small list (e.g., baseline plus top 3–5 unique candidates) using training-only results. Do not view holdout scores when choosing finalists.

### Phase 2 outputs

- Persistent study with **400 additional trials**.
- `phase2_best.conf`, `phase2_top10.csv`, `phase2_report.md`.
- `finalists_manifest.json` with source trial IDs, full parameter hashes, score version, training scores and selection reason.
- Runtime/cost and failure summaries.

### Phase 2 exit criteria

- 500 requested Optuna trials attempted across the two phases, or a transparently recorded operator-initiated early stop.
- Only allowlisted and type/range-valid configs have reached practice mode.
- Results are version-consistent and traceable to individual command executions.
- Finalists were frozen using **training data only**.

---

## 8. Detailed Phase 3 — Final validation (independent samples)

**Objective:** Determine whether the candidate improves performance on data **not used for optimization** and whether it is safe to export for optional deployment.

### Validation protocol

1. **Require untouched holdout IDs** — prompt the operator to provide at least three available independent practice sample IDs, preferably spanning different chromosomes and mutation profiles. List/check IDs using the existing practice sample API. Do not invent missing sample IDs.
2. **Check data leakage** — ensure there is no identical `sample_id`, BAM/truth hash, or known same underlying record/region between train and holdout sets. If donor metadata exists, prefer donor-disjoint validation.
3. **Freeze finalists and scoring version** — compare the untouched baseline and all finalists under the exact same evaluation settings and pinned score version.
4. **Evaluate each candidate on every holdout sample**, serially and without Optuna suggestions. Preserve raw per-sample score, validator-recordability, SNP/indel metrics, run duration, GATK errors and output artifacts.
5. **Measure reliability** — calculate mean, median, minimum and standard deviation of normalized score; mean score lift over baseline; per-sample improvement/regression; count of invalid and timed-out runs. Use paired comparisons on the same samples.
6. **Optional repeatability check** — rerun the leading candidate and baseline on at least one holdout sample to investigate nondeterminism (same versions/data/seed) and report any material score variation.
7. **Apply pre-registered promotion gate** — a conservative initial gate is: the selected candidate is valid on **all** required holdout samples, has **positive mean score lift** over baseline, and shows no severe pre-specified per-sample regression (e.g., worse by more than `0.02` in `COMBINED_FINAL`). These are proposed engineering thresholds, adjustable **before** opening the holdout results, not after.
8. **Do not retune on holdout failures** — if the best training candidate fails the gate, report `NOT_PROMOTED`, preserve the holdout as a consumed evaluation set, and plan any future optimization with a newly reserved test set.
9. **Export with explicit human review** — produce `best_validated_gatk.conf`, `final_validation_report.md`, and a deployment command suggestion. Do not automatically copy into `configs/gatk.conf`.

### Example comparison report schema

| Candidate | Train mean | Holdout mean | Holdout minimum | Mean lift vs baseline | Recordable / total | Decision |
|---|---:|---:|---:|---:|---:|---|
| Original baseline | measured | measured | measured | `0.000` | measured | control |
| Phase 1 best | measured | measured | measured | measured | measured | accept / reject |
| Phase 2 best | measured | measured | measured | measured | measured | accept / reject |

Do not populate this table with fictitious numbers or suggest a guaranteed improvement. Keep score normalization consistent: `COMBINED_FINAL` is 0–1; the displayed raw AdvancedScorer value is 0–100.

### Phase 3 success criteria

- Independent holdout IDs were never read as Optuna feedback.
- The promotion decision is auditable and based on pre-established gates.
- Raw logs and data provenance are saved for every comparison.
- The only production-facing output is a separate proposed `.conf` file awaiting operator approval.

---

## 9. Architecture and code organization

Create a **new, isolated package** under the Minos repository, e.g.:

```text
minos_subnet/
├── configs/
│   └── gatk.conf                         # EXISTING FILE — NEVER MODIFY
├── tuning/
│   ├── __init__.py
│   ├── cli.py                            # argparse/typer CLI and commands
│   ├── baseline.py                       # read, snapshot, canonicalize .conf
│   ├── parameters.py                     # allowlist, ranges, cross-field validation
│   ├── search_spaces.py                  # phase1 and phase2 sampler definitions
│   ├── practice_runner.py                # safe subprocess and working-dir discipline
│   ├── score_parser.py                   # plain/Rich score and discard parsing
│   ├── objective.py                      # multi-sample evaluation and aggregation
│   ├── storage.py                        # study, cache, trial manifest and artifacts
│   ├── reports.py                        # CSV, JSON and Markdown reporting
│   ├── validate.py                       # holdout scoring and promotion gates
│   └── models.py                         # typed results / error classification
├── tests/
│   └── tuning/
│       ├── test_parameters.py
│       ├── test_baseline_roundtrip.py
│       ├── test_score_parser.py
│       ├── test_practice_runner.py
│       ├── test_objective.py
│       ├── test_storage_resume.py
│       └── test_holdout_no_leakage.py
└── tuning_runs/                          # gitignored; generated artifacts
    └── <run-id>/
        ├── manifest.json
        ├── baseline/gatk.original.conf
        ├── baseline/scores.json
        ├── optuna.sqlite3
        ├── trials/
        │   └── 000042/
        │       ├── gatk.conf
        │       ├── trial.json
        │       └── <sample-id>/
        │           ├── stdout.log
        │           ├── stderr.log
        │           └── result.json
        ├── phase1/
        ├── phase2/
        └── final/
```

Use `pathlib.Path`, dataclasses/Pydantic where already supported by project dependencies, and explicit typed records. Avoid introducing heavyweight infrastructure or a web dashboard for v1. The **CLI and generated reports** are sufficient.

### Key interfaces

```python
@dataclass
class EvaluationResult:
    sample_id: str
    config_sha256: str
    scoring_version: str | None
    status: str                # success, discarded, timeout, etc.
    combined_final: float | None
    advanced_score: float | None
    elapsed_seconds: float
    exit_code: int | None
    log_dir: str
    metrics: dict
    error: str | None


def build_candidate(base: dict, suggested: dict) -> dict:
    """Return validated full 25-param config; never mutate base."""


def evaluate(config_path: Path, sample_id: str, study_manifest: dict) -> EvaluationResult:
    """Run Minos practice, classify result, preserve logs, validate version."""


def objective(trial: optuna.Trial) -> float:
    """Suggest phase-specific values, evaluate train set, and return frozen objective."""
```

The signatures are architectural guidance, not drop-in implementations. Cursor should match the checked-out project's real imports and avoid forking the Minos scorer when a safe reuse path exists.

---

## 10. Proposed CLI / operator experience

The following commands are **interfaces to implement**, not commands already provided by Minos.

### Initial setup and configuration check

```bash
python -m tuning.cli preflight \
  --baseline configs/gatk.conf \
  --sample-id 2ea4a48e-ba7f-43d8-86e7-81ae5a99196f
```

### Run Phase 1

```bash
python -m tuning.cli optimize \
  --phase 1 \
  --trials 100 \
  --baseline configs/gatk.conf \
  --train-samples tuning_train_samples.txt \
  --run-dir tuning_runs/my_run \
  --seed 42 \
  --workers 1
```

### Run Phase 2

```bash
python -m tuning.cli optimize \
  --phase 2 \
  --trials 400 \
  --run-dir tuning_runs/my_run \
  --workers 1 \
  --resume
```

### Final validation

```bash
python -m tuning.cli validate \
  --run-dir tuning_runs/my_run \
  --validation-samples tuning_validation_samples.txt \
  --finalists 5
```

### Report and export

```bash
python -m tuning.cli report --run-dir tuning_runs/my_run
python -m tuning.cli export --run-dir tuning_runs/my_run \
  --output tuning_runs/my_run/final/best_validated_gatk.conf
```

**Interface behaviors:** `--resume` uses persisted study/trial data; `--workers` above 1 should fail clearly until isolation is implemented. `--trials` defines the **total requested trials for that phase**, not extra trials on every resume. `export` must require Phase 3 approval criteria, or offer a clearly named exploratory export that cannot be mistaken for a validated production recommendation. A `--dry-run` mode should generate/config-validate commands without calling GATK.

### Sample-list text format

```text
# One sample ID per line; no overlap between files.
2ea4a48e-ba7f-43d8-86e7-81ae5a99196f
# Add additional verified training sample IDs here.
```

Do **not** use that same ID in the validation file. Source all other IDs from the actual practice sample catalog.

---

## 11. Data integrity, safety, retries and recovery

### 11.1 No active-miner modifications

- Record `sha256(configs/gatk.conf)` at startup and verify it remains identical at the end of every phase.
- Write generated files only under the requested `--run-dir`.
- Do not call `cp best.conf configs/gatk.conf` or update an active miner configuration from automation.
- If the hash changes externally, stop or show a high-severity warning and require a new explicit baseline snapshot.

### 11.2 Process isolation / collision prevention

The practice scorer's current implementation generates `score_run/output.vcf.gz` relative to a shared sample BAM directory. A distinct per-trial `.conf` file alone **does not isolate output VCFs**. Therefore:

- Default **one global practice evaluation at a time**.
- Hold a cross-process file lock for a given `sample_id` (preferably global lock for the initial implementation).
- Before every run, ensure stale output is not mistaken for a newly created VCF. Inspect/validate the scorer's failure path and artifact handling, and never trust a file merely because it exists.
- Parallel execution is a separate enhancement requiring isolated sample/scorer work directories or a safe upstream `output_dir` parameter, plus collision tests.

### 11.3 Retry policy

| Failure | Action |
|---|---|
| Temporary HTTP/platform error | Retry up to 2 times with backoff, then mark failed |
| Docker daemon unavailable / widespread infrastructure failure | Pause the study, surface diagnostics |
| GATK invalid parameter / deterministic malformed config | Do not retry; classify and investigate generator |
| Subprocess timeout | Kill owned process group, record timeout; one bounded retry only if justified |
| Missing/unparseable score | Mark score error; preserve raw log, do not return dummy score |
| Score version mismatch | Abort phase until version alignment is restored; never mix scales |
| User interruption | Graceful checkpoint and resumption without changing study definitions |

Retries are **per sample evaluation**, not extra Optuna suggestions. Store retry count and each attempt's logs.

### 11.4 Resume semantics

- Use SQLite-backed `RDBStorage`, `load_if_exists=True`, and stable phase study names.
- Freeze the manifest; reject resume when seed, train samples, scorer version, objective definition, ranges, or source commit differs unless a **new study/run ID** is created.
- Recover orphaned `RUNNING` trials after a crash with careful status reconciliation, never blindly mark them successful.
- Compute remaining trial slots as `requested_phase_total - already_accounted_for_trials`, avoiding a fresh 100 or 400 on every resume.
- Record both completed and failed counts; if failures are numerous, do not silently inflate the requested trial budget without operator consent.

### 11.5 Storage hygiene

- Store text/json metrics and small configs permanently.
- Make retention of large intermediate files configurable.
- Exclude `tuning_runs/`, SQLite files, reference data, downloaded BAMs, VCF intermediates and secrets from Git commits.
- Record start/end timestamps, host/resource metadata where safe, wall-clock time, and any timeout status.

---

## 12. Testing and acceptance criteria

### Unit tests (no GATK or network required)

1. Parse the exact supplied 25-parameter baseline and verify type counts: **14 ints, 7 floats, 2 bools, 2 enums**.
2. Serialize/reparse all values without changing their types, hashes (with canonical serialization), or enum case.
3. Reject every out-of-range field, unsupported key, wrong integer/float type, invalid enum, illegal booleans and non-finite floats.
4. Enforce region-size constraints, fixed `emit_ref_confidence=NONE`, fixed ploidy and contamination.
5. Score parser accepts the plain text `COMBINED_FINAL ... what a validator records` and Rich-rendered `COMBINED_FINAL` result; strips ANSI escape codes.
6. Score parser rejects missing score, out-of-range, `ZERO-INPUT`, `validator discards`, wrong scoring version and process failures, including when a numeric `COMBINED_FINAL` is present.
7. Multi-sample objective computes mean and population std from identical sample sets; missing/invalid samples do not become zero or missing-value successes.
8. Phase 1 emits exactly 15 suggested + 10 fixed fields; Phase 2 emits exactly 22 suggested + 3 fixed fields.
9. Prevent train/holdout overlap and forbid Optuna evaluation on holdout IDs.
10. Crash/restart test proves `--resume` respects the total requested trial count and reproduces stored settings.
11. File integrity test proves `configs/gatk.conf` is byte-identical before and after simulated runs.

### Mock integration tests

- Replace the practice subprocess with recorded success/error/timeout fixtures.
- Check command argument ordering, absolute config path, sample ID, cwd and `shell=False`.
- Verify per-trial logs and metadata written for success, retry, discard and failure.
- Ensure global evaluation lock prevents overlapping executions.
- Verify reporting and best-config export only when validation gate passes.

### Real smoke tests (resource-dependent)

1. Run the user's baseline practice command **once** and manually confirm the extracted score/version.
2. Run one dry-run synthetic trial to validate its config without launching GATK.
3. Execute one real generated trial on the provided training sample.
4. Confirm output artifacts/metrics match Minos practice and no baseline file changed.
5. Only then authorize the 100 + 400 trial workloads.

### Definition of done

- [ ] Three-phase workflow implemented with 100 + 400 trial budgets and independent validation.
- [ ] Exact Minos practice CLI integrated (not a fabricated custom GATK command).
- [ ] All 25 parameters represented; phase search spaces, types and bounds validated against local Minos code.
- [ ] Baseline immutable; separate export file generated only after validations.
- [ ] Parser extracts **recordable** `COMBINED_FINAL`; scorer version pinned.
- [ ] Complete experiment provenance, per-sample results, trial logs and Optuna SQLite state persisted.
- [ ] Safe timeout, retry, crash-resume, deduplication, and single-worker protection implemented.
- [ ] Phase reports compare baseline vs finalists fairly; holdout never used for fitting.
- [ ] Unit/mock tests pass; at least one verified practice-mode smoke test passes.
- [ ] README includes installation, exact run/resume/report/validate/export commands and failure troubleshooting.

---

## 13. Implementation sequence for Cursor

Implement in small reviewable increments, **not one monolithic coding pass**:

**Milestone A — Repository inspection and architecture confirmation**

- Read the files in Section 2 and document verified output signatures, scorer version behavior, exact parameter schema, and output directory collisions.
- Create the package/file skeleton and CLI interface.
- Identify current Python environment and safest dependency management (prefer add-on requirements or existing pyproject tooling).

**Milestone B — Reliable practice adapter**

- Implement baseline parsing/validation and `.conf` generation.
- Implement safe subprocess runner, locks, logs, timeout/retry and score parser.
- Implement one real practice-mode smoke test and mock fixtures. Do not proceed without reliable score extraction.

**Milestone C — Phase 1**

- Implement 15-variable study, immutable manifest, Optuna storage/resume and 100-trial orchestration.
- Add CSV/JSON/Markdown summary, candidate export and score provenance.

**Milestone D — Phase 2**

- Add the seven further search variables, cross-field constraints, distinct study, warm-start shortlist and 400-trial orchestration.
- Add caching/deduplication, richer reports and comparison dashboard in Markdown/CSV (not a required web UI).

**Milestone E — Final validation and release hygiene**

- Add disjoint holdout manifest validation, paired baseline comparisons, promotion gates and explicit export.
- Run tests; write operator documentation; verify baseline hash unchanged.

At each milestone, show changed files, tests executed, known limitations and the next acceptance gate. The system must work with a single supplied training ID for smoke tests but must **not** claim independent validation without independently supplied sample IDs.

---

## 14. Risks and mitigation

| Risk | Why it matters | Mitigation |
|---|---|---|
| Overfit to one BAM/sample | Best training score may not generalize | Multi-sample training objective and strict holdout evaluation |
| Scoring version changes mid-run | v1/v2 scores are on different scales | Pin version; abort and start a new study on change |
| Unscorable/discarded output | Numeric score may not count on-chain | Parse validator-recordability message and explicit discard cases |
| Huge compute requirement | 500 trials × multiple samples × GATK can be expensive | Baseline runtime benchmark, checkpointing, cache and resource budgets |
| Concurrent output corruption | Practice scorer uses a fixed `score_run` directory | Single worker and process lock; isolate before enabling parallelism |
| Out-of-range parameters | Minos rejects whole config | Local validation against actual whitelist and strict types |
| Illogical biology/workflow knobs | Can reduce accuracy or change output semantics | Fix ploidy, reference-confidence mode and contamination unless independently justified |
| Platform unavailable or fallback scoring | Silent v1 fallback may invalidate comparisons | Verify authoritative scoring version and refuse ambiguous runs |
| Source/package drift | Same config could score differently | Hash repository, Docker image, reference, truth and scorer versions |
| Accidental live-miner modification | Active config could be damaged | Immutable baseline, separate run directory, explicit non-deploying export |

---

## 15. Cursor execution instruction (copy into Cursor chat)

> Implement the Minos GATK Optuna tuning system described in `MINOS_GATK_OPTUNA_DEVELOPMENT_PLAN.md`. Start by reading the actual repository code paths identified in Section 2. Preserve `configs/gatk.conf` byte-for-byte and do not trigger live miner submissions. Build the reliable practice evaluator, score/version parser and tests **before** scheduling expensive Optuna runs. Deliver Phase 1 (100 trials), then Phase 2 (400 additional trials), then independent Phase 3 validation. Use only approved GATK parameter names/types/ranges from `templates/tool_params.py`; generate an isolated `.conf` per trial; optimize validator-recordable `COMBINED_FINAL`; keep a pinned scoring version; store all results in persistent SQLite and per-trial logs; provide a resumable CLI and final reports. Implement code and smoke tests, but **do not launch the 500-trial workload automatically**. Present preflight/runtime findings, changed files and a precise command for the operator to start each phase.

---

## 16. External and project references

- Minos main repository: https://github.com/minos-protocol/minos_subnet
- Practice/self-scoring CLI: https://github.com/minos-protocol/minos_subnet/blob/main/README.md
- Canonical parameter ranges documentation: https://github.com/minos-protocol/minos_subnet/blob/main/docs/parameter_ranges.md
- Actual parameter enforcement: https://github.com/minos-protocol/minos_subnet/blob/main/templates/tool_params.py
- Scoring v1/v2: https://github.com/minos-protocol/minos_subnet/blob/main/docs/scoring.md
- Minos tuning guidance: https://github.com/minos-protocol/minos_subnet/blob/main/docs/tuning_guide.md
- Practice implementation: https://github.com/minos-protocol/minos_subnet/blob/main/neurons/miner.py
- Optuna documentation: https://optuna.readthedocs.io/

**Note:** This document is an implementation plan. It does not claim that 500 trials have been executed, that a score has been measured, or that a winning Minos configuration has already been found.

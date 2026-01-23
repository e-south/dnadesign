## Generation and constraints

Generation is driven by a plan. Each plan item is a constraint bucket with a quota or fraction,
and DenseGen enforces fixed motifs and regulator requirements through the dense-arrays solver.

### Contents
- [Plan definition](#plan-definition) - quota/fraction rules and plan item shape.
- [Promoter constraints](#promoter-constraints) - fixed motifs and spacing.
- [Side biases](#side-biases-positional-preferences) - left/right placement preferences.
- [Solver strategy](#solver-strategy) - solution ordering and backend selection.
- [Stage‑B sampling controls](#stage-b-sampling-controls) - how libraries are built and resampled.
- [Regulator constraints](#regulator-constraints) - per-plan requirements and validation.

---

### Plan definition

Each plan item has a `name` and either `quota` or `fraction`.

```yaml
plan:
  - name: sigma70
    quota: 200
    required_regulators: ["LexA", "CpxR"]
    min_required_regulators: 2
    min_count_by_regulator:
      LexA: 2
    fixed_elements:
      promoter_constraints:
        - upstream: "TTGACA"
          downstream: "TATAAT"
          spacer_length: [15, 19]
```
Note: `generation.sequence_length` must be at least as long as the widest motif in the library or fixed elements; DenseGen fails fast if a motif cannot fit.

---

### Promoter constraints

Use `fixed_elements.promoter_constraints` to enforce fixed motifs and spacing. Motifs must be A/C/G/T only.

Fields:
- `upstream`, `downstream` (motif strings)
- `spacer_length: [min, max]`
- `upstream_pos`, `downstream_pos` (optional ranges)

---

### Side biases (positional preferences)

Use `fixed_elements.side_biases` to bias motifs toward the left or right of the sequence.

```yaml
fixed_elements:
  side_biases:
    left: ["GAAATAACATAATTGA", "TTATATTTTACCCATTT"]
    right: ["CATAAGAAAAA", "CATTCATTTG"]
```

Rules:
- Motifs must be A/C/G/T only.
- Motifs must exist in the sampled library (DenseGen fails fast if missing).
- A motif cannot appear in both `left` and `right`.

---

### Solver strategy

DenseGen exposes dense-arrays solution modes via `solver.strategy`:

- `iterate` - yield solutions in descending score.
- `diverse` - yield solutions with diversity-biased ordering.
- `optimal` - only the best solution per library.
- `approximate` - heuristic solution per library (no solver backend required).
- `strands` - `single | double` (default: `double`).

DenseGen fails fast if the requested solver backend is unavailable; use `dense validate-config --probe-solver` or `dense inspect config --probe-solver` to check availability before long runs.

```yaml
solver:
  backend: CBC
  strategy: diverse
  time_limit_seconds: 10
  strands: double
```
DenseGen fails fast if the solver cannot apply requested time limits or thread counts (CBC does not support threads).

---

### Stage‑B sampling controls

Stage‑B reminder: Stage‑A sampling lives under `densegen.inputs[].sampling` and produces TFBS pools.
Stage‑B sampling below selects solver libraries from those pools (or from library artifacts) and
is the only place resampling happens. Stage‑B outcomes are recorded in `outputs/libraries/*`,
`outputs/tables/attempts.parquet`, and `outputs/meta/run_manifest.json`.

Per‑field guide (what it does → when to use → failure modes → artifacts impacted):

- `pool_strategy` — chooses Stage‑B library construction mode (full vs subsample vs iterative). Use `full`
  for tiny pools, `subsample` for large pools, `iterative_subsample` to resample aggressively. Failure
  modes: `full` ignores **Stage‑B resampling** and can stall if pools are weak; `iterative_subsample` with low caps
  can terminate early. Artifacts: `outputs/libraries/*`, `outputs/tables/attempts.parquet`, `outputs/meta/run_manifest.json`.
- `library_source` — Stage‑B source of libraries (`build` vs `artifact`). Use `artifact` for
  deterministic replays. Failure: missing or mismatched artifact metadata. Artifacts: `outputs/libraries/*`,
  `outputs/tables/attempts.parquet`, `outputs/meta/run_manifest.json`.
- `library_artifact_path` — path to a Stage‑B library artifact directory when `library_source: artifact`.
  Use when replaying a prior Stage‑B build. Failure: path missing or incompatible metadata. Artifacts:
  `outputs/libraries/*`, `outputs/tables/attempts.parquet`, `outputs/meta/run_manifest.json`.
- `library_size` — number of TFBS per Stage‑B library (subsample modes). Use to control library
  diversity/size. Failure: too small can starve constraints; too large can over‑constrain the solver.
  Artifacts: `outputs/libraries/*`, `outputs/tables/attempts.parquet`, `outputs/meta/run_manifest.json`.
- `subsample_over_length_budget_by` — Stage‑B budget in bp to bias sampling toward longer libraries.
  Use to penalize over‑length libraries when pools contain long motifs. Failure: too low can bias
  against required motifs. Artifacts: `outputs/libraries/*`, `outputs/tables/attempts.parquet`, `outputs/meta/run_manifest.json`.
- `library_sampling_strategy` — Stage‑B selection policy (`tf_balanced`, `uniform_over_pairs`,
  `coverage_weighted`). Use `tf_balanced` for even TF coverage, `uniform_over_pairs` for pair diversity,
  `coverage_weighted` to boost under‑used motifs. Failure: aggressive weighting can overfit recent runs.
  Artifacts: `outputs/libraries/*`, `outputs/tables/attempts.parquet`, `outputs/meta/run_manifest.json`.
- `coverage_boost_alpha` — Stage‑B weighting strength for `coverage_weighted`. Use small values (e.g., 0.1)
  to avoid oscillation. Failure: too large can destabilize coverage. Artifacts: `outputs/libraries/*`,
  `outputs/tables/attempts.parquet`, `outputs/meta/run_manifest.json`.
- `coverage_boost_power` — Stage‑B exponent for `coverage_weighted`. Use to shape how quickly under‑used
  motifs are boosted. Failure: extreme values can flatten or over‑amplify weights. Artifacts:
  `outputs/libraries/*`, `outputs/tables/attempts.parquet`, `outputs/meta/run_manifest.json`.
- `avoid_failed_motifs` — Stage‑B penalty toggle for motifs tied to failed solve attempts. Use when
  repeated failures are dominated by a few motifs. Failure: can over‑penalize rare motifs in small pools.
  Artifacts: `outputs/libraries/*`, `outputs/tables/attempts.parquet`, `outputs/meta/run_manifest.json`.
- `failure_penalty_alpha` — Stage‑B penalty strength when `avoid_failed_motifs` is enabled. Use low values
  to soften penalties. Failure: too large can collapse library diversity. Artifacts: `outputs/libraries/*`,
  `outputs/tables/attempts.parquet`, `outputs/meta/run_manifest.json`.
- `failure_penalty_power` — Stage‑B penalty exponent when `avoid_failed_motifs` is enabled. Use to sharpen
  or smooth penalties. Failure: extreme values can zero‑out motifs. Artifacts: `outputs/libraries/*`,
  `outputs/tables/attempts.parquet`, `outputs/meta/run_manifest.json`.
- `cover_all_regulators` — Stage‑B rule to ensure each TF appears in the library. Use when constraints
  require per‑TF coverage. Failure: can be impossible for sparse pools. Artifacts: `outputs/libraries/*`,
  `outputs/tables/attempts.parquet`, `outputs/meta/run_manifest.json`.
- `unique_binding_sites` — Stage‑B uniqueness filter at the TF+TFBS pair level. Use to avoid duplicate
  sites. Failure: can under‑fill libraries when pools are small. Artifacts: `outputs/libraries/*`,
  `outputs/tables/attempts.parquet`, `outputs/meta/run_manifest.json`.
- `max_sites_per_regulator` — Stage‑B cap per TF. Use to prevent dominance by a single TF. Failure:
  too low can make libraries infeasible for constraint plans. Artifacts: `outputs/libraries/*`,
  `outputs/tables/attempts.parquet`, `outputs/meta/run_manifest.json`.
- `relax_on_exhaustion` — Stage‑B relaxation toggle when sampling can’t fill a library. Use with small
  pools to avoid hard failures. Failure: relaxed libraries can violate intended coverage. Artifacts:
  `outputs/libraries/*`, `outputs/tables/attempts.parquet`, `outputs/meta/run_manifest.json`.
- `allow_incomplete_coverage` — Stage‑B permit missing TF coverage when pools are sparse. Use for
  exploratory runs. Failure: can hide missing TFs unless monitored. Artifacts: `outputs/libraries/*`,
  `outputs/tables/attempts.parquet`, `outputs/meta/run_manifest.json`.
- `iterative_max_libraries` — Stage‑B cap for `iterative_subsample` library rebuilds. Use to bound
  runtime. Failure: too low can terminate early with unmet quotas. Artifacts: `outputs/libraries/*`,
  `outputs/tables/attempts.parquet`, `outputs/meta/run_manifest.json`.
- `iterative_min_new_solutions` — Stage‑B threshold to decide whether a new library “worked.”
  Use to prevent wasteful **Stage‑B resampling**. Failure: too high can force endless resamples. Artifacts:
  `outputs/libraries/*`, `outputs/tables/attempts.parquet`, `outputs/meta/run_manifest.json`.

### Run scheduling (round‑robin)

`runtime.round_robin` controls **scheduling**, not Stage‑B sampling. When enabled, DenseGen interleaves
plan items across inputs so each plan advances in turn (one subsample per pass). This is useful when
you have multiple constraint sets (e.g., different fixed sequences) and want a single run to progress
each design target in parallel. Round‑robin is distinct from Stage‑B sampling (`generation.sampling`):
Stage‑B library sampling still uses the same policy per plan, but round‑robin can trigger more frequent
Stage‑B library rebuilds when `pool_strategy: iterative_subsample` is used. Expect extra compute if many
plans are active.

Stage‑A PWM sampling is performed **once per run** and cached across round‑robin passes. If you need a
fresh Stage‑A sample, start a new run with `dense run --fresh` (or stage a new workspace).

### Runtime policy knobs (Stage‑B resampling + stop conditions)

Key `runtime.*` controls:
- `arrays_generated_before_resample` — number of successful arrays to emit before forcing a new
  Stage‑B library (for iterative subsampling).
- `stall_seconds_before_resample` — idle time with no new solutions before Stage‑B resampling.
  This also applies a per‑solve time limit (seconds) for solver‑based strategies; set to `0` to disable.
- `stall_warning_every_seconds` — how often to log stall warnings.
- `max_resample_attempts` / `max_total_resamples` — caps on resample retries.
- `max_seconds_per_plan` — time budget per plan item (0 = no limit).
- `max_failed_solutions` / `max_duplicate_solutions` — guardrails to stop when failure/duplicate
  counts are too high.

---

### Regulator constraints

DenseGen supports three regulator constraint modes per plan item:

- `required_regulators`: when `min_required_regulators` is unset, enforce at least one site
  per listed regulator (all-of, applied to final sequences).
- `min_required_regulators`: when set, enforce at least K distinct regulators in the final sequence. If `required_regulators` is provided, it becomes the candidate set (k-of-n). If `required_regulators` is empty, the constraint applies to the full regulator pool.
- `min_count_by_regulator`: enforce per-regulator minimum counts.

Solver strategies (`iterate|diverse|optimal`) enforce these constraints at the solver level;
`approximate` runs are validated in the pipeline to keep behavior consistent.

Notes:
- `min_count_by_regulator` takes precedence over the global `runtime.min_count_per_tf`
  for listed regulators (DenseGen uses the maximum of the two).

---

@e-south

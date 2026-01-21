## Generation and constraints

Generation is driven by a plan. Each plan item is a constraint bucket with a quota or fraction,
and DenseGen enforces fixed motifs and regulator requirements through the dense-arrays solver.

### Contents
- [Plan definition](#plan-definition) - quota/fraction rules and plan item shape.
- [Promoter constraints](#promoter-constraints) - fixed motifs and spacing.
- [Side biases](#side-biases-positional-preferences) - left/right placement preferences.
- [Solver strategy](#solver-strategy) - solution ordering and backend selection.
- [Sampling controls](#sampling-controls) - how libraries are built and resampled.
- [Regulator constraints](#regulator-constraints) - per-plan requirements and validation.

---

### Plan definition

- Each plan item has a `name` and either `quota` or `fraction`.
- Mixing quotas and fractions across items is not allowed.

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
          spacer_length: [16, 18]
```
Note: `generation.sequence_length` must be at least as long as the widest motif in the library
or fixed elements; DenseGen fails fast if a motif cannot fit.

---

### Promoter constraints

Use `fixed_elements.promoter_constraints` to enforce fixed motifs and spacing. Motifs must be
A/C/G/T only.

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
- `approximate` - heuristic solution per library (no solver options; backend optional).
- `strands` - `single | double` (default: `double`).
DenseGen fails fast if the requested solver backend is unavailable; use
`dense validate-config --probe-solver` or `dense inspect config --probe-solver`
to check availability before long runs.

```yaml
solver:
  backend: CBC
  strategy: diverse
  options: ["Threads=8", "TimeLimit=10"]
  strands: double
  allow_unknown_options: false
```

DenseGen validates solver option keys for known backends and fails fast on unknown options. If you
need to pass custom solver flags, set `solver.allow_unknown_options: true` explicitly.

---

### Sampling controls

`generation.sampling` controls how binding-site libraries are built (pool strategy, coverage,
uniqueness, caps, and relaxation). DenseGen records sampling policy and outcomes in metadata.

Key fields:
- `pool_strategy`: `full | subsample | iterative_subsample`
- `library_source`: `build | artifact` (use `artifact` to replay prebuilt libraries)
- `library_artifact_path`: path to `outputs/libraries` from `dense stage-b build-libraries`
- `library_size` (used for subsample strategies)
- `library_sampling_strategy` (`tf_balanced | uniform_over_pairs | coverage_weighted`)
- `coverage_boost_alpha`, `coverage_boost_power` (used with `coverage_weighted`)
- `avoid_failed_motifs`, `failure_penalty_alpha`, `failure_penalty_power` (optional penalties for motifs tied to failed solves)
- `cover_all_regulators`, `unique_binding_sites`, `max_sites_per_regulator`
- `iterative_max_libraries`, `iterative_min_new_solutions`

Notes:
- `pool_strategy: full` uses a single library (no resampling) and ignores `library_size`, `subsample_over_length_budget_by`,
  and related sampling caps/strategies (DenseGen warns in `dense validate-config`/`dense inspect plan`).
- `subsample` can resample reactively on stalls/duplicate guards.
- `iterative_subsample` resamples proactively after `arrays_generated_before_resample` or when a
  library under-produces.
- `unique_binding_sites` enforces uniqueness at the regulator+sequence pair level.
- `coverage_weighted` dynamically boosts underused TFBS based on the run’s usage counts.
- `avoid_failed_motifs: true` down-weights TFBS that repeatedly appear in failed solve attempts (tracked in attempts.parquet).

### Stage‑A vs Stage‑B sampling (mental model)

**Stage‑A (input sampling)** lives under `densegen.inputs[].sampling` and defines how TFBS pools
are generated from PWMs (e.g., DenseGen log‑odds vs FIMO p‑values, thresholds, mining limits,
length policy). Stage‑A produces the realized TFBS pool (`input_tfbs_count`), which is cached
once per run and reused across round‑robin passes.

**Stage‑B (library sampling)** lives under `densegen.generation.sampling` and selects a **solver
library** from the Stage‑A pool (or from a binding‑site table / sequence library). This is where
`pool_strategy`, `library_size`, and sampling strategies (tf‑balanced, uniform over pairs,
coverage‑weighted) apply. Stage‑B is the only place that resampling happens.

Use `dense stage-a build-pool` to materialize pools and `dense stage-b build-libraries` to preview
solver libraries without running the solver.
To **replay** a specific library artifact deterministically, set
`generation.sampling.library_source: artifact` and point
`generation.sampling.library_artifact_path` at the library artifact directory.

### Run scheduling (round‑robin)

`runtime.round_robin` controls **scheduling**, not sampling. When enabled, DenseGen interleaves plan
items across inputs so each plan advances in turn (one subsample per pass). This is useful when you
have multiple constraint sets (e.g., different fixed sequences) and want a single run to progress
each design target in parallel.

Round‑robin is **distinct from Stage‑B sampling** (`generation.sampling`): library sampling still
uses the same policy per plan, but round‑robin can trigger more frequent library rebuilds when
`pool_strategy: iterative_subsample` is used. Expect extra compute if many plans are active.

Input PWM sampling is performed **once per run** and cached across round‑robin passes. If you
need a fresh PWM sample, start a new run with `dense run --fresh` (or stage a new workspace).

### Runtime policy knobs (resampling + stop conditions)

Key `runtime.*` controls:
- `arrays_generated_before_resample` — number of successful arrays to emit before forcing a new
  library (for iterative subsampling).
- `stall_seconds_before_resample` — idle time with no new solutions before resampling.
- `stall_warning_every_seconds` — how often to log stall warnings.
- `max_resample_attempts` / `max_total_resamples` — caps on resample retries.
- `max_seconds_per_plan` — time budget per plan item (0 = no limit).
- `max_failed_solutions` / `max_duplicate_solutions` — guardrails to stop when failure/duplicate
  counts are too high.

---

### Regulator constraints

DenseGen supports three regulator constraint modes per plan item:

- `required_regulators`: when `min_required_regulators` is **unset**, enforce at least one site
  per listed regulator (**all-of**, applied to final sequences).
- `min_required_regulators`: when set, enforce at least K distinct regulators in the **final
  sequence**. If `required_regulators` is provided, it becomes the candidate set (k-of-n).
  If `required_regulators` is empty, the constraint applies to the full regulator pool.
- `min_count_by_regulator`: enforce per-regulator minimum counts.

Solver strategies (`iterate|diverse|optimal`) enforce these constraints at the solver level;
`approximate` runs are validated in the pipeline to keep behavior consistent.

Notes:
- `min_count_by_regulator` takes precedence over the global `runtime.min_count_per_tf`
  for listed regulators (DenseGen uses the maximum of the two).

---

@e-south

# DenseGen config reference

Use this page when you need exact YAML keys and constraints.
Unknown keys are hard errors.
All relative paths resolve from the config file directory.

Sampling split:
- Stage-A sampling keys live under `densegen.inputs[].sampling`
- Stage-B sampling keys live under `densegen.generation.sampling`

If you want concepts first, read:
- [inputs guide](../guide/inputs.md)
- [sampling guide](../guide/sampling.md)
- [generation guide](../guide/generation.md)

### Contents
- [Top-level](#top-level) - required roots and plotting.
- [`densegen.inputs[]`](#densegeninputs) - input sources and Stage‑A sampling.
- [`densegen.run`](#densegenrun) - run identifier and root.
- [`densegen.output`](#densegenoutput) - output targets and schema.
- [`densegen.generation`](#densegengeneration) - plan and fixed elements.
- [`densegen.motif_sets`](#densegenmotif_sets) - reusable motif dictionaries.
- [`densegen.generation.plan_templates`](#densegengenerationplan_templates) - matrix plan expansion.
- [`densegen.generation.sequence_constraints`](#densegengenerationsequence_constraints) - global final-sequence motif rules.
- [`densegen.generation.sampling`](#densegengenerationsampling-stage-b-sampling) - Stage-B library building controls.
- [`densegen.solver`](#densegensolver) - backend and strategy.
- [`densegen.runtime`](#densegenruntime) - retry and guard rails.
- [`densegen.postprocess.pad`](#densegenpostprocesspad) - pad policy.
- [`densegen.logging`](#densegenlogging) - log file configuration.
- [`plots`](#plots) - plotting options and defaults.
- [Minimal example](#minimal-example) - smallest runnable config.

---

### Top-level

- `densegen` (required)
- `densegen.schema_version` (required; supported: `2.9`)
- `densegen.run` (required; run-scoped I/O root)
- `plots` (optional; required `source` when `output.targets` has multiple sinks)

---

### `densegen.inputs[]`

Choose one input type per entry:

PWM inputs perform **Stage‑A sampling** (sampling sites from PWMs) via
`densegen.inputs[].sampling`. This is distinct from **Stage‑B sampling**
(`densegen.generation.sampling` for Stage‑B sampling), which selects solver libraries from the realized TFBS pool.

- `type: binding_sites`
  - `path` - CSV, Parquet, or XLSX file
  - `format` - `csv | parquet | xlsx` (optional if extension is `.csv`/`.parquet`/`.xlsx`)
  - `columns.regulator` (default: `tf`)
  - `columns.sequence` (default: `tfbs`)
  - `columns.site_id` (optional)
  - `columns.source` (optional)
  - Empty regulator/sequence rows are errors
  - Duplicate regulator+sequence rows are allowed (use Stage‑B `generation.sampling.unique_binding_sites` or
    `generation.sampling.unique_binding_cores` to dedupe)
  - Sequences must be A/C/G/T only
  - `tfbs_core` is derived as the full binding-site sequence for core-level uniqueness checks
- `type: sequence_library`
  - `path` - CSV or Parquet file
  - `format` - `csv | parquet` (optional if extension is `.csv`/`.parquet`)
  - `sequence_column` (default: `sequence`)
  - Empty or null sequences are errors
  - Sequences must be A/C/G/T only
- `type: background_pool`
  - Stage‑A negative selection (random DNA filtered to avoid TFBS hits)
  - `sampling` (required):
    - `n_sites` (int > 0)
    - `mining`:
      - `batch_size` (int > 0)
      - `budget.candidates` (int > 0; fixed candidate count)
    - `length`
      - `policy`: `exact | range` (default: `range`)
      - `range` or `exact` (required by policy)
    - `uniqueness.key`: `sequence` (only option)
    - `gc` (optional): `min`, `max` in [0, 1]
    - `filters`
      - `forbid_kmers`: list of A/C/G/T kmers to exclude
      - `fimo_exclude` (optional):
        - `pwms_input`: list of PWM input names to screen against
        - `allow_zero_hit_only`: when true, reject any FIMO hit
        - `max_score_norm`: required when `allow_zero_hit_only=false`
        - FIMO exclusion uses a p-value threshold of 1e-4 and scans both strands
  - FIMO resolves via `MEME_BIN` or PATH (pixi users should run `pixi run dense ...`).
- `type: pwm_meme`
  - `path` - MEME PWM file
  - `motif_ids` (optional list) - choose motifs by ID
  - `sampling` (required Stage‑A config):
    - `strategy`: `consensus | stochastic | background`
    - `n_sites` (int > 0)
    - `mining` - batch controls for FIMO mining:
      - `batch_size` (int > 0; default 100000) - candidates per FIMO batch
      - `budget` (required)
        - `mode`: `tier_target | fixed_candidates`
        - `target_tier_fraction` (float in (0, 1]; required when `mode=tier_target`)
        - `candidates` (int > 0; required when `mode=fixed_candidates`)
        - `max_candidates` (optional int > 0; hard cap for tier_target)
        - `max_seconds` (optional float > 0; escape hatch for tier_target)
        - `min_candidates` (optional int > 0; floor before tier_target can stop)
        - `growth_factor` (float > 1; default 1.25)
      - `log_every_batches` (int > 0; default 1)
    - `fixed_candidates` is the recommended mining mode (direct, user-set budget).
      `tier_target` is advanced and may stop early at caps/time; shortfalls are recorded in the manifest.
    - `bgfile` (optional path) - MEME bfile-format background model for FIMO (one base per line,
      e.g., `A 0.25`); also used for Stage-A score normalization and MMR information-content weights
    - `keep_all_candidates_debug` (bool, default false) - write candidate Parquet logs to
      `outputs/pools/candidates/` for inspection (overwritten by `stage-a build-pool --fresh`
      or `dense run --fresh`)
    - `include_matched_sequence` (bool, default true; must be true for PWM sampling) - include
      `fimo_matched_sequence` in TFBS outputs (config validation rejects false)
    - `tier_fractions` (optional list of three floats in (0, 1], non‑decreasing, sum ≤ 1.0; default
      `[0.001, 0.01, 0.09]`). Used for diagnostic tiers and the cumulative rung ladder for MMR pool selection.
    - `length`
      - `policy`: `exact | range` (default: `exact`)
      - `range`: `[min, max]` (required when `policy=range`; `min` can be below motif length,
        in which case Stage‑A trims to the max‑information window per candidate)
    - `trimming`
      - `window_length` (optional int > 0; trims PWM to a max‑information window before Stage‑A sampling)
      - `window_strategy`: `max_info` (window selection strategy)
    - `uniqueness`
      - `key`: `sequence | core` (default `core` for PWM inputs)
      - `cross_regulator_core_collisions`: `allow | warn | error` (default `warn`).
        For multi-motif PWM inputs, this checks for the same `tfbs_core` appearing under
        different regulators in the same Stage-A pool build.
    - `selection`
      - `policy`: `top_score | mmr` (default `top_score`)
      - `rank_by`: `score | score_norm` (default `score`; `score_norm` is length-normalized)
      - `alpha` (float in (0, 1]; MMR score weight)
      - `pool` (required when `policy=mmr`)
        - `min_score_norm` (optional float in (0, 1]; hard lower bound for MMR pool inclusion using
          the fraction of theoretical max log-odds score:
          `best_hit_score / pwm_theoretical_max_score`)
        - `max_candidates` (optional int > 0; upper bound on the deterministic MMR target pool size;
          default target pool is `ceil(10.0 * n_sites)`)
        - `relevance_norm` (optional: `percentile | minmax_raw_score`; default `minmax_raw_score`)
      - MMR pool selection uses the cumulative rung ladder derived from `sampling.tier_fractions`,
        evaluated by post-gate counts, then slices to a deterministic target pool.
    - `consensus` requires `n_sites: 1`
    - `background` samples cores from the PWM background distribution before padding
    - FIMO resolves `fimo` via `MEME_BIN` or PATH; pixi users should run `pixi run dense ...` so it is available.
    - Eligibility is `best_hit_score > 0` and requires a FIMO hit.
- Algorithmic behavior (eligibility, tiering, tier-target mining math, and MMR diversity) is defined in:
  - `../guide/sampling.md`
- `type: pwm_meme_set`
  - `paths` - list of MEME PWM files (merged into a single TF pool)
  - `motif_ids` (optional list) - choose motifs by ID across files
  - `sampling` (required Stage‑A config) - same fields as `pwm_meme`
- `type: pwm_jaspar`
  - `path` - JASPAR PFM file
  - `motif_ids` (optional list) - choose motifs by ID
  - `sampling` (required Stage‑A config) - same fields as `pwm_meme`
- `type: pwm_matrix_csv`
  - `path` - CSV matrix with A/C/G/T columns
  - `motif_id` (required) - single motif ID label
  - `columns` (optional) - map columns to `A/C/G/T` (defaults to literal column names)
  - `sampling` (required Stage‑A config) - same fields as `pwm_meme`
- `type: pwm_artifact`
  - `path` - per-motif JSON artifact (contract-first; see `docs/reference/motif_artifacts.md`)
  - `sampling` (required Stage‑A config) - same fields as `pwm_meme`
- `type: pwm_artifact_set`
  - `paths` - list of per-motif JSON artifacts (one file per motif)
  - `sampling` (required Stage‑A config) - same fields as `pwm_meme`
  - `overrides_by_motif_id` (optional dict) - per‑motif Stage‑A sampling overrides (deep‑merged
    onto the base `sampling`, so partial overrides are allowed)
    - `uniqueness.cross_regulator_core_collisions` must stay consistent across base and overrides
- `type: usr_sequences`
  - `dataset` - USR dataset name
  - `root` - USR root path (required; no fallback)
  - `limit` (optional) - max rows
  - Sequences must be A/C/G/T only

Input paths resolve relative to the config file directory.
Outputs (tables), logs, and plots must resolve inside `outputs/` under `densegen.run.root`.

---

### `densegen.run`

- `id` - run identifier (string; required; must not contain path separators)
- `root` - run root directory (required; must exist on disk)

---

### `densegen.output`

- `schema`: shared output schema (`bio_type`, `alphabet`) used for IDs and validation (required).
- `targets`: list of sinks to write (`usr`, `parquet`)
- When multiple targets are set, outputs must be in sync before a run; mismatches are errors.
- `usr` (required when `targets` includes `usr`)
  - `dataset`, `root`, `chunk_size`, `health_event_interval_seconds`, `allow_overwrite`
  - `health_event_interval_seconds` (float > 0; default 60) controls cadence for `densegen_health` USR events
  - `npz_fields` (optional list of metadata keys to offload into NPZ artifacts; see outputs doc)
  - `npz_root` (optional path for NPZ artifacts; defaults to `<dataset>/_artifacts/densegen_npz`)
- `parquet` (required when `targets` includes `parquet`)
  - `path` (file), `deduplicate`, `chunk_size`
  - `path` must be a `.parquet` file (single-file output)
- `output.usr.root` and `output.parquet.path` must be inside `outputs/` under `densegen.run.root`

---

### `densegen.generation`

- `sequence_length` (int > 0)
- `sequence_length` should be >= the widest required motif (library TFBS or fixed elements); if it
  is shorter, Stage‑B records infeasibility and warns.
- `sampling` (Stage‑B; see below)
- Exactly one of `plan` or `plan_templates` is required.
  - `plan` (non-empty list)
  - Each item: `name` and `quota` (int > 0)
  - `sampling.include_inputs` (required) - input names that feed the plan‑scoped pool.
  - `fixed_elements.promoter_constraints[]` supports `name`, `upstream`, `downstream`,
    `spacer_length`, `upstream_pos`, `downstream_pos`
  - Promoter motifs must be non-empty A/C/G/T strings (normalized to uppercase)
  - `fixed_elements.side_biases` supports motif placement preferences:
    - `left`: list of motifs biased toward the 5prime side
    - `right`: list of motifs biased toward the 3prime side
    - Motifs must be A/C/G/T and must exist in the sampled library
  - `regulator_constraints` (required)
    - `groups` (list) - regulator groups that must appear in each solution.
      - Each group: `name`, `members`, `min_required`.
      - Group members must match Stage‑A pool `tf` labels (for PWM inputs, this is the motif ID).
      - Members must be unique across groups; `min_required` must be > 0 and <= group size.
      - For sequence-only inputs (no regulators), set `groups: []`.
    - `min_count_by_regulator` (dict, optional) - per-regulator minimum counts
      - Keys must match group members.
      - DenseGen uses the maximum of this value and `runtime.min_count_per_tf`.
- `plan_templates` (non-empty list; mutually exclusive with `plan`)
  - Use this for combinatorial promoter panels without duplicating YAML.
  - Template keys:
    - `base_name`
    - one quota mode: `quota_per_variant` or `total_quota` (`distribution_policy: uniform` required with `total_quota`)
    - `sampling`, `regulator_constraints`
    - `fixed_elements.promoter_matrix`:
      - `name`
      - `upstream_from_set`, `downstream_from_set` (motif set names)
      - `pairing.mode`: `zip | cross_product | explicit_pairs`
      - `pairing.pairs` (required only when `mode=explicit_pairs`)
      - `spacer_length`, `upstream_pos`, optional `downstream_pos`
- `plan_template_max_expanded_plans` (int > 0; default 256)
  - Hard cap on total expanded plan count across all templates to prevent accidental config blow-ups.
- `plan_template_max_total_quota` (int > 0; default 4096)
  - Hard cap on total quota after template expansion to prevent accidental
    `quota_per_variant x variants` run explosions.
  - Prefer `total_quota` with `distribution_policy: uniform` for large
    combinatorial templates when you want bounded library size.
- `sequence_constraints` (optional)
  - Use this for global final-sequence validation and constrained pad/gap fill.
  - `forbid_kmers[]` rules:
    - `name`
    - `patterns_from_motif_sets` (non-empty list of motif set names)
    - `include_reverse_complements` (bool)
    - `scope`: `outside_allowed_placements`
    - `strands`: `forward | both`
  - `allowlist[]`:
    - `kind`: `fixed_element_instance`
    - `selector.fixed_element`: `promoter`
    - `selector.component`: `upstream | downstream`
    - `match_exact_coordinates` must be true

---

### `densegen.motif_sets`

- Optional dictionary used by `generation.plan_templates` and `generation.sequence_constraints`.
- Shape: `set_name -> { variant_id -> motif_sequence }`.
- Motifs must be non-empty A/C/G/T strings.
- Set names and variant IDs must be non-empty strings.

---

### `densegen.generation.sampling` (Stage-B sampling)

These controls apply to **Stage‑B sampling** (library construction) after Stage‑A input sampling.
`library_size` does not change Stage‑A sampling counts. `library_size` also bounds the motif count
offered to the solver for binding-site and PWM-sampled inputs.
For conceptual behavior (what a library is, coverage/uniqueness enforcement, and resampling), see:
- `../guide/sampling.md`

- `pool_strategy`: `full | subsample | iterative_subsample`
- `library_source`: `build | artifact` (use `artifact` to replay prebuilt libraries)
- `library_artifact_path`: required when `library_source: artifact` (path to `outputs/libraries`;
  must be inside `outputs/` under `densegen.run.root`)
- `library_size` (int > 0; used for subsample strategies)
- `library_sampling_strategy` (Stage‑B): `tf_balanced | uniform_over_pairs | coverage_weighted`
- `coverage_boost_alpha` (Stage‑B; float >= 0; used when `library_sampling_strategy=coverage_weighted`)
- `coverage_boost_power` (Stage‑B; float > 0; used when `library_sampling_strategy=coverage_weighted`)
- `avoid_failed_motifs` (bool; when true, down-weight TFBS that frequently fail solves)
- `failure_penalty_alpha` (float >= 0; penalty strength for failed motifs)
- `failure_penalty_power` (float > 0; penalty exponent for failed motifs)
- `cover_all_regulators` (bool; defaults to false)
- `unique_binding_sites` (bool)
- `unique_binding_cores` (bool; requires `tfbs_core` in pools)
- `max_sites_per_regulator` (int > 0 or null)
- `relax_on_exhaustion` (bool)
- `iterative_max_libraries` (int > 0; only valid when `pool_strategy=iterative_subsample`)
- `iterative_min_new_solutions` (int >= 0; only valid when `pool_strategy=iterative_subsample`)

Notes:
- When `library_source: artifact`, DenseGen replays the libraries found in
  `library_artifact_path` and validates that `pool_strategy`, Stage‑B `library_sampling_strategy`,
  and `library_size` match the artifact metadata. Stage‑B sampling is not rebuilt.

---

### `densegen.solver`

- `backend`: solver name string (required unless `strategy: approximate`).
  - Common values: `CBC`, `GUROBI` (depends on your dense-arrays install).
- `strategy`: `iterate | diverse | optimal | approximate`
- `time_limit_seconds` (float > 0, optional)
- `threads` (int > 0, optional)
- `strands`: `single | double` (default: `double`)
  - `time_limit_seconds` and `threads` are invalid when `strategy: approximate`
  - `threads` is rejected for CBC backends (OR-Tools does not apply it)

---

### `densegen.runtime`

- `round_robin` (bool) - interleave plan items across inputs (one subsample per plan per pass).
  Use this when you have multiple distinct constraint sets (e.g., different fixed sequences) and want
  a single run to advance each plan in turn. This **does not** change Stage‑B sampling logic; it only
  changes scheduling. With `pool_strategy: iterative_subsample`, round‑robin can increase how often
  libraries are rebuilt, so expect additional compute if many plans are active.
- `arrays_generated_before_resample` (int > 0)
- `min_count_per_tf` (int >= 0)
- `max_duplicate_solutions`, `stall_seconds_before_resample`, `stall_warning_every_seconds`
  - `stall_seconds_before_resample` controls how long to wait with no new solutions before resampling; the timer resets on each new solution; `0` disables.
- `max_consecutive_failures`, `max_seconds_per_plan`, `max_failed_solutions`
  - `max_consecutive_failures` stops the run after N consecutive libraries yield zero solutions; `0` disables.
- `leaderboard_every` (int >= 0; 0 disables periodic leaderboard logs)
- `checkpoint_every` (int >= 0; 0 disables run_state checkpoints)
- `random_seed` (int)

---

### `densegen.postprocess.pad`

- `mode`: `off | strict | adaptive`
- `end`: `5prime | 3prime`
- `max_tries` (int > 0)
- `gc.mode`: `off | range | target`
- `gc.min`, `gc.max` (floats in [0, 1]) — target GC range when `gc.mode=range`
- `gc.target`, `gc.tolerance` (floats in [0, 1]) — target center and tolerance when `gc.mode=target`
  - `gc.min_pad_length` (int >= 0) — if the pad length is shorter than this value:
  - `strict` → error
  - `adaptive` → relax GC bounds to `[0, 1]` and record the relaxation

---

### `densegen.logging`

- `log_dir` (required) - directory for log files (relative to config path, must be inside
  `outputs/` under `densegen.run.root`).
- `level` (e.g., `INFO`)
- `suppress_solver_stderr` (bool)
- `print_visual` (bool; show dense-arrays ASCII placement visuals in progress output)
- `progress_style`: `auto | stream | summary | screen` (default `summary`)
  - `auto`: adapt at runtime:
    - interactive + cursor-capable terminal → `screen`
    - interactive + `TERM=dumb` → `stream`
    - non-interactive output → `summary`
  - `stream`: per‑sequence logs (controlled by `progress_every`)
  - `summary`: suppress per‑sequence logs; keep periodic leaderboard summaries
  - `screen`: use in-place Rich `Live` dashboard updates on interactive terminals.
    - Requires a non-`dumb` terminal (`TERM` must support cursor controls). If `TERM=dumb`, DenseGen fails fast with an actionable error.
    - Requires terminal output; if output is redirected/non-interactive, DenseGen fails fast and instructs you to use `summary` or `stream`.
- `progress_every` (int >= 0) - log/refresh interval in sequences (`0` disables per‑sequence logging)
- `progress_refresh_seconds` (float > 0) - minimum seconds between screen refreshes
- `show_tfbs` (bool) - include TFBS sequences in progress output
- `show_solutions` (bool) - include full solution sequences in progress output
- `visuals.tf_colors` (mapping; required when `print_visual: true`)
  - Maps display TF label → Rich color string for colored label lines in the dense-arrays visual.
    Use the display name shown in progress output (e.g., `lexA` for `lexA_CTGT...`,
    `background` for a background-pool input).
  - Missing TFs are an error (no fallback).
  - If promoter constraints add fixed motifs to the optimizer library, include a color for the
    fixed-elements label derived from `promoter_constraints[].name`. Multiple names are joined
    with `+` in the order they appear (e.g., `sigma70_consensus+sigma54`).

Example:

```yaml
densegen:
  logging:
    print_visual: true
    visuals:
      tf_colors:
        lexA: "#66C2A5"
        cpxR: "#FC8D62"
        baeR: "#8DA0CB"
        sigma70_consensus: "#E78AC3"
```

---

### `plots`

- `source`: `usr | parquet` (required if `output.targets` has multiple sinks)
- `out_dir` (optional; default `outputs/plots`; must be inside `outputs/` under `densegen.run.root`)
- `format` (optional; `png | pdf | svg`, default `pdf`)
- `default`: list of plot names to run when `dense plot` is invoked (defaults to all)
- `options`: dict keyed by plot name (strict; unknown options error)
- `style`: global style dict applied to every plot (can be overridden per plot). Common keys:
  - `seaborn_style` (bool; default `true`) — set to `false` if seaborn styles are unavailable.
- `sample_rows`: optional cap on rows loaded for plotting (reads the first N rows for speed)

---

### Minimal example

```yaml
densegen:
  schema_version: "2.9"
  run:
    id: demo
    root: "."
  inputs:
    - name: demo
      type: binding_sites
      # Provide a TF/TFBS table (CSV/Parquet) in your run inputs directory.
      path: inputs/binding_sites.csv
      format: csv

  output:
    targets: [parquet]
    schema:
      bio_type: dna
      alphabet: dna_4
    parquet:
      path: outputs/tables/records.parquet
      deduplicate: true
      chunk_size: 128

  generation:
    sequence_length: 60
    plan:
      - name: default
        quota: 100
        sampling:
          include_inputs: [demo]

  solver:
    backend: CBC
    strategy: diverse
    time_limit_seconds: 5
    strands: double

  runtime:
    round_robin: true
    arrays_generated_before_resample: 20
    min_count_per_tf: 0
    max_duplicate_solutions: 3
    stall_seconds_before_resample: 30
    stall_warning_every_seconds: 15
    max_consecutive_failures: 25
    max_seconds_per_plan: 0
    max_failed_solutions: 0
    leaderboard_every: 50
    random_seed: 42

  postprocess:
    pad:
      mode: strict
      end: 5prime
      gc:
        mode: range
        min: 0.4
        max: 0.6
        target: 0.5
        tolerance: 0.1
        min_pad_length: 0
      max_tries: 2000

  logging:
    log_dir: outputs/logs
    level: INFO
    suppress_solver_stderr: true
    print_visual: false
```

---

@e-south

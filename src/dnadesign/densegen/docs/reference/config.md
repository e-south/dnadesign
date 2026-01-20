## DenseGen Config Reference

This is the strict YAML schema for DenseGen. Unknown keys are errors and all paths resolve
relative to the config file directory. Use this reference for exact field names; see the guide
for conceptual flow.

### Contents
- [Top-level](#top-level) - required roots and plotting.
- [`densegen.inputs[]`](#densegeninputs) - input sources and sampling.
- [`densegen.run`](#densegenrun) - run identifier and root.
- [`densegen.output`](#densegenoutput) - output targets and schema.
- [`densegen.generation`](#densegengeneration) - plan and fixed elements.
- [`densegen.generation.sampling`](#densegengenerationsampling) - library building controls.
- [`densegen.solver`](#densegensolver) - backend and strategy.
- [`densegen.runtime`](#densegenruntime) - retry and guard rails.
- [`densegen.postprocess.gap_fill`](#densegenpostprocessgap_fill) - gap fill policy.
- [`densegen.logging`](#densegenlogging) - log file configuration.
- [`plots`](#plots) - plotting options and defaults.
- [Minimal example](#minimal-example) - smallest runnable config.

---

### Top-level

- `densegen` (required)
- `densegen.schema_version` (required; supported: `2.1`, `2.2`, `2.3`)
- `densegen.run` (required; run-scoped I/O root)
- `plots` (optional; required `source` when `output.targets` has multiple sinks)

---

### `densegen.inputs[]`

Choose one input type per entry:

PWM inputs perform **input sampling** (sampling sites from PWMs) via
`densegen.inputs[].sampling`. This is distinct from **library sampling**
(`densegen.generation.sampling`), which selects a solver library from the realized TFBS pool.

- `type: binding_sites`
  - `path` - CSV, Parquet, or XLSX file
  - `format` - `csv | parquet | xlsx` (optional if extension is `.csv`/`.parquet`/`.xlsx`)
  - `columns.regulator` (default: `tf`)
  - `columns.sequence` (default: `tfbs`)
  - `columns.site_id` (optional)
  - `columns.source` (optional)
  - Empty regulator/sequence rows are errors
  - Duplicate regulator+sequence rows are allowed (use `generation.sampling.unique_binding_sites` to dedupe)
  - Sequences must be A/C/G/T only
- `type: sequence_library`
  - `path` - CSV or Parquet file
  - `format` - `csv | parquet` (optional if extension is `.csv`/`.parquet`)
  - `sequence_column` (default: `sequence`)
  - Empty or null sequences are errors
  - Sequences must be A/C/G/T only
- `type: pwm_meme`
  - `path` - MEME PWM file
  - `motif_ids` (optional list) - choose motifs by ID
  - `sampling` (required):
    - `strategy`: `consensus | stochastic | background`
    - `n_sites` (int > 0)
    - `oversample_factor` (int > 0)
    - `max_candidates` (optional int > 0; caps candidate generation)
    - `max_seconds` (optional float > 0; time limit for candidate generation)
    - `scoring_backend`: `densegen | fimo` (default: `densegen`)
    - `score_threshold` or `score_percentile` (exactly one; **densegen** backend only)
    - `pvalue_threshold` (float in (0, 1]; **fimo** backend only)
    - `selection_policy`: `random_uniform | top_n | stratified` (default: `random_uniform`; fimo only)
    - `pvalue_bins` (optional list of floats; must end with `1.0`) - p‑value bin edges for stratified sampling
    - `pvalue_bin_ids` (deprecated; use `mining.retain_bin_ids`)
    - `mining` (optional; fimo only) - batch/time controls for mining via FIMO:
      - `batch_size` (int > 0; default 100000) - candidates per FIMO batch
      - `max_batches` (optional int > 0) - max batches per motif
      - `max_seconds` (optional float > 0) - max seconds per motif mining loop
      - `retain_bin_ids` (optional list of ints) - select p‑value bins to retain (0‑based indices);
        retained bins are the only bins reported in yield summaries
      - `log_every_batches` (int > 0; default 1) - log per‑bin yield summaries every N batches
    - `bgfile` (optional path) - MEME bfile-format background model for FIMO
    - `keep_all_candidates_debug` (bool, default false) - write raw FIMO TSVs to `outputs/meta/fimo/` for inspection
    - `include_matched_sequence` (bool, default false) - include `fimo_matched_sequence` in TFBS outputs
    - `length_policy`: `exact | range` (default: `exact`)
    - `length_range`: `[min, max]` (required when `length_policy=range`; `min` >= motif length)
    - `trim_window_length` (optional int > 0; trims PWM to a max‑information window before sampling)
    - `trim_window_strategy`: `max_info` (window selection strategy)
    - `consensus` requires `n_sites: 1`
    - `background` selects low-scoring sequences (<= threshold/percentile; or pvalue >= threshold for fimo)
    - FIMO resolves `fimo` via `MEME_BIN` or PATH; pixi users should run `pixi run dense ...` so it is available.
    - Canonical p‑value bins (default): `[1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]`
      (bin 0 is `(0, 1e-10]`, bin 1 is `(1e-10, 1e-8]`, etc.)
    - FIMO runs log per‑bin yield summaries (hits, accepted, selected). If `retain_bin_ids` is set,
      only those bins are reported; otherwise all bins are reported. `selection_policy: stratified`
      makes the selected‑bin distribution explicit for mining workflows.
    - When `mining` is enabled, `max_seconds` caps per‑batch candidate generation while
      `mining.max_seconds` caps the overall mining loop.
- `type: pwm_meme_set`
  - `paths` - list of MEME PWM files (merged into a single TF pool)
  - `motif_ids` (optional list) - choose motifs by ID across files
  - `sampling` (required) - same fields as `pwm_meme`
- `type: pwm_jaspar`
  - `path` - JASPAR PFM file
  - `motif_ids` (optional list) - choose motifs by ID
  - `sampling` (required) - same fields as `pwm_meme`
- `type: pwm_matrix_csv`
  - `path` - CSV matrix with A/C/G/T columns
  - `motif_id` (required) - single motif ID label
  - `columns` (optional) - map columns to `A/C/G/T` (defaults to literal column names)
  - `sampling` (required) - same fields as `pwm_meme`
- `type: pwm_artifact`
  - `path` - per-motif JSON artifact (contract-first; see `docs/reference/motif_artifacts.md`)
  - `sampling` (required) - same fields as `pwm_meme`
- `type: pwm_artifact_set`
  - `paths` - list of per-motif JSON artifacts (one file per motif)
  - `sampling` (required) - same fields as `pwm_meme`
  - `overrides_by_motif_id` (optional dict) - per-motif sampling overrides
- `type: usr_sequences`
  - `dataset` - USR dataset name
  - `root` - USR root path (required; no fallback)
  - `limit` (optional) - max rows
  - Sequences must be A/C/G/T only

Input paths resolve relative to the config file directory.
Output, logs, and plots must resolve inside `densegen.run.root`.

---

### `densegen.run`

- `id` - run identifier (string; required)
- `root` - run root directory (required; config must live inside it)

---

### `densegen.output`

- `schema`: shared output schema (`bio_type`, `alphabet`) used for IDs and validation (required).
- `targets`: list of sinks to write (`usr`, `parquet`)
- When multiple targets are set, outputs must be in sync before a run; mismatches are errors.
- `usr` (required when `targets` includes `usr`)
  - `dataset`, `root`, `chunk_size`, `allow_overwrite`
- `parquet` (required when `targets` includes `parquet`)
  - `path` (file), `deduplicate`, `chunk_size`
  - `path` must be a `.parquet` file (single-file output)
- `output.usr.root` and `output.parquet.path` must be inside `densegen.run.root`

---

### `densegen.generation`

- `sequence_length` (int > 0)
- `sequence_length` must be >= the widest required motif (library TFBS or fixed elements)
- `quota` (int > 0)
- `sampling` (see below)
- `plan` (required, non-empty)
  - Each item: `name`, and either `quota` or `fraction`
  - Mixing quotas and fractions across items is not allowed.
  - `fixed_elements.promoter_constraints[]` supports `name`, `upstream`, `downstream`,
    `spacer_length`, `upstream_pos`, `downstream_pos`
  - Promoter motifs must be non-empty A/C/G/T strings (normalized to uppercase)
  - `fixed_elements.side_biases` supports motif placement preferences:
    - `left`: list of motifs biased toward the 5prime side
    - `right`: list of motifs biased toward the 3prime side
    - Motifs must be A/C/G/T and must exist in the sampled library
  - `required_regulators` (list) - regulators that must appear in each solution when
    `min_required_regulators` is unset (**all-of**).
  - `min_required_regulators` (int > 0, optional) - require at least K distinct regulators
    **in the final sequence**. When set alongside `required_regulators`, those regulators
    become the candidate set (k-of-n). If `required_regulators` is empty, the requirement
    applies to the full regulator pool.
  - `min_count_by_regulator` (dict, optional) - per-regulator minimum counts
    - For regulators listed here, DenseGen uses the maximum of this value and
      `runtime.min_count_per_tf`.

---

### `densegen.generation.sampling`

These controls apply after PWM input sampling. `library_size` does not change PWM sampling counts.
Under schema `2.2+`, `library_size` also bounds the motif count offered to the solver for
binding-site and PWM-sampled inputs.

- `pool_strategy`: `full | subsample | iterative_subsample`
- `library_size` (int > 0; used for subsample strategies)
- `library_sampling_strategy`: `tf_balanced | uniform_over_pairs | coverage_weighted` (schema `2.2+`)
- `coverage_boost_alpha` (float >= 0; used when `library_sampling_strategy=coverage_weighted`)
- `coverage_boost_power` (float > 0; used when `library_sampling_strategy=coverage_weighted`)
- `avoid_failed_motifs` (bool; when true, down-weight TFBS that frequently fail solves)
- `failure_penalty_alpha` (float >= 0; penalty strength for failed motifs)
- `failure_penalty_power` (float > 0; penalty exponent for failed motifs)
- `subsample_over_length_budget_by` (>= 0; reported as a target bp length)
- `cover_all_regulators` (bool)
- `unique_binding_sites` (bool)
- `max_sites_per_regulator` (int > 0 or null)
- `relax_on_exhaustion` (bool)
- `allow_incomplete_coverage` (bool)
- `iterative_max_libraries` (int > 0 when `pool_strategy=iterative_subsample`)
- `iterative_min_new_solutions` (int >= 0)

---

### `densegen.solver`

- `backend`: solver name string (required unless `strategy: approximate`).
  - Common values: `CBC`, `GUROBI` (depends on your dense-arrays install).
- `strategy`: `iterate | diverse | optimal | approximate`
- `options` (list of solver option strings)
  - `options` must be empty when `strategy: approximate`
- `strands`: `single | double` (default: `double`)

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
- `max_resample_attempts`, `max_total_resamples`, `max_seconds_per_plan`, `max_failed_solutions`
- `leaderboard_every` (int >= 0; 0 disables periodic leaderboard logs)
- `checkpoint_every` (int >= 0; 0 disables run_state checkpoints)
- `random_seed` (int)

---

### `densegen.postprocess.gap_fill`

- `mode`: `off | strict | adaptive`
- `end`: `5prime | 3prime`
- `gc_min`, `gc_max`, `max_tries`

---

### `densegen.logging`

- `log_dir` (required) - directory for log files (relative to config path, must be inside
  `densegen.run.root`).
- `level` (e.g., `INFO`)
- `suppress_solver_stderr` (bool)
- `print_visual` (bool)
- `progress_style`: `stream | summary | screen` (default `stream`)
  - `stream`: per‑sequence logs (controlled by `progress_every`)
  - `summary`: suppress per‑sequence logs; keep periodic leaderboard summaries
  - `screen`: clear and redraw a compact dashboard at `progress_refresh_seconds`
- `progress_every` (int >= 0) - log/refresh interval in sequences (`0` disables per‑sequence logging)
- `progress_refresh_seconds` (float > 0) - minimum seconds between screen refreshes

---

### `plots`

- `source`: `usr | parquet` (required if `output.targets` has multiple sinks)
- `out_dir` (optional; default `outputs`; must be inside `densegen.run.root`)
- `format` (optional; `png | pdf | svg`, default `png`)
- `default`: list of plot names to run when `dense plot` is invoked (defaults to all)
- `options`: dict keyed by plot name (strict; unknown options error)
- `style`: global style dict applied to every plot (can be overridden per plot)
- `sample_rows`: optional cap on rows loaded for plotting (reads the first N rows for speed)
  - `seaborn_style: true` requires seaborn styles to be available; otherwise set to `false`.

---

### Minimal example

```yaml
densegen:
  schema_version: "2.3"
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
      path: outputs/dense_arrays.parquet
      deduplicate: true
      chunk_size: 128

  generation:
    sequence_length: 60
    quota: 100
    plan:
      - name: default
        quota: 100

  solver:
    backend: CBC
    strategy: diverse
    options: []
    strands: double

  runtime:
    round_robin: true
    arrays_generated_before_resample: 20
    min_count_per_tf: 0
    max_duplicate_solutions: 3
    stall_seconds_before_resample: 30
    stall_warning_every_seconds: 15
    max_resample_attempts: 3
    max_total_resamples: 500
    max_seconds_per_plan: 0
    max_failed_solutions: 0
    leaderboard_every: 50
    random_seed: 42

  postprocess:
    gap_fill:
      mode: strict
      end: 5prime
      gc_min: 0.4
      gc_max: 0.6
      max_tries: 2000

  logging:
    log_dir: outputs/logs
    level: INFO
    suppress_solver_stderr: true
    print_visual: true
```

# DenseGen Config Reference

This is the **strict** YAML schema for DenseGen. Unknown keys are errors.

## Top-level

- `densegen` (required)
- `densegen.schema_version` (required; supported: `2.1`)
- `densegen.run` (required; run-scoped I/O root)
- `plots` (optional; required `source` when `output.targets` has multiple sinks)

## `densegen.inputs[]`

Choose one input type per entry:

- `type: binding_sites`
  - `path` — CSV or Parquet file
  - `format` — `csv | parquet` (optional if extension is `.csv`/`.parquet`)
  - `columns.regulator` (default: `tf`)
  - `columns.sequence` (default: `tfbs`)
  - `columns.site_id` (optional)
  - `columns.source` (optional)
  - Empty/duplicate regulator+sequence rows are errors
  - Sequences must be A/C/G/T only
- `type: sequence_library`
  - `path` — CSV or Parquet file
  - `format` — `csv | parquet` (optional if extension is `.csv`/`.parquet`)
  - `sequence_column` (default: `sequence`)
  - Empty or null sequences are errors
  - Sequences must be A/C/G/T only
- `type: pwm_meme`
  - `path` — MEME PWM file
  - `motif_ids` (optional list) — choose motifs by ID
  - `sampling` (required):
    - `strategy`: `consensus | stochastic | background`
    - `n_sites` (int > 0)
    - `oversample_factor` (int > 0)
    - `score_threshold` **or** `score_percentile` (exactly one)
    - `consensus` requires `n_sites: 1`
    - `background` selects low-scoring sequences (<= threshold/percentile)
- `type: pwm_jaspar`
  - `path` — JASPAR PFM file
  - `motif_ids` (optional list) — choose motifs by ID
  - `sampling` (required) — same fields as `pwm_meme`
- `type: pwm_matrix_csv`
  - `path` — CSV matrix with A/C/G/T columns
  - `motif_id` (required) — single motif ID label
  - `columns` (optional) — map columns to `A/C/G/T` (defaults to literal column names)
  - `sampling` (required) — same fields as `pwm_meme`
- `type: usr_sequences`
  - `dataset` — USR dataset name
  - `root` — USR root path (required; no fallback)
  - `limit` (optional) — max rows
  - Sequences must be A/C/G/T only

Input paths resolve relative to the config file directory.
Output, logs, and plots must resolve inside `densegen.run.root`.

## `densegen.run`

- `id` — run identifier (string; required)
- `root` — run root directory (required; must exist; config must live inside it)

## `densegen.output`

- `schema`: shared output schema (`bio_type`, `alphabet`) used for IDs and validation (required).
- `targets`: list of sinks to write (`usr`, `parquet`)
- When multiple targets are set, outputs must be **in sync** before a run; mismatches are errors.
- `usr` (required when `targets` includes `usr`)
  - `dataset`, `root`, `chunk_size`, `allow_overwrite`
- `parquet` (required when `targets` includes `parquet`)
  - `path` (directory), `deduplicate`, `chunk_size`
  - `path` must be a directory (DenseGen writes `part-*.parquet` files)
- `output.usr.root` and `output.parquet.path` must be inside `densegen.run.root`

## `densegen.generation`

- `sequence_length` (int > 0)
- `quota` (int > 0)
- `sampling` (see below)
- `plan` (required, non-empty)
  - Each item: `name`, and **either** `quota` **or** `fraction`
  - Mixing quotas and fractions across items is not allowed.
  - `fixed_elements.promoter_constraints[]` supports `name`, `upstream`, `downstream`,
    `spacer_length`, `upstream_pos`, `downstream_pos`
  - Promoter motifs must be non-empty A/C/G/T strings (normalized to uppercase)
  - `fixed_elements.side_biases` supports motif placement preferences:
    - `left`: list of motifs biased toward the 5′ side
    - `right`: list of motifs biased toward the 3′ side
    - Motifs must be A/C/G/T and **must exist in the sampled library**
  - `required_regulators` (list) — regulators that must appear in each solution
  - `min_required_regulators` (int > 0, optional) — require at least K distinct regulators
  - `min_count_by_regulator` (dict, optional) — per‑regulator minimum counts
    - For regulators listed here, DenseGen uses the **maximum** of this value and
      `runtime.min_count_per_tf`.

## `densegen.generation.sampling`

- `pool_strategy`: `full | subsample | iterative_subsample`
- `library_size` (int > 0; used for subsample strategies)
- `subsample_over_length_budget_by` (>= 0)
- `cover_all_regulators` (bool)
- `unique_binding_sites` (bool)
- `max_sites_per_regulator` (int > 0 or null)
- `relax_on_exhaustion` (bool)
- `allow_incomplete_coverage` (bool)
- `iterative_max_libraries` (int > 0 when `pool_strategy=iterative_subsample`)
- `iterative_min_new_solutions` (int >= 0)

## `densegen.solver`

- `backend`: solver name string (required unless `strategy: approximate`).
  - Common values: `CBC`, `GUROBI` (depends on your dense-arrays install).
- `strategy`: `iterate | diverse | optimal | approximate`
- `options` (list of solver option strings)
  - `options` must be empty when `strategy: approximate`
- `strands`: `single | double` (default: `double`)

## `densegen.runtime`

- `round_robin` (bool)
- `arrays_generated_before_resample` (int > 0)
- `min_count_per_tf` (int >= 0)
- `max_duplicate_solutions`, `stall_seconds_before_resample`, `stall_warning_every_seconds`
- `max_resample_attempts`, `max_total_resamples`, `max_seconds_per_plan`, `max_failed_solutions`
- `random_seed` (int)

## `densegen.postprocess.gap_fill`

- `mode`: `off | strict | adaptive`
- `end`: `5prime | 3prime`
- `gc_min`, `gc_max`, `max_tries`

## `densegen.logging`

- `log_dir` (required) — directory for log files (relative to config path, must be inside `densegen.run.root`).
- `level` (e.g., `INFO`)
- `suppress_solver_stderr` (bool)
- `print_visual` (bool)

## `plots`

- `source`: `usr | parquet` (required if `output.targets` has multiple sinks)
- `out_dir` (optional; default `plots`; must be inside `densegen.run.root`)
- `default`: list of plot names to run when `dense plot` is invoked (defaults to all)
- `options`: dict keyed by plot name (strict; unknown options error)
- `style`: global style dict applied to every plot (can be overridden per plot)
- `sample_rows`: optional cap on rows loaded for plotting (reads the first N rows for speed)
  - `seaborn_style: true` requires seaborn styles to be available; otherwise set to `false`.

## Minimal example

```yaml
densegen:
  schema_version: "2.1"
  run:
    id: demo
    root: "."
  inputs:
    - name: demo
      type: binding_sites
      path: inputs/tf2tfbs_mapping_cpxR_LexA.csv
      format: csv

  output:
    targets: [parquet]
    schema:
      bio_type: dna
      alphabet: dna_4
    parquet:
      path: outputs/parquet
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
    random_seed: 42

  postprocess:
    gap_fill:
      mode: strict
      end: 5prime
      gc_min: 0.4
      gc_max: 0.6
      max_tries: 2000

  logging:
    log_dir: logs
    level: INFO
    suppress_solver_stderr: true
    print_visual: true
```

# DenseGen usage demo (end-to-end)

This walkthrough shows the full DenseGen flow:
binding sites in → sampling → solve → sequences out → optional analysis.

DenseGen is decoupled from USR. The demo uses Parquet only.

## 1) Prepare binding sites

DenseGen expects a binding-site table with `tf,tfbs` columns by default (A/C/G/T only).
You can start from the provided example:

```
src/dnadesign/densegen/runs/_campaigns/template/inputs/tf2tfbs_mapping_cpxR_LexA.csv
```

If you generate TFBSs elsewhere (PWM scans, curated motifs), export to the same format.
Sample PWM files live in `src/dnadesign/densegen/runs/_campaigns/pwm_demo/inputs/` for quick testing.

## 2) Create a self-contained run directory (recommended)

```
densegen/
  runs/
    2026-01-14_demo/
      config.yaml
      inputs/
      outputs/
        parquet/
        usr/
      plots/
      logs/
```

Copy the example CSV into `inputs/` and create `config.yaml` in this directory.

## Optional: PWM sampling (low-percentile example)

DenseGen can sample binding sites directly from PWM files. The example below
uses a **low percentile** (background-like) sampling strategy. Swap to
`pwm_meme` or `pwm_matrix_csv` with the same `sampling` block if desired.

```yaml
inputs:
  - name: pwm_demo
    type: pwm_jaspar
    path: inputs/pwm_demo.jaspar
    motif_ids: [DemoTF]
    sampling:
      strategy: background
      n_sites: 200
      oversample_factor: 5
      score_percentile: 10
```

## 3) Write a minimal config

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
    quota: 50
    sampling:
      pool_strategy: subsample
      library_size: 16
      subsample_over_length_budget_by: 120
      cover_all_regulators: true
      unique_binding_sites: true
      max_sites_per_regulator: null
      relax_on_exhaustion: false
      allow_incomplete_coverage: false
      iterative_max_libraries: 50
      iterative_min_new_solutions: 1
    plan:
      - name: sigma70
        quota: 50
        fixed_elements:
          promoter_constraints:
            - upstream: "TTGACA"
              downstream: "TATAAT"
              spacer_length: [16, 18]

  solver:
    backend: CBC
    strategy: diverse
    options: ["TimeLimit=10"]

  runtime:
    round_robin: true
    arrays_generated_before_resample: 10
    min_count_per_tf: 0
    max_duplicate_solutions: 3
    stall_seconds_before_resample: 30
    stall_warning_every_seconds: 15
    max_resample_attempts: 3
    max_total_resamples: 200
    max_seconds_per_plan: 0
    max_failed_solutions: 0
    random_seed: 42

  postprocess:
    gap_fill:
      mode: adaptive
      end: 5prime
      gc_min: 0.40
      gc_max: 0.60
      max_tries: 2000

  logging:
    log_dir: logs
    level: INFO
    suppress_solver_stderr: true
    print_visual: true

plots:
  source: parquet
  out_dir: plots
```

All relative paths resolve against the config directory. There are no fallbacks.

## 4) Validate, plan, describe

```bash
uv run dense validate -c path/to/config.yaml
uv run dense plan -c path/to/config.yaml
uv run dense describe -c path/to/config.yaml
```

## 5) Run generation

```bash
uv run dense run -c path/to/config.yaml
```

## 6) Optional analysis (plots)

```bash
uv run dense plot -c path/to/config.yaml --only tf_usage,tf_coverage
```

Plots are written to `plots/` in the run directory.

## Optional: add USR output

USR is an optional I/O adapter. To write both Parquet and USR:

```yaml
output:
  targets: [usr, parquet]
  schema:
    bio_type: dna
    alphabet: dna_4
  usr:
    dataset: demo_densegen
    root: /path/to/usr/datasets
    allow_overwrite: false
  parquet:
    path: outputs/parquet
    deduplicate: true
```

When multiple outputs are configured, DenseGen requires them to be in sync before writing.

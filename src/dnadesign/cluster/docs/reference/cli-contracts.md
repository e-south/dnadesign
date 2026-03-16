## cluster CLI contracts

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-16

This page collects the stable command, layout, and data-shape contracts for `cluster`.

### Command surface

- `uv run cluster fit --help`
- `uv run cluster umap --help`
- `uv run cluster analyze --help`
- `uv run cluster intra-sim --help`
- `uv run cluster sweep --help`
- `uv run cluster delete-columns --help`

### Dataset and feature-column contract

- Work against a USR dataset (`--dataset`) or a CSV/Parquet file (`--file`).
- `cluster` assumes one explicit chosen feature definition already exists.
- `fit` can consume one vector column (`--x-col`) or a multi-column matrix (`--x-cols`).
- `fit` is method-based: the current supported method id is `leiden`, and method presets live under `presets/method/`.
- Method-specific fit knobs flow through `presets/method/*.yaml`, `job.params.method_params`, or repeated `--method-param key=value`.
- Legacy top-level fit method keys in `job.params` are rejected; keep package-level I/O keys at the top level and method-specific keys under `method_params`.
- `cluster sweep` is method-scoped and requires `--method`; only methods that expose a sweep contract can run it.
- `cluster` does not generate upstream feature columns. The infer-backed workflow is one common route, not the generic package input contract.

### Jobs, presets, and results layout

`cluster` uses two composable config surfaces:

- `presets/`: reusable partials by kind (`method`, `umap`, `plot`, `analysis`)
- `jobs/`: concrete checked-in invocations that bind datasets/files, names, and I/O
- checked-in job path fields such as `file`, `usr_root`, `highlight`, `out`, and `out_dir` resolve relative to the job file itself

Layout:

```text
cluster/
  presets/
    method/
    umap/
    plot/
    analysis/
  jobs/
    <fit_alias>/
      fit.yaml
      umap.yaml
      umap_categorical.yaml
      analyze.yaml
  results/
```

Design-by-contract rule:

- If a job references a preset and also specifies the same method keys, the CLI errors.
- Put reusable method knobs in the preset or in `job.params.method_params`, and use `--method-param key=value` for one-off overrides.

Plotting precedence:

- CLI > `job.plot` > `preset.plot` > defaults

### OPAL-join contract

If requested hues or metrics refer to `obj__/pred__/sel__` fields that are missing from the current dataset, `cluster umap` and `cluster analyze` require:

- `--opal-campaign`
- exactly one of `--opal-run latest|round:<n>|run_id:<rid>` or `--opal-as-of-round <n>`
- optional `--opal-fields` for extra OPAL columns

The CLI resolves the OPAL slice and records which parquet parts were used.

### Results and artifacts

- Checked-in inputs: `jobs/`, `presets/`
- Generated outputs: one writable results root
- Default resolution order:
  - `DNADESIGN_CLUSTER_RESULTS_DIR`
  - nearest project `cluster/results/`
  - `./results` in the current working directory when the command is running outside the built-in package tree
- Run store contents typically include `run.json`, `labels.parquet`, `umap/`, `analysis/`, and `records.md`
- `analysis/analysis.json` records the resolved downstream analysis contract, including any OPAL join inputs, when `cluster analyze` runs
- Override the results directory with `DNADESIGN_CLUSTER_RESULTS_DIR`
- `cluster` fails fast instead of defaulting runtime state under `src/dnadesign/cluster/results`
- Use `--reuse auto|require|reattach` when reusing fit attachments
- Method-scoped sweep outputs are opt-in runtime artifacts under the chosen `--out-dir`, not built-in package assets
- Use `cluster delete-columns --dataset <dataset> --all --write --yes` to remove attached `cluster__*` columns
- When `cluster analyze` omits `--out-dir`, `--cluster-col` must be a fit label column of the form `cluster__<NAME>` so the analysis stays fit-scoped under the run store

### Environment variables

- `DNADESIGN_USR_ROOT`: resolve USR datasets without passing full roots
- `DNADESIGN_OPAL_CAMPAIGNS_ROOT`: resolve OPAL campaign names for joins
- `DNADESIGN_CLUSTER_ROOT`: resolve the writable workspace `cluster/` directory for jobs, presets, and default results
- `DNADESIGN_CLUSTER_RESULTS_DIR`: choose a different results directory

### Troubleshooting

- `gc_content` and `seq_length` require a `sequence` column.
- `intra_sim` requires `cluster__<NAME>__intra_sim`; run `cluster intra-sim` first.
- Numeric hues must be numeric; NaN/Inf rows are dropped with a clear note.
- If OPAL columns are missing, add the required `--opal-*` flags instead of expecting silent fallback behavior.

### Verify next

- Return to [exploratory clustering workflow](../workflows/exploratory-clustering.md) for runnable sequences.
- Use [cluster verification contract](verification.md) when you need the deterministic package-local verification loop.
- Return to [cluster ownership boundary](../concepts/ownership-boundary.md) when you need to decide between `cluster`, upstream feature generation, and OPAL.

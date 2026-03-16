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
- `cluster` assumes one explicit chosen `X` column already exists.
- `fit` can consume one vector column (`--x-col`) or a multi-column matrix when the command surface allows it.
- `cluster` does not generate `infer__...` columns and does not decide which upstream representation should become `X`.

### Jobs, presets, and results layout

`cluster` uses two composable config surfaces:

- `presets/`: reusable partials by kind (`fit`, `umap`, `plot`, `analysis`)
- `jobs/`: concrete checked-in invocations that bind datasets/files, names, and I/O

Layout:

```text
cluster/
  presets/
    fit/
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

- If a job references a preset and also specifies the same algorithm keys, the CLI errors.
- Put reusable knobs in the preset or override them via CLI flags, but do not duplicate them.

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
- Generated outputs: `results/`
- Run store contents typically include `run.json`, `labels.parquet`, `umap/`, `analysis/`, and `records.md`
- Override the results directory with `DNADESIGN_CLUSTER_RESULTS_DIR`
- Use `--reuse auto|require|reattach` when reusing fit attachments
- Use `cluster delete-columns --dataset <dataset> --all --write --yes` to remove attached `cluster__*` columns

### Environment variables

- `DNADESIGN_USR_ROOT`: resolve USR datasets without passing full roots
- `DNADESIGN_OPAL_CAMPAIGNS_ROOT`: resolve OPAL campaign names for joins
- `DNADESIGN_CLUSTER_RESULTS_DIR`: choose a different results directory

### Troubleshooting

- `gc_content` and `seq_length` require a `sequence` column.
- `intra_sim` requires `cluster__<NAME>__intra_sim`; run `cluster intra-sim` first.
- Numeric hues must be numeric; NaN/Inf rows are dropped with a clear note.
- If OPAL columns are missing, add the required `--opal-*` flags instead of expecting silent fallback behavior.

### Verify next

- Return to [exploratory clustering workflow](../workflows/exploratory-clustering.md) for runnable sequences.
- Return to [cluster ownership boundary](../concepts/ownership-boundary.md) when you need to decide between `cluster`, upstream feature generation, and OPAL.

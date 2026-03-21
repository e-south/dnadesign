## Cluster CLI contracts

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
- `uv run cluster workspaces where`
- `uv run cluster workspaces init --help`

### Public execution API

- External tools that need in-process execution should use the public [`../../api.py`](../../api.py) surface instead of importing `dnadesign.cluster.src.*`.
- Supported ad hoc helpers are `run_fit()`, `run_umap()`, `run_analyze()`, `run_sweep()`, and `list_runs()`.
- Supported workspace helpers are `run_fit_workspace()`, `run_umap_workspace()`, `run_analyze_workspace()`, `run_sweep_workspace()`, and `list_workspace_runs()`.
- Workspaces remain the canonical reusable machine config surface, but sibling tools no longer need to synthesize workspace state for one-off file or USR executions.

### Dataset and feature-column contract

- Work against a USR dataset (`--dataset`) or a CSV/Parquet file (`--file`).
- `cluster` assumes one feature column or exported matrix already exists.
- `fit` can consume one vector column (`--x-col`) or a multi-column matrix (`--x-cols`).
- `fit` is method-based: built-in method ids are `leiden` and `kmeans`, and method presets live under `presets/method/`.
- Method-specific fit knobs flow through `presets/method/*.yaml`, workspace `method_params`, or repeated `--method-param key=value`.
- Legacy top-level fit method keys are rejected; keep package-level I/O keys at the top level and method-specific keys under `method_params`.
- `cluster sweep` is method-scoped and requires `--method`; only methods that expose a sweep contract can run it.
- `cluster` does not generate upstream feature columns. The infer-backed workflow is one common route, not the generic package input contract.

### Workspaces, presets, and results layout

`cluster` uses two composable config surfaces:

- `presets/`: reusable partials by kind (`method`, `umap`, `plot`, `analysis`)
- `workspaces/`: canonical checked-in runnable configs that bind datasets/files, names, and I/O
- workspace config path fields such as `file`, `usr_root`, `highlight`, `out`, and `out_dir` resolve relative to the workspace `config.yaml`
- packaged workspace ids are read-only built-in configs; when used directly they materialize runtime outputs under `./workspaces/<id>/outputs/cluster/` from the current working directory

Layout:

```text
cluster/
  workspaces/
    <workspace-id>/
      config.yaml
      outputs/
        cluster/
  presets/
    method/
    umap/
    plot/
    analysis/
```

Design-by-contract rule:

- Workspace config is the canonical reusable machine config surface for `fit`, `umap`, and `analyze`.
- Workspace config rejects unknown top-level, section, and supported nested plot keys; stale or misspelled keys are contract errors, not ignored hints.
- If workspace config references a preset and also specifies the same method keys, the CLI errors.
- Put reusable method knobs in the preset or in `method_params`, and use `--method-param key=value` for one-off overrides.

Plotting precedence:

- CLI > `workspace.<section>.plot` > `preset.plot` > defaults
- `cluster umap` renders PNGs by default, but `--no-plots` or `umap.plot.enabled: false` disables plotting while still recording coords and any requested overlay writes.

### OPAL-join contract

If requested hues or metrics refer to `obj__/pred__/sel__` fields that are missing from the current dataset, `cluster umap` and `cluster analyze` require:

- `--opal-campaign`
- exactly one of `--opal-run latest|round:<n>|run_id:<rid>` or `--opal-as-of-round <n>`
- optional `--opal-fields` for extra OPAL columns

The CLI resolves the OPAL slice and records which parquet parts were used.

### Results and artifacts

- Checked-in reusable inputs: `workspaces/`, `presets/`
- Generated outputs: one explicit writable artifact root
- `fit`, `umap`, `analyze`, `sweep`, and `runs list` require exactly one of:
  - `--workspace <workspace-id|path>`
  - `--results-root <path>`
- Built-in workspace outputs live under `./workspaces/<workspace-id>/outputs/cluster/`
- Run store contents group by stable alias and immutable run slug
- Fit artifacts live under `<results-root>/<alias>/fits/<run-slug>/`
- UMAP artifacts live under `<results-root>/<alias>/umap/<run-slug>/`
- Analysis artifacts live under `<results-root>/<alias>/analysis/<run-slug>/` when `--out-dir` is omitted
- Sweep artifacts live under `<results-root>/<alias>/sweeps/<run-slug>/` when `--out-dir` is omitted
- `analysis/<run-slug>/analysis.json` records the resolved downstream analysis contract, including any OPAL join inputs, when `cluster analyze` runs
- `sweep.json` records the resolved method-scoped sweep contract, and `cluster runs list` includes sweep rows via `sweep_path`
- `cluster` fails fast instead of defaulting runtime state under `src/dnadesign/cluster/` outside an explicitly selected workspace output root, or inferring from the current directory
- All attached overlay columns use one namespace contract for both USR datasets and generic files: `cluster__<run>__...`
- Use `--reuse auto|require|reattach` when reusing fit attachments
- Fit reuse is exact on both input signature and method signature; cached labels are not reused across method-contract changes.
- Repeated runs may reuse one stable alias, but they do not overwrite prior ledger-targeted artifact paths.
- Method-scoped sweep outputs are runtime artifacts under the chosen workspace or explicit `--results-root`; if `--out-dir` is provided, it must stay inside that artifact root
- Use `cluster delete-columns --dataset <dataset> --all --write --yes` to remove attached `cluster__*` columns
- When `cluster analyze` omits `--out-dir`, `--cluster-col` must be a fit label column of the form `cluster__<NAME>` so the analysis stays fit-scoped under the run store

### Environment variables

- `DNADESIGN_USR_ROOT`: resolve USR datasets without passing full roots
- `DNADESIGN_OPAL_CAMPAIGNS_ROOT`: resolve OPAL campaign names for joins

### Workspace utilities

- `uv run cluster workspaces where`: show the built-in workspace source root and the current working directory
- `uv run cluster workspaces init --id my_run --root /tmp`: scaffold one local workspace with `config.yaml` and `outputs/cluster/`
- `uv run cluster workspaces list`: show packaged demo workspace ids
- `uv run cluster workspaces show --help`: inspect one workspace config and artifact root
- `uv run cluster runs list --workspace promoter_clusters_v1`: read a workspace-scoped run ledger without materializing a new run store
- `uv run cluster fit --workspace <workspace-id>`
- `uv run cluster umap --workspace <workspace-id>`
- `uv run cluster analyze --workspace <workspace-id>`

### Troubleshooting

- `gc_content` and `seq_length` require a `sequence` column.
- `intra_sim` requires `cluster__<NAME>__intra_sim`; `cluster intra-sim --cluster-col cluster__<NAME> --write` attaches that exact column.
- Numeric hues must be numeric; NaN/Inf rows are dropped with a clear note.
- If OPAL columns are missing, add the required `--opal-*` flags instead of expecting silent fallback behavior.

### Verify next

- Return to [exploratory clustering workflow](../workflows/exploratory-clustering.md) for runnable sequences.
- Use [cluster verification contract](verification.md) when you need the deterministic package-local verification loop.
- Return to [cluster ownership boundary](../concepts/ownership-boundary.md) when you need to decide between `cluster`, upstream feature generation, and OPAL.

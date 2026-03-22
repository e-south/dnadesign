## Exploratory clustering workflow

**Type:** workflow
**Plane:** downstream-tool
**Owner-boundary:** cluster
**Entry artifact:** one chosen feature column or exported matrix in a USR dataset or file input
**Exit artifact:** cluster run records, embeddings, plots, and downstream exploratory summaries
**Registry-id:** cluster.downstream.exploratory-clustering
**Summary:** Explore one chosen feature column or exported matrix through clustering, UMAP, and downstream summaries.
**Execution-kind:** exploratory
**Progress-kind:** cluster-run-index

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-19

Use this workflow when one feature column or exported matrix already exists and the next task is exploratory clustering, UMAP visualization, or unsupervised summaries.

### Preconditions

- One chosen feature column or exported matrix is already present.
- The input is a USR dataset (`--dataset`) or a file (`--file`) with that chosen feature column or matrix.
- If that feature column or matrix does not exist yet, return to [promoter characterization feature matrix](../../../usr/docs/operations/promoter-characterization-feature-matrix.md) for the shared infer-backed workflow.

### First fit, UMAP, and analysis pass

The examples below use the packaged workspace `promoter_clusters_v1`, which targets the USR dataset `60bp_dual_promoter_cpxR_LexA`.
When you run a packaged workspace id directly, cluster reads the built-in config as a template and writes runtime artifacts under `./workspaces/<id>/outputs/cluster/`, not under the installed package tree.
Outside a source checkout, resolve the dataset via `--usr-root` or `DNADESIGN_USR_ROOT`.

#### 1. Fit

```bash
# Fit one clustering over the chosen feature column using the packaged workspace config.
uv run cluster fit --workspace promoter_clusters_v1
```

If you need to override one method-specific knob without editing the preset, use repeatable `--method-param key=value` flags such as `--method-param resolution=0.9`.

#### 2. UMAP

```bash
# Render one UMAP view over the same chosen feature column.
uv run cluster umap --workspace promoter_clusters_v1
```

For large runs where you only need coordinates or downstream overlays, disable PNG rendering explicitly:

```bash
# Render coordinates only when plots are not needed.
uv run cluster umap --workspace promoter_clusters_v1 --no-plots
```

#### 3. Optional intra-cluster similarity

```bash
# Attach an intra-cluster similarity column for later exploratory hues.
uv run cluster intra-sim \
  --dataset 60bp_dual_promoter_cpxR_LexA \
  --cluster-col cluster__promoter_clusters_v1 \
  --write --allow-overwrite
```

Re-run `cluster umap` after this step when you want the `intra_sim` hue.

#### 4. Analysis

```bash
# Run the default analysis battery for the fitted clustering.
uv run cluster analyze --workspace promoter_clusters_v1
```

### Highlight variants

Single-hue highlight:

```bash
# Render a dedicated highlight view using one ids file.
uv run cluster umap \
  --dataset 60bp_dual_promoter_cpxR_LexA \
  --name promoter_clusters_v1 \
  --x-col infer__evo2_7b__60bp_dual_promoter_cpxR_LexA__logits_mean \
  --preset umap.promoter_set1 \
  --highlight /path/to/ids.parquet \
  --attach-coords \
  --write --allow-overwrite
```

Categorical highlight:

```bash
# Render a categorical highlight view using one ids file plus one highlight column.
uv run cluster umap \
  --dataset 60bp_dual_promoter_cpxR_LexA \
  --name promoter_clusters_v1 \
  --x-col infer__evo2_7b__60bp_dual_promoter_cpxR_LexA__logits_mean \
  --preset umap.promoter_set1 \
  --highlight /path/to/ids_with_round.parquet \
  --highlight-hue-col observed_round \
  --attach-coords \
  --write --allow-overwrite
```

### Optional OPAL-join path

If the hues or metrics you want depend on `obj__/pred__/sel__` columns that are not already in the dataset, `cluster umap` and `cluster analyze` can join them from OPAL.

Required flags:

- `--opal-campaign`
- exactly one of `--opal-run latest|round:<n>|run_id:<rid>` or `--opal-as-of-round <n>`
- optional `--opal-fields` when you need more than the fields implied by the chosen hues or metrics

See [cluster CLI contracts](../reference/cli-contracts.md#opal-join-contract) for exact contract details.

### Standalone direct invocation

If you are not using a checked-in workspace, keep the run store explicit:

```bash
# Fit one standalone clustering run into an explicit results root.
uv run cluster fit \
  --results-root /tmp/cluster-promoter-demo \
  --dataset 60bp_dual_promoter_cpxR_LexA \
  --x-col infer__evo2_7b__60bp_dual_promoter_cpxR_LexA__logits_mean \
  --preset method.leiden.fine \
  --name promoter_clusters_v1 \
  --write --allow-overwrite
```

Standalone method-scoped sweeps use the same explicit artifact-root contract:

```bash
# Sweep one method family into an explicit standalone results root.
uv run cluster sweep \
  --results-root /tmp/cluster-promoter-demo \
  --dataset 60bp_dual_promoter_cpxR_LexA \
  --x-col infer__evo2_7b__60bp_dual_promoter_cpxR_LexA__logits_mean \
  --method leiden
```

If another Python tool needs to run the same flow in-process, use the public helpers in [`../../api.py`](../../api.py) instead of importing `dnadesign.cluster.src.*`. Use workspace helpers for reusable checked-in workspaces and ad hoc helpers for one-off file or USR executions.

### Verify next

- Inspect the workspace artifact root under `workspaces/promoter_clusters_v1/outputs/cluster/promoter_clusters_v1/` for `records.md`, `fits/<run-slug>/run.json`, `umap/<run-slug>/`, and `analysis/<run-slug>/analysis.json`.
- Inspect `workspaces/promoter_clusters_v1/outputs/cluster/<alias>/sweeps/<run-slug>/` for first-class `sweep.json` artifacts when you run `cluster sweep`.
- Confirm `analysis/<run-slug>/analysis.json` captures the resolved analysis request, including any OPAL join campaign or round selectors.
- Confirm the chosen `cluster__<name>` columns and optional coordinate columns exist when `--write` or `--attach-coords` was used.
- If you are editing package code or docs rather than only running the workflow, use the [cluster verification contract](../reference/verification.md) before widening to broader repo checks.
- If exploratory work has answered the structural question and you now need supervised label/train/select, continue with [USR dataset with infer-derived X -> OPAL active learning](../../../opal/docs/workflows/usr-infer-x-active-learning.md).

### Related docs

- [cluster docs index by workflow](../README.md)
- [cluster CLI contracts](../reference/cli-contracts.md)
- [cluster ownership boundary](../concepts/ownership-boundary.md)
- [promoter characterization feature matrix](../../../usr/docs/operations/promoter-characterization-feature-matrix.md)

## Exploratory clustering workflow

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-16

Use this workflow when one explicit feature definition already exists and the next task is exploratory clustering, UMAP visualization, or unsupervised summaries.

### Preconditions

- One explicit chosen feature definition is already present.
- The input is a USR dataset (`--dataset`) or a file (`--file`) with that chosen feature definition.
- If the chosen feature definition does not exist yet, return to [promoter characterization feature matrix](../../../usr/docs/operations/promoter-characterization-feature-matrix.md) for the repository's canonical infer-backed route.

### First fit, UMAP, and analysis pass

The examples below use `promoter_clusters_v1` as a fit alias and a USR dataset named `60bp_dual_promoter_cpxR_LexA`.

#### 1. Fit

```bash
# Fit one clustering over the chosen feature column using the current Leiden method preset.
uv run cluster fit \
  --dataset 60bp_dual_promoter_cpxR_LexA \
  --x-col infer__evo2_7b__60bp_dual_promoter_cpxR_LexA__logits_mean \
  --preset method.leiden.fine \
  --name promoter_clusters_v1 \
  --write --allow-overwrite
```

If you need to override one method-specific knob without editing the preset, use repeatable `--method-param key=value` flags such as `--method-param resolution=0.9`.

Job-driven equivalent:

```bash
# Run the checked-in fit job for the same configuration.
uv run cluster fit --job src/dnadesign/cluster/jobs/promoter_clusters_v1/fit.yaml
```

#### 2. UMAP

```bash
# Render one UMAP view over the same chosen feature column.
uv run cluster umap \
  --dataset 60bp_dual_promoter_cpxR_LexA \
  --name promoter_clusters_v1 \
  --x-col infer__evo2_7b__60bp_dual_promoter_cpxR_LexA__logits_mean \
  --preset umap.promoter_set1 \
  --attach-coords \
  --write --allow-overwrite
```

Job-driven equivalent:

```bash
# Run the checked-in UMAP job for the same configuration.
uv run cluster umap --job src/dnadesign/cluster/jobs/promoter_clusters_v1/umap.yaml
```

#### 3. Optional intra-cluster similarity

```bash
# Attach an intra-cluster similarity column for later exploratory hues.
uv run cluster intra-sim \
  --dataset 60bp_dual_promoter_cpxR_LexA \
  --cluster-col cluster__promoter_clusters_v1 \
  --out-col cluster__promoter_clusters_v1__intra_sim \
  --write --allow-overwrite
```

Re-run `cluster umap` after this step when you want the `intra_sim` hue.

#### 4. Analysis

```bash
# Run the default analysis battery for the fitted clustering.
uv run cluster analyze \
  --dataset 60bp_dual_promoter_cpxR_LexA \
  --cluster-col cluster__promoter_clusters_v1 \
  --preset analysis.promoter_set1
```

Job-driven equivalent:

```bash
# Run the checked-in analysis job for the same fitted clustering.
uv run cluster analyze --job src/dnadesign/cluster/jobs/promoter_clusters_v1/analyze.yaml
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

### Verify next

- Inspect the resolved results root under `<fit_alias>/` for `run.json`, `records.md`, `umap/`, `analysis/`, and `analysis/analysis.json`.
- Confirm `analysis/analysis.json` captures the resolved analysis request, including any OPAL join campaign or round selectors.
- Confirm the chosen `cluster__<name>` columns and optional coordinate columns exist when `--write` or `--attach-coords` was used.
- If you are editing package code or docs rather than only running the workflow, use the [cluster verification contract](../reference/verification.md) before widening to broader repo checks.
- If exploratory work has answered the structural question and you now need supervised label/train/select, continue with [USR dataset with infer-derived X -> OPAL active learning](../../../opal/docs/workflows/usr-infer-x-active-learning.md).

### Related docs

- [cluster docs index by workflow](../README.md)
- [cluster CLI contracts](../reference/cli-contracts.md)
- [cluster ownership boundary](../concepts/ownership-boundary.md)
- [promoter characterization feature matrix](../../../usr/docs/operations/promoter-characterization-feature-matrix.md)

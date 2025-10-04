## cluster

Parquet/CSV‑first **clustering + UMAP + analyses** with a run store, caching, presets, and assertive, decoupled design.

---

### Install
Add to your `pyproject.toml`:
```toml
[project.scripts]
cluster = "dnadesign.cluster.src.cli.app:main"
```

---

### Architecture & workflow
```
CLI (Typer + Rich) ─┬─ io.detect/read/write (USR | Parquet | CSV)
                    ├─ algo.leiden (+ sweep)
                    ├─ umap.compute/plot (hues, highlights)
                    ├─ analysis.(composition|diversity|differential|intra_similarity)
                    ├─ runs.(signatures|store|index|reuse)
                    └─ presets.(loader|schema)
```
1) **Load** the data needed (projection; float32 vectors).  
2) **Validate** (duplicates; id↔sequence bijection; fixed‑width X; finite values).  
3) **Fit / UMAP / Analyze** with **narrative logs** and **progress bars**.  
4) **Attach minimally** (opt‑in extras) and **catalog runs** with signatures.  
5) **Plot & save** under the same **run directory** with config‑encoded filenames.

---

### Path resolution
You can provide either:

* `--dataset <name>` (USR mode, requires `records.parquet` under `<USR_ROOT>/<name>/`)
* `--file /path/to/table.parquet|.csv` (generic mode)

If neither is given, auto‑detect:

* In a folder with `records.parquet`: treat as **USR** if it looks like `usr/datasets/<name>/`, otherwise **generic**.
* USR root comes from `--usr-root` or `DNADESIGN_USR_ROOT` env var.

---

### Column creation policy
* `cluster__<NAME>` (`int32`)
* `cluster__<NAME>__meta` (compact JSON; includes signature)
* opt‑in: `cluster__<NAME>__umap_x/y`, `cluster__<NAME>__intra_sim`

---

### Commands
Try `cluster --help`.

#### `cluster fit`
Run Leiden clustering on a chosen **X** column(s), attach columns, and register a **fit** run:

* attaches: `cluster__<NAME>` and `cluster__<NAME>__meta` (compact JSON)
* optional: per‑row **silhouette quality** (`cluster__<NAME>__quality`) with sampling for big N  
  Use `--silhouette` to enable. By default up to 20k points are scored; pass `--full-silhouette` to score all rows.
* reuse: `--reuse auto|require|never|reattach` (defaults to `auto`)
* write policy: **USR** attaches atomically; **generic** writes `--out` or `--inplace` with `.bak`
* presets: `--preset <name>` (kind: `"fit"`) pre-fills parameters; **explicit flags always win**

#### `cluster umap`
Compute UMAP and save **coords + plots** under the **same run**. Attach coords with `--attach-coords`.

* plot hues: by `cluster`, `gc_content`, `seq_length`, `intra_sim`, `numeric:<col>`, `categorical:<col>`
* highlight by ids file or categorical values
* all plots saved as `cluster/batch_results/<FIT_SLUG>/umap/<UMAP_SLUG>/plots/<HUE>.png`
* presets: `--preset <name>` (kinds: `"umap"` and optionally `"plot"`) to pre-fill neighbors/min_dist/metric and plot settings (`dims`, `alpha`, `size`)

#### `cluster analyze`
Run composition, diversity, and differential enrichment on an existing `cluster__<NAME>`.

* outputs CSVs and optional PNGs
* if `--out-dir` is omitted, outputs land under `batch_results/<FIT_SLUG>/analysis/<GROUP_BY>/`
* plot filenames encode the target, e.g. `composition_proportions__cluster__NAME__by_source.png`

#### `cluster intra-sim`
Compute **mean global alignment similarity** per row within each cluster (requires `sequence`).

* attaches `cluster__<NAME>__intra_sim`
* safeguards for huge clusters: sampling with `--max-per-cluster`, `--sample-if-larger`

#### `cluster sweep`
Leiden **resolution sweep** with replicate seeds; writes a Parquet summary and PNG with **suggested resolution**.

#### Run store + presets
`cluster runs list` shows all fits and UMAPs recorded in `cluster/cluster_log/index.parquet`.
`cluster presets list|show` surfaces built‑in, user, or project presets for **fit**, **umap**, and **plot**.

---

### Quick start
Fit on a USR dataset and attach columns:
```bash
cluster fit \
  --dataset 60bp_dual_promoter_cpxR_LexA \
  --x-col infer__evo2_7b__60bp_dual_promoter_cpxR_LexA__logits_mean \
  --name ldn_v1 \
  --write
```

UMAP + plots + attach coords, reusing that fit:
```bash
cluster umap \
  --dataset 60bp_dual_promoter_cpxR_LexA \
  --name ldn_v1 \
  --x-col infer__evo2_7b__60bp_dual_promoter_cpxR_LexA__logits_mean \
  --color-by cluster --color-by gc_content \
  --attach-coords --write
```

Analyses (defaults to run‑store output layout when `--out-dir` omitted):
```bash
cluster analyze \
  --dataset 60bp_dual_promoter_cpxR_LexA \
  --cluster-col cluster__ldn_v1 \
  --group-by source \
  --composition --diversity --difffeat --plots
```

Resolution sweep:
```bash
cluster sweep \
  --dataset 60bp_dual_promoter_cpxR_LexA \
  --x-col infer__evo2_logits_mean_pooled \
  --neighbors 15 \
  --min 0.05 --max 1.0 --step 0.05 \
  --replicates 5 --seeds 1,2,3,4,5 \
  --out-dir cluster/batch_results/sweeps/lexA
```

Use the plot presets and produce one PNG per hue:

```bash
cluster umap \
  --dataset 60bp_dual_promoter_cpxR_LexA \
  --name ldn_v1 \
  --x-col infer__evo2_7b__60bp_dual_promoter_cpxR_LexA__logits_mean \
  --preset umap.promoter_set1 \
  --opal-campaign prom60-etoh-cipro-andgate \
  --opal-run latest \
  --attach-coords --write --allow-overwrite
```

Each hue yields a file like:

```
cluster/cluster_log/<FIT_SLUG>/umap/<UMAP_SLUG>/plots/<name>.<hue>.png
```

> Hues you can use anywhere: `cluster`, `gc_content`, `seq_length`, `intra_sim`, `numeric:<col>`, `categorical:<col>`.

Attach OPAL metrics on‑the‑fly and summarize per cluster:

```bash
cluster analyze \
  --dataset 60bp_dual_promoter_cpxR_LexA \
  --cluster-col cluster__ldn_v1 \
  --preset analysis.promoter_set1
```

Outputs go by default to:

```
cluster/cluster_log/ldn_v1/analysis/source/
  numeric_summary__cluster__ldn_v1.csv
  numeric_violin__cluster__ldn_v1__infer__evo2_7b__...__ll_mean.png
  numeric_violin__cluster__ldn_v1__opal__prom60-etoh-cipro-andgate__latest_pred_scalar.png
  numeric_violin__cluster__ldn_v1__obj__logic_fidelity.png
  numeric_violin__cluster__ldn_v1__obj__effect_scaled.png
```

**Note on caching vs attached columns**

* The **run store** (`cluster_log`) caches labels, coords, plots, and an index. Deleting it removes cached artifacts but **does not** remove any columns attached to your dataset.
* The **dataset itself** (USR `records.parquet`) may have columns like `cluster__<NAME>` already attached. That’s why `cluster umap` still works even if you delete the run store: UMAP recomputes from `X`, and hues like `cluster` come from the dataset column.
* You can relocate the run store via `DNADESIGN_CLUSTER_RUNS_DIR=/path/to/central/dir`.

---

### Presets

Presets reduce CLI flag bloat and keep **sensible defaults** nearby:

* search order: **project** (`./cluster/presets/*.yaml`) → **user** (`~/.dnadesign/cluster/presets/*.yaml`) → **built‑ins**
* kinds: `"fit"` (Leiden params), `"umap"` (neighbors/min_dist/metric), `"plot"` (alpha/size/legend)
* CLI flags always **override** preset values.

Fit preset example:
```yaml
# cluster/presets/leiden.fine.yaml
name: "leiden.fine"
kind: "fit"
params: { neighbors: 20, resolution: 0.8, scale: false, metric: "euclidean" }
```
Then run:
```bash
cluster fit --dataset <ds> --x-col <X> --preset leiden.fine --name ldn_v2 --write
```

You can also color UMAPs by any numeric column via `numeric:<col>`, any categorical column via `categorical:<col>`, or the current cluster labels via `cluster`. Presets support multiple hue specs via `plot.color_by` (list). 

For example:
```yaml
# cluster/presets/plot.paper.umap.yaml
name: "plot.paper.umap"
kind: "plot"
plot:
  color_by:
    - "cluster"
    - "gc_content"
    - "numeric:infer__evo2_7b__60bp_dual_promoter_cpxR_LexA__ll_mean"
    - "numeric:opal__prom60-etoh-cipro-andgate__latest_pred_scalar"
    - "numeric:obj__logic_fidelity"
    - "numeric:obj__effect_scaled"
  alpha: 0.6     # also overridable by --alpha
  size: 4.0
  dims: [12, 12]
  font_scale: 1.2
```

---

@e-south

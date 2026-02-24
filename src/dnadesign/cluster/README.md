## cluster
This CLI clusters sequences, computes UMAP embeddings, renders plots for many **hues** (cluster labels, GC, numeric/categorical columns, highlights), and runs post-hoc analyses (composition, diversity, differential, numeric summaries). It records everything in a **run store** (`results/`) and lets you reuse runs.

The system is built around two *composable* concepts:

- **Presets** — reusable partials by **kind** (`fit`, `umap`, `plot`, `analysis`) that capture algorithm knobs and plotting style;
- **Jobs** — concrete, checked‑in invocations you can run repeatedly (dataset bindings, names, I/O, highlights, etc.).

---

## Install

Add to `pyproject.toml`:

```toml
[project.scripts]
cluster = "dnadesign.cluster.cli:main"
````

---

## Concepts at a glance

* **Dataset vs file**: Work against a USR dataset (`--dataset`) or a CSV/Parquet (`--file`). The CLI autodetects if you run inside a `records.parquet` folder.
* **Fit**: Run Leiden clustering on an X matrix (vector column or multi‑column).
* **UMAP**: Compute embeddings, attach coordinates (optional), and render one PNG per **hue** (coloring).
* **Analysis**: Composition/diversity/differential and numeric summaries/violin plots.
* **Hues**:

  * Built‑ins: `cluster`, `gc_content`, `seq_length`, `intra_sim`
  * Your data: `numeric:<col>`, `categorical:<col>`
  * Special: `highlight` (single or *categorical* highlight by a column in a supplied ids file)
* **OPAL joins**: If a hue references `obj__/pred__/sel__` columns not present in your dataset, the CLI can join them from an OPAL campaign.
* **Run store**: Repeatable outputs live under `results/<FIT>/…`, with an `index.parquet` and per‑run `records.md`.

---

## Folder layout

```
cluster/
  presets/
    fit/         # Leiden hyperparameters (neighbors, resolution, ...)
    umap/        # UMAP hyperparameters (+ which hues to render)
    plot/        # Visual style blocks (optional; can be referenced by umap presets)
    analysis/    # Analysis batteries / options
  jobs/
    <fit_alias>/
      fit.yaml
      umap.yaml
      umap_categorical.yaml   # (new) categorical highlight demonstration
      analyze.yaml
  results/       # run store (auto‑managed)
```

**Presets** = reusable partials by **kind**; **Jobs** = concrete invocations you commit to your repo.

|                   | Presets                                 | Jobs (checked‑in)                             |
| ----------------- | --------------------------------------- | --------------------------------------------- |
| Purpose           | Defaults & style (portable)             | Bind dataset/file/name & run options          |
| Scope             | kind ∈ {fit, umap, plot, analysis}      | command ∈ {fit, umap, analyze}                |
| May contain       | Algo knobs; hues; style (plot)          | I/O, dataset/file, run alias, highlight path  |
| Must **not** have | dataset names, file paths, run aliases  | algorithm knobs **if** a preset is referenced |
| Precedence        | CLI > job.plot > preset.plot > defaults |                                               |

> **Design by contract:**
> If a job references a preset and *also* specifies the same algorithm keys, the CLI errors. Move knobs into the preset or pass via CLI flags.

---

## Hues & highlighting

You can request the following hue specs (in `umap` presets/jobs or CLI):

* Built‑ins:

  * `cluster` — colors by `cluster__<NAME>`
  * `gc_content` — requires `sequence` column
  * `seq_length` — requires `sequence` column
  * `intra_sim` — requires `cluster__<NAME>__intra_sim` (compute via `cluster intra-sim`)
* Your data:

  * `numeric:<col>` — any numeric column (strictly validated; NaN/Inf rows dropped with a clear note)
  * `categorical:<col>` — string/integer categories (legend capped by `legend.max_items`)
* Special:

  * `highlight` — *dedicated* highlight plot (gray background + highlighted ids)
    and optional **overlay rings** on other hues.

Put your full **set of hues** in the UMAP preset (`plot.color_by: [...]`).
Each hue yields a file like:

```
results/<FIT_ALIAS>/umap/<fit>.<hue>.png
```

### Highlight modes and controls

* **Single‑hue highlight** (IDs only): dedicated `highlight` plot shows *filled* points (red by default).
* **Categorical highlight** (`--highlight-hue-col <col>`): dedicated `highlight` plot shows *filled*, color‑coded categories + legend.
* **Overlays on other hues**: Optional **rings** around highlighted ids. Configure via:

```yaml
plot:
  highlight:
    overlay: true|false   # overlays rings on every non-"highlight" hue
    size: 60.0            # size of ring overlays (not the filled highlight plot)
    marker: "o"
    facecolor: "none"
    edgecolor: "red"      # or provide a 'palette' when using categorical highlight
    legend: false
```

> If you don’t want highlight rings on other hues (e.g., `gc_content`), set `overlay: false`.

---

## OPAL joins (predictions & objectives)

If any requested hue refers to `obj__/pred__/sel__` columns that are *not* in your dataset, `cluster umap/analyze` will **assertively** require an OPAL campaign and run selector:

* `--opal-campaign` — path or campaign name (resolvable via `$DNADESIGN_OPAL_CAMPAIGNS_ROOT` or repo’s `dnadesign/opal/campaigns/`)
* Exactly one of:

  * `--opal-run latest|round:<n>|run_id:<rid>` or
  * `--opal-as-of-round <n>`
* `--opal-fields` — add extra OPAL fields beyond what your hues/metrics require

The CLI logs the exact OPAL parquet parts it used and the resolved slice.

---

## Quick recipes (copy/paste)

> The examples below use `ldn_v1` as a fit alias and a USR dataset named `60bp_dual_promoter_cpxR_LexA`.

### 1) Fit

**CLI**

```bash
cluster fit \
  --dataset 60bp_dual_promoter_cpxR_LexA \
  --x-col infer__evo2_7b__60bp_dual_promoter_cpxR_LexA__logits_mean \
  --preset fit.leiden.fine \
  --name ldn_v1 \
  --write --allow-overwrite
```

**Job**

```bash
cluster fit --job cluster/jobs/ldn_v1/fit.yaml
```

---

### 2A) UMAP — single‑hue highlight (IDs only)

**CLI**

```bash
cluster umap \
  --dataset 60bp_dual_promoter_cpxR_LexA \
  --name ldn_v1 \
  --x-col infer__evo2_7b__60bp_dual_promoter_cpxR_LexA__logits_mean \
  --preset umap.promoter_set1 \
  --highlight /path/to/ids.parquet \
  --attach-coords --write --allow-overwrite
```

**Job**

```bash
cluster umap --job cluster/jobs/ldn_v1/umap.yaml
```

This yields a **dedicated `highlight` plot** (gray background + filled red highlights) and, if `overlay: true`, **ring overlays** on all other hues.

---

### 2B) UMAP — **categorical** highlight (new)

Assume `/path/to/ids_with_round.parquet` has columns:

* `id`
* `observed_round` (ints like 0,1,2,… — treated as categories)

**CLI**

```bash
cluster umap \
  --dataset 60bp_dual_promoter_cpxR_LexA \
  --name ldn_v1 \
  --x-col infer__evo2_7b__60bp_dual_promoter_cpxR_LexA__logits_mean \
  --preset umap.promoter_set1 \
  --highlight /path/to/ids_with_round.parquet \
  --highlight-hue-col observed_round \
  --attach-coords --write --allow-overwrite
```

**Job**

```bash
cluster umap --job cluster/jobs/ldn_v1/umap_categorical.yaml
```

The dedicated `highlight` plot shows filled, color‑coded categories with a legend titled `highlight by observed_round`. On other hues, **overlay rings** are colored per category (set `overlay: false` to disable). You can supply a palette:

```yaml
plot:
  highlight:
    legend: true
    palette:
      "0": "#1f77b4"
      "1": "#ff7f0e"
      "2": "#2ca02c"
# or: palette: "tab20"
```

---

### 3) (Optional) Intra‑cluster similarity → enables `intra_sim` hue

```bash
cluster intra-sim \
  --dataset 60bp_dual_promoter_cpxR_LexA \
  --cluster-col cluster__ldn_v1 \
  --out-col cluster__ldn_v1__intra_sim \
  --write --allow-overwrite
```

Re-run UMAP to render the `intra_sim` hue plot.

---

### 4) Analysis battery

**CLI**

```bash
cluster analyze \
  --dataset 60bp_dual_promoter_cpxR_LexA \
  --cluster-col cluster__ldn_v1 \
  --preset analysis.promoter_set1
```

**Job**

```bash
cluster analyze --job cluster/jobs/ldn_v1/analyze.yaml
```

Outputs go to `results/ldn_v1/analysis/` unless overridden.

---

### 5) Resolution sweep (helper)

```bash
cluster sweep \
  --dataset 60bp_dual_promoter_cpxR_LexA \
  --x-col infer__evo2_7b__60bp_dual_promoter_cpxR_LexA__logits_mean \
  --neighbors 30 --res-min 0.05 --res-max 1.0 --step 0.05 \
  --replicates 5 --out-dir ./sweep_ldn_v1
```

---

## Jobs — schema & precedence

Jobs are checked‑in YAMLs with **one** command and a **params** block. Put re‑usable style into presets.

**Precedence for plotting**: CLI > `job.plot` > `preset.plot` > defaults.

**Schema snippets** (see concrete jobs below):

```yaml
# jobs/<fit_alias>/fit.yaml
command: "fit"
params:
  dataset: "..."
  name: "ldn_v1"
  x-col: "..."                # or: x-cols: "col1,col2,col3"
  preset: "fit.leiden.fine"   # do NOT duplicate algo keys here

# jobs/<fit_alias>/umap.yaml
command: "umap"
params:
  dataset: "..."
  name: "ldn_v1"
  x-col: "..."
  preset: "umap.promoter_set1"
  highlight: "/abs/path/to/ids.parquet"   # omit if not needed
  attach_coords: true
  write: true
  allow_overwrite: true
plot:
  # lightweight overrides here (prefer presets for reusables)
  dims: [14, 10]
  size: 6.0
  alpha: 0.6

# jobs/<fit_alias>/analyze.yaml
command: "analyze"
params:
  dataset: "..."
  cluster_col: "cluster__ldn_v1"
  preset: "analysis.promoter_set1"
```

---

## Presets (fit, umap, plot, analysis)

* **Fit** presets declare Leiden knobs (neighbors, resolution, metric, etc.).
* **UMAP** presets declare UMAP knobs *and* the hue list (`plot.color_by`).
* **Plot** presets are optional style blocks you can reference from a UMAP preset.
* **Analysis** presets declare which batteries to run and which numeric columns to summarize.

> If you reference a preset in a job, **do not** repeat those algo keys in the job.

---

## Run store, reuse, and cleanup

* All runs live under `results/` (override via `$DNADESIGN_CLUSTER_RESULTS_DIR`).
* `results/<FIT>/` contains `run.json`, `labels.parquet`, `umap/`, `analysis/`, and `records.md` logging effective parameters.
* To **reuse** fit attachments without recompute, use `--reuse auto|require|reattach` (see `cluster fit --help`).
* To remove attached `cluster__*` columns from a dataset/file:

```bash
cluster delete-columns --dataset 60bp_dual_promoter_cpxR_LexA --all --write --yes
```

---

## Troubleshooting & tips

* **“My red highlight points don’t get larger when I change size”**
  Fixed: the dedicated `highlight` plot now respects `plot.highlight.size`, `marker`, and `alpha`. Previously a hidden `1.2×` multiplier was always used. See “Bug fix” below.
* **“Highlight changes apply to other hues like `gc_content`”**
  That’s the *overlay rings* feature. Disable overlays by setting:

  ```yaml
  plot:
    highlight:
      overlay: false
  ```
* **Hue failures**

  * `gc_content`/`seq_length` require a `sequence` column and now fail clearly if missing.
  * `intra_sim` requires `cluster__<NAME>__intra_sim` — run `cluster intra-sim` first.
  * Numeric hues must be strictly numeric; non‑finite rows (NaN/Inf) are dropped with a concise log.
* **OPAL columns not found**
  If hues reference `obj__/pred__/sel__` and those columns are missing, provide `--opal-campaign` and a run selector. The CLI prints exactly which OPAL parts and slice it used.

---

## Environment variables

* `DNADESIGN_USR_ROOT` — resolve USR datasets without `--usr-root`.
* `DNADESIGN_OPAL_CAMPAIGNS_ROOT` — resolve campaign names for OPAL joins.
* `DNADESIGN_CLUSTER_RESULTS_DIR` — choose a different results directory.

---

## Changelog (core UX relevant)

* Dedicated `highlight` plot respects `plot.highlight.size`/`marker`/`alpha`.
* Assertive `gc_content`/`seq_length` (error if `sequence` missing).
* Clearer UMAP preset examples and job files for single & categorical highlights.

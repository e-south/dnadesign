## OPAL Plots

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-26


Plot plugins own their rendering, but their public contract is shape-first metadata: required sources, required columns, tidy output schema, failure modes, and artifact manifests.

### How it works

- Prefer a separate plots config file (e.g., `plots.yaml`) and reference it from `campaign.yaml` via `plot_config`.
- Run plots with:

```bash
uv run opal plot --config /path/to/campaign.yaml \
  [--plot-config /path/to/plots.yaml] \
  [--round latest|all|3|1,3,7|2-5] \
  [--name my_plot] \
  [--tag my_tag]
```

- Discover available plot kinds:

```bash
uv run opal plot --list
uv run opal plot --list-config --config /path/to/campaign.yaml
uv run opal plot --describe scatter_score_vs_rank
uv run opal plot --list --json
uv run opal plot --describe metric_over_rounds --json
```

### Minimal YAML schema

**campaign.yaml**

```yaml
plot_config: plots.yaml
```

**plots.yaml**

```yaml
plot_defaults:
  output:
    format: "png"
    dpi: 600

plot_presets:
  fold_change_base:
    kind: fold_change_vs_logic_fidelity
    params:
      intensity_log2_offset_delta: 0.0
      y_axis: score

plots:
  - name: score_vs_rank_latest        # unique instance label
    kind: scatter_score_vs_rank       # plugin id registered in plots registry
    round_selector: latest            # optional plot-local override for --round
    round_variants: [latest, each]    # optional manifest fan-out for dropdown scopes

    # Optional extra sources (built-ins auto-injected: records, outputs)
    data:
      - name: extra_csv
        path: ./extras/scores.csv

    # Opaque, plugin-specific params (required for plot knobs)
    params:
      score_field: "pred__score_selected"  # default selected score field from run_pred
      surface_label: "Selected objective score: Score = -MSE(y_hat, target)"
      metric_label: "Score = -MSE(y_hat, target)"  # optional objective-scale axis/notebook label
      legend_metric_label: "negative MSE score"    # optional compact legend label
      metric_expression: "score = -MSE(y_hat, target); MSE = d^-1 sum_c((y_hat_c - target_c)^2)"
      y_axis:
        scale_class: densegen_plan_logic4_negative_mse
        limits: [-0.25, 0.0]
        include_zero_tick: true
      hue: null                       # or "round"
      highlight_selected: false

    # Optional output tuning
    output:
      dir: "{campaign}/plots/{kind}/{name}"  # {campaign|workdir|kind|name|round_suffix}
      filename: "{name}{round_suffix}.png"
      dpi: 600
      format: "png"                   # png|svg|pdf (png default)
      save_data: false                # save tidy CSV next to the image

  - name: fold_change_numeric
    preset: fold_change_base
    params:
      hue: pred__score_selected
      cbar: true
```

**Notes:**
- Plotting knobs must live under `params:`. Top‑level plotting keys are rejected.
- Use plot-level `round_selector:` only for plot scope, not rendering knobs. It accepts
  the same selectors as `--round` and lets all-round bundles include primitives that
  are inherently single-round. `--run-id` remains single-round and takes precedence.
- Use `round_variants:` only when one configured plot should write multiple
  manifest-backed scopes. Valid entries are normal round selectors plus
  `configured` and `each`; `each` expands from `outputs/ledger/runs.parquet`
  and fails visibly when no run ledger exists. Do not combine `round_variants`
  with `--run-id`.
- `scatter_score_vs_rank` should use `pred__score_selected` unless you intentionally target another ledger metric.
- Vector heatmaps are generic vector plots: campaigns should pass display-ready
  `channel_labels`, `value_label`, `reference_mse_metric_label`, and
  `reference_mse_expression` rather than relying on raw channel slugs.
- Use `enabled: false` to keep a plot entry without running it.
- Presets merge into each plot entry; entry values override preset values.
- Inline `plots:` in campaign.yaml is still supported, but `plot_config` keeps runtime config lean.
- `data:` paths are resolved relative to the plots YAML that declares them.
- `sfxi_logic_fidelity_closeness` is strict by default (`on_violin_invalid: error`); set
  `params.on_violin_invalid: line` or `params.violin: false` explicitly for small sample sizes.

**Built-ins injected** (resolved from the campaign config):

* `records` → resolved from `data.location` in `campaign.yaml`
* `outputs` → campaign `outputs/` directory
* `ledger_predictions_dir` → `outputs/ledger/predictions/`
* `ledger_runs_parquet` → `outputs/ledger/runs.parquet`
* `ledger_labels_parquet` → `outputs/ledger/labels.parquet`

Ledger sinks always live under `context.workspace.outputs_dir` (e.g., `outputs/ledger/`).

---

### Artifact manifests

Configured plots are manifest-backed artifacts. Each plot attempt writes:

- rendered media such as `score_vs_rank.png`;
- optional tidy CSVs when `output.save_data: true` and the plugin calls
  `context.save_df(...)`;
- a per-plot manifest named like `<plot-stem>.manifest.json`;
- an aggregate `plot_manifest.json` index in the output directory.

The per-plot manifest uses schema `opal.plot_artifact.v1` and records:

| field | purpose |
| --- | --- |
| `name`, `kind` | configured plot instance and registered plot primitive |
| `status` | `written` or `failed` for current plot attempts; future schemas may add explicit `skipped` or `stale` states |
| `run_id`, `rounds` | explicit run/round scope used for input resolution |
| `params` | merged plot parameters after defaults and presets |
| `inputs` | resolved built-in and custom data paths with file size and mtime |
| `outputs.media` | rendered image/SVG/PDF files |
| `outputs.tidy_csv` | tidy CSV files saved by the plugin |
| `metadata` | `PlotMeta` summary, capability, data shape, tidy schema, and failure modes |
| `quality` | tidy CSV schema validation status when a plot declares `metadata.tidy_schema` |
| `freshness` | mtime-based freshness summary for resolved inputs and outputs |
| `caption`, `review_purpose` | manifest-backed human purpose text; plot params can override generic metadata text |
| `warnings`, `error` | structured nonfatal and fatal plot outcomes |

Review and generated notebook surfaces should read manifests first. Extra files
on disk are advisory only; they can trigger stale-file warnings, but they are
not current evidence unless referenced by the active manifest.

When `round_variants` expands one configured plot into multiple artifacts, the
aggregate index contains multiple `opal.plot_artifact.v1` manifests with the
same `name` and different `rounds` / filename suffixes. Notebook visual
surfaces group those rows under one plot-level dropdown and expose a second
manifest-backed plot-scope dropdown only when multiple scopes exist.

Plot capability metadata is the dropdown contract. It records objective family,
data layer, round behavior, label requirement, model-artifact requirement, and
tidy-data availability so notebooks can distinguish configured-but-missing,
generated-current, generated-stale, and stale-unmanifested surfaces.

---

### SFXI diagnostics plots

These objective-specific plots reuse shared SFXI math and are safe to run without retraining.
Diagnostic plots always render the full dataset; sampling parameters are not supported.

### Plot kinds + params

- **`feature_importance_bars`**: overlays per-round model feature importances from
  `outputs/rounds/round_<k>/model/feature_importance.csv`.
  - params: `order_policy` (`preserve|sort_index`), `alpha`, `figsize_in`
  - requires feature importance artifacts (for example RF with `emit_feature_importance: true`)
- **`feature_importance_heatmap`**: matrix heatmap of feature importance over rounds.
  - rows are stable feature IDs, columns are rounds, values are importances;
    default `order_policy: sort_index` preserves the full ordinal feature axis
    for dense X surfaces such as the 8192-D LatentDNA/Evo2 candidate table
  - params: `order_policy` (`preserve|sort_index|max_importance`),
    optional debugging `top_n`, `figsize_in`, `cmap`, `rasterized`
  - writes tidy CSV columns `round`, `feature_id`, `importance`, `rank`,
    and `source_path`
- **`metric_over_rounds`**: scalar summary over rounds for selected/top-k/pool cohorts.
  - params: `metric`, `cohorts`, `summaries`, `top_k`, `threshold`,
    `y_axis`, `y_limits`, `y_reference_lines`, `highlight_round`,
    `figsize_in`
  - default summary is `mean`. Add `median` only when it answers a stated
    review question; add `count` when the visible cohort size matters.
  - objective-scale plots should declare `metric_label`,
    `legend_metric_label`, and `metric_expression` in `params`. Add
    `surface_label` when the notebook dropdown needs more objective detail than
    the plot title can carry without clipping. OPAL carries those strings into
    axes, legends, manifests, notebook alt text, campaign visual choices, and
    campaign-set visual manifests instead of presenting a vague selected-score label.
  - use `y_axis.scale_class`, `y_axis.limits`, `y_axis.reference_lines`, and
    `y_axis.include_zero_tick` when a compatible campaign family has a known
    objective scale. For example, negative MSE scores can fix zero as the best
    possible score while still making small null spikes visible in their full
    objective-scale context.
  - `pred__score_selected` is the selected score from the configured objective.
    Its units are objective-specific; compare it within compatible campaigns,
    not as a shared effect-size scale across unrelated label families.
  - `band: iqr` is a within-campaign cohort distribution band, not a
    multi-seed confidence interval. Campaign-set notebooks consume
    materialized collection visual manifests derived from strict
    `opal.campaign_collection.v2` comparison views. Per-set views show matched
    campaign pairs without an interval unless replicate units are present and
    declared. Collection views that aggregate across units must label IQR bands
    and Student-t mean confidence intervals separately.
  - writes tidy CSV columns `round`, `cohort`, `metric`, `summary`, and
    `value`
- **`percent_high_activity_over_rounds`**: thresholded scalar distribution plus
  optional percent-above-threshold line over rounds.
  - params: `metric`, `threshold`, `mode`, `hue`, `size_by`,
    `highlight_round`, `figsize_in`
- **`vector_summary_heatmap`**: vector-channel summary over rounds.
  - rows are an optional reference vector followed by chronological rounds
  - params: `vector_field`, `cohorts`, `channel_labels`,
    `include_reference_vector`, `reference_vector`, `reference_label`,
    `aggregation`, `top_k`, `figsize_in`, `cmap`, `reference_mse_panel`
  - writes tidy CSV columns `row_type`, `round`, `cohort`, `channel`, and
    `value`
  - tidy CSV reference rows are expected only when
    `include_reference_vector` is enabled
  - when `reference_mse_panel` is enabled, reference-MSE rows are the MSE
    between the selected cohort mean vector and the declared reference vector.
    Use `reference_mse_metric_label`, `reference_mse_expression`,
    `reference_mse_y_limits`, and `reference_mse_reference_lines` only when the
    MSE panel should carry an explicit loss expression and comparable scale.
    They are model-output diagnostics, not replicate confidence intervals.
  - use `reference_vector` for any explicit vector baseline. Its length must
    match the plotted vector, whether the campaign uses a four-channel logic
    vector, a count vector, or a measured SFXI vector.

- **`sfxi_factorial_effects`**: factorial-effects map from predicted logic vectors.
  - params: `size_by` (default `obj__effect_scaled`), `include_labels`, `rasterize_at`
- **`fold_change_vs_logic_fidelity`**: SFXI tradeoff scatter for logic fidelity
  against fold-change/effect/score.
  - params: `y_axis`, `hue_field`, `size_by`, `alpha`
  - optional reference overlay params: `reference_overlay: true`,
    `reference_collection_id`, `reference_campaign_id`,
    `reference_batch_id`, `reference_metric_id`, and `reference_y_axis`
  - reference overlays are loaded from materialized `sfxi_ref__...` columns in
    the campaign records table and are validated against OPAL's public SFXI API
    version, objective name, state order, active setpoint vector, finite metric
    values, and non-empty metric provenance before plotting.
- **`sfxi_setpoint_sweep`**: objective landscape across discrete setpoints (current-round labels).
  - rendered as a heatmap with setpoints as columns (vector labels) and diagnostic metrics as rows.
  - rows report median `logic_fidelity`, median `effect_scaled`, and median `score` over observed labels.
  - uses `logic_exponent_beta`, `intensity_exponent_gamma`, and `intensity_log2_offset_delta` from `objective__params`.
  - params: `y_col` (default `y_obs`), `percentile`, `min_n`, `eps`, `delta`
- **`sfxi_support_diagnostics`**: distance-to-labeled-logic vs score (OOD check).
  - params: `y_axis`, `hue`, `batch_size`
- **`sfxi_uncertainty`**: uncertainty vs score (artifact model; RF ensemble score std).
  - uses `logic_exponent_beta`, `intensity_exponent_gamma`, and `intensity_log2_offset_delta` from `objective__params`.
  - params: `kind` (score), `y_axis`, `hue`
- **`sfxi_intensity_scaling`**: denom + clip fractions + E_raw distribution (current-round labels).
  - params: `y_col` (default `y_obs`), `percentile`, `min_n`, `eps`, `delta`, `include_pool`

SFXI plots that use current-round labels, model artifacts, or labels-as-of-round
should normally declare `round_selector: latest` when they live in a campaign
bundle that is run with `--round all`. This keeps scalar/vector round-history
plots all-round while label/model diagnostics resolve to a single, explicit
round in the manifest.

For notebooks that need per-round plot browsing, configure round-history plots
with `round_variants: [all, each]` and single-round SFXI diagnostics with
`round_variants: [latest, each]`. The plot runner will write one artifact per
declared scope; the notebook will only expose scopes present in
`plot_manifest.json`.

For round-history plots, `params.highlight_round: latest` marks current-round
points on the full history without changing data selection. Use this for the
"whole population, highlight current round" case. Use plot-level
`round_selector: latest` only when the primitive itself requires a single round.

### SFXI reference-overlay contract

SFXI overlay records that need to cross package boundaries should use the public
`dnadesign.opal.api.sfxi.to_sfxi_reference_overlay_records` helper. It emits
registry-compatible `sfxi_ref__...` columns with
`reference_instance_id`, `collection_id`, `batch_id`, `campaign_id`,
`metric_id`, `metric_value`, `metric_provenance`, `source_ref`, `score_ref`, and
the OPAL SFXI channels `logic_fidelity`, `effect_raw`, `effect_scaled`, and
`sfxi`. Plot code should consume that namespaced surface when it needs
reference overlays instead of recomputing or inventing plot-local field names.

Consumers that already have materialized overlay rows should validate them with
`dnadesign.opal.api.sfxi.validate_sfxi_reference_overlay_records(...)` before
plotting. OPAL's built-in `fold_change_vs_logic_fidelity` reference overlay
path does this automatically and fails fast when a reference row was scored with
a different SFXI API version, objective, state order, or setpoint vector.

### Example YAML

```yaml
plots:
  - name: sfxi_factorial_map
    kind: sfxi_factorial_effects
    round_selector: latest
    params:
      size_by: obj__effect_scaled
      include_labels: true

  - name: sfxi_setpoint_sweep
    kind: sfxi_setpoint_sweep
    round_selector: latest
    params: {}

  - name: sfxi_support_diag
    kind: sfxi_support_diagnostics
    round_selector: latest
    params:
      y_axis: score
      hue: effect_scaled

  - name: sfxi_uncertainty
    kind: sfxi_uncertainty
    round_selector: latest
    params:
      kind: score

  - name: sfxi_intensity_scaling
    kind: sfxi_intensity_scaling
    round_selector: latest
    params:
      include_pool: true
```


### Writing a new plot

1. Create a module in `dnadesign/opal/src/plots/` and register it:

```python
from ..registries.plots import PlotMeta, register_plot

@register_plot(
    "my_cool_plot",
    meta=PlotMeta(
        summary="Short operator-facing purpose.",
        requires=("records", "ledger_predictions_dir"),
        data_shape="scalar_over_rounds",
        tidy_schema=("round", "cohort", "metric", "summary", "value"),
        failure_modes=("missing metric column", "ambiguous run_id"),
        params={"metric": "numeric field to summarize"},
    ),
)
def render(context, params):
    # context: campaign_dir, workspace, rounds, data_paths, output_dir, filename, dpi, format, logger, save_data
    # - Read from context.data_paths (e.g., "records", your custom sources)
    # - Ledger sinks live under context.workspace.outputs_dir
    # - Build tidy DataFrame(s)
    # - Plot with matplotlib/seaborn (your call)
    # - Save to context.output_dir / context.filename
    # - Optionally write tidy CSV via context.save_df(df) if context.save_data
    ...
```

2. In your campaign YAML, add:

```yaml
plots:
  - name: my_cool_plot_instance
    kind: my_cool_plot
    params: { ... }
```

Plot kinds should be data-shape primitives rather than campaign-specific one-offs.
Prefer categories such as scalar over rounds, vector over rounds, matrix heatmap,
categorical composition, selected overlap, attribution matrix,
uncertainty/support distribution, objective decomposition, candidate audit table,
or calibration. Campaign semantics such as SFXI setpoints or study metadata
should configure those primitives through `params:` and input data, not fork the
plot ontology unless a genuinely new data shape is required.

Study-owned plot plugins must use the same API. Register the plot with
`@register_plot(...)`, declare `PlotMeta`, read only from `PlotContext`
`data_paths` / workspace surfaces, write media to `context.output_dir /
context.filename`, and let OPAL write the artifact manifest. A study report may
keep separate aggregate EDA figures, but those figures are not OPAL notebook
evidence unless they are produced through this plot API and appear in
`plot_manifest.json`. OPAL intentionally fails fast on unknown plot kinds,
unknown top-level plot keys, missing media, and malformed tidy CSVs rather than
silently scraping arbitrary files.

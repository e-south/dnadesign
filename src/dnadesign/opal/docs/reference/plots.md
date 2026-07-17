## OPAL Plots

**Owner:** dnadesign-maintainers
**Last verified:** 2026-07-17


Plot plugins own their rendering, but their public contract is shape-first metadata: required sources, required columns, tidy output schema, failure modes, and artifact manifests.

### How it works

- Prefer a separate plots config file (e.g., `plots.yaml`) and reference it from `campaign.yaml` via `plot_config`.
- Run plots with:

```bash
uv run opal plot --config /path/to/campaign.yaml \
  [--view selection-view-id] \
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
      score_field: "view__selection_score" # selected view projected from run_pred
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
      hue: view__selection_score
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
- Plot runs project one named selection view from `pred__selection_views`.
  Multi-view campaigns require `--view`; projected fields use the `view__`
  prefix, including `view__selection_score`, `view__rank_competition`, and
  `view__is_selected`.
- Vector heatmaps are generic vector plots: campaigns should pass display-ready
  `channel_labels`, `value_label`, `reference_mse_metric_label`, and
  `reference_mse_expression` rather than relying on raw channel slugs.
- Use `enabled: false` to keep a plot entry without running it.
- Presets merge into each plot entry; entry values override preset values.
- `campaign.yaml` references one `plot_config`; plot definitions do not live inline with runtime configuration.
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
| `selection_view_id`, `run_id`, `rounds` | explicit view/run/round scope used for input resolution |
| `params` | merged plot parameters after defaults and presets |
| `inputs` | resolved built-in and custom data paths with file size and mtime |
| `outputs.media` | rendered image/SVG/PDF files |
| `outputs.tidy_csv` | tidy CSV saved by the plugin, including its generation-time SHA-256 digest |
| `metadata` | `PlotMeta` summary, capability, data shape, tidy schema, and failure modes |
| `quality` | tidy CSV schema validation status when a plot declares `metadata.tidy_schema` |
| `freshness` | mtime-based freshness summary for resolved inputs and outputs |
| `caption`, `review_purpose` | manifest-backed human purpose text; plot params can override generic metadata text |
| `premise`, `decision_value`, `rationale` | the single claim tested, why the view changes a decision, and why this encoding is used |
| `alt_text`, `non_claim_boundary`, `tier` | accessible visual description, explicit claim limit, and review hierarchy |
| `warnings`, `error` | structured nonfatal and fatal plot outcomes |

Review and generated notebook surfaces should read manifests first. Extra files
on disk are advisory only; they can trigger stale-file warnings, but they are
not current evidence unless referenced by the active manifest.

Interactive notebook adapters verify their declared tidy output before parsing
it. The table must be the manifest's single `tidy_csv` output, remain inside the
campaign `outputs/plots` root after path resolution, and match the recorded
SHA-256. Regenerate older or changed plot artifacts instead of rebinding them at
notebook load time.

When `round_variants` expands one configured plot into multiple artifacts, the
aggregate index contains multiple `opal.plot_artifact.v1` manifests with the
same `name` and different `rounds` / filename suffixes. Notebook visual
surfaces group those rows under one plot-level dropdown and expose a second
manifest-backed plot-scope dropdown only when multiple scopes exist.

Multi-view plot indexes are written under
`outputs/plots/selection_views/<view>/`. A plot manifest without
`selection_view_id` is invalid for a multi-view review surface.

Plot capability metadata is the dropdown contract. It records objective family,
data layer, round behavior, label requirement, model-artifact requirement, and
tidy-data availability so notebooks can distinguish configured-but-missing,
generated-current, generated-stale, and stale-unmanifested surfaces.

---

### Response-Magnitude Feasibility decision plots

These plots are operative QA surfaces for campaigns using
`response_magnitude_feasibility_v1`. They read one unambiguous prediction-ledger
run and recompute the persisted feasibility score with the canonical public
math API. A mismatch fails plot generation. They do not read Reader trajectories
or redefine assay semantics.

- **`response_magnitude_feasibility_frontier`**: plots raw response separation against
  minimum target-ON reference-relative magnitude. Color is the signed
  target-OFF decision margin `q_off = (tau_off - b_off) / s_off`, dashed lines are the configured response and ON
  boundaries, and outlined diamonds are selected candidates. This is the
  primary candidate-universe view.
- **`response_magnitude_feasibility_constraint_decomposition`**: shows selected
  candidates by `q_response`, `q_on`, `q_off`, and `min(q)`. Zero is the pass
  boundary for each requirement, and the feasibility column must equal the
  row-wise minimum of the first three columns. This is the primary
  handoff-review view.

Both plots require the active selection view's `selection.params.score_ref` to
be `feasibility_margin`. Their default labels remain
assay-neutral; study plot configs should provide exact response and
reference-relative fluorescence labels when the Reader handoff declares those
quantities. Predicted feasibility is decision support, not measured promoter
behavior. Candidate-table `usr_label__primary` and `usr_label__aliases` fields
are optional presentation metadata: the frontier falls back to a compact
candidate ID when they are absent. Observed-event `display_label` values remain
source-projected metadata and never participate in identity, training, or
ranking.

---

### Multistate Response Behavior decision plots

These plots are operative review surfaces for campaigns using
`multistate_response_behavior_v1`. They resolve one exact selection view and
run, replay `pred__y_hat_model` through the public behavior mathematics, and
fail if the replayed `behavior_score` differs from the persisted score. Measured
points are replayed from the run-pinned `labels/observed_events.parquet`
artifact. They do not read the mutable labels ledger or infer assay semantics.

- **`multistate_response_behavior_frontier`**: plots the response-family score
  against the target-ON signal-family score. Color encodes the target-OFF
  signal-suppression family score. Diamonds identify candidates actually
  allocated to the active view; they are not reconstructed raw top-k rows.
  The manifest-backed layered-scatter adapter provides independent prediction,
  allocation, observed-batch, and label controls.
- **`multistate_response_behavior_selected_decomposition`**: shows every
  K-state response, ON-signal, and OFF-signal-suppression coordinate for the
  allocated candidates, followed by the three family scores, hard bottleneck,
  and smooth behavior score. An outline marks the lowest state-level
  coordinate.

Neither plot draws or implies a feasibility boundary. Zero is a
reference-direction value, not a pass threshold, and a positive score does not
mean that every coordinate is positive. The generic objective knows only a
reference-relative signal; a study-owned plot configuration may name that
signal more specifically when its assay contract supports the claim.

The built-in `scatter_score_vs_rank`, `observed_objective_over_rounds`,
`metric_over_rounds`, `vector_summary_heatmap`, feature-importance surfaces,
selection-batch review, and BaseRender panel remain reusable. They should not
be copied under behavior-specific names.

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
  - `view__selection_score` is the selected score from the projected selection view.
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
- **`observed_objective_over_rounds`**: measured candidate objective values by
  observed batch under one commensurate objective contract.
  - shows every candidate, the batch median, the between-candidate IQR, and an
    optional cumulative-best trajectory and zero boundary;
  - requires `round_selector: all`, no global `run_id`, and an explicit
    `params.run_series` map with exactly one run and contract digest per round;
  - the contract digest binds the selection view, objective parameters,
    calibration, target mask, score reference and direction, Y ingest, Y-space,
    label-source kind, and verified observed-event artifact;
  - cumulative run snapshots must retain every prior candidate-round event
    unchanged. Duplicate, changed, or dropped events fail closed;
  - campaign-history labels without batch IDs are grouped under a derived
    `round-<observed_round>` batch label. A non-empty Y-space remains required;
  - objective plugins must explicitly declare pointwise observed replay. RMF
    supports it because its scores depend only on the eight observed values and
    fixed parameters. SFXI does not because its score includes training-state
    normalization;
  - the IQR is between-candidate spread, not assay uncertainty or a confidence
    interval. Observed round and batch identify measurement timing, not
    selection provenance;
  - batch candidate counts are printed on the axis and preserved in tidy data.
    The cumulative best is monotone by construction. Neither it nor a batch
    shift establishes X-to-Y learning, predictor improvement, selection-policy
    improvement, or a causal round effect. Those claims require a prospectively
    frozen prediction-and-baseline analysis that binds the selection policy.

  Compute each pinned digest through the public helper, then place it in the
  plot configuration:

  ```python
  from dnadesign.opal.api import observed_objective_run_contract_sha256

  digest = observed_objective_run_contract_sha256(
      outputs_dir=campaign_dir / "outputs",
      selection_view_id="ethanol",
      as_of_round=1,
      run_id="r1-...",
  )
  ```

  ```yaml
  - name: observed_objective_history
    kind: observed_objective_over_rounds
    round_selector: all
    params:
      zero_boundary: true
      show_cumulative_best: true
      run_series:
        schema_version: opal.observed_objective_run_series.v1
        runs:
          - as_of_round: 0
            run_id: r0-...
            contract_sha256: <sha256>
          - as_of_round: 1
            run_id: r1-...
            contract_sha256: <sha256>
  ```

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
    # - Read from context.data_paths (e.g., "records" or declared custom sources)
    # - Ledger sinks live under context.workspace.outputs_dir
    # - Build tidy DataFrame(s)
    # - Plot with matplotlib/seaborn
    # - Save to context.output_dir / context.filename
    # - Optionally write tidy CSV via context.save_df(df) if context.save_data
    ...
```

2. Add the plot instance to the campaign YAML:

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

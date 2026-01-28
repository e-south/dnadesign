# OPAL Plots

This document covers **plot plugins** and the `PlotContext` helper. Plots are bespoke by design: each plugin owns its data loading, joins, and styling.

### How it works

- Prefer a separate plots config file (e.g., `plots.yaml`) and reference it from `campaign.yaml` via `plot_config`.
- Run plots with:

```bash
opal plot --config /path/to/campaign.yaml \
  [--plot-config /path/to/plots.yaml] \
  [--round latest|all|3|1,3,7|2-5] \
  [--name my_plot] \
  [--tag my_tag]
```

- Discover available plot kinds:

```bash
opal plot --list
opal plot --list-config --config /path/to/campaign.yaml
opal plot --describe scatter_score_vs_rank
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

    # Optional extra sources (built-ins auto-injected: records, outputs)
    data:
      - name: extra_csv
        path: ./extras/scores.csv

    # Opaque, plugin-specific params (required for plot knobs)
    params:
      score_field: "score_sfxi"       # required by the starter plugin
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
      hue: pred__y_obj_scalar
      cbar: true
```

**Notes:**
- Plotting knobs must live under `params:`. Top‑level plotting keys are rejected.
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

## SFXI diagnostics plots

These plots reuse shared SFXI math and are safe to run without retraining.
Diagnostic plots always render the full dataset; sampling parameters are not supported.

### Plot kinds + params

- **`sfxi_factorial_effects`**: factorial-effects map from predicted logic vectors.
  - params: `size_by` (default `obj__effect_scaled`), `include_labels`, `rasterize_at`
- **`sfxi_setpoint_sweep`**: objective landscape across discrete setpoints (current-round labels).
  - rendered as a heatmap with setpoints as columns (vector labels) and diagnostic metrics as rows.
  - default heatmap omits `denom_used` to preserve contrast.
  - params: `y_col` (default `y_obs`), `top_k`, `tau`, `percentile`, `min_n`, `eps`, `delta`
- **`sfxi_support_diagnostics`**: distance-to-labeled-logic vs score (OOD check).
  - params: `y_axis`, `hue`, `batch_size`
- **`sfxi_uncertainty`**: uncertainty vs score (artifact model; RF ensemble score std).
  - params: `y_axis`, `hue`
- **`sfxi_intensity_scaling`**: denom + clip fractions + E_raw distribution (current-round labels).
  - params: `y_col` (default `y_obs`), `percentile`, `min_n`, `eps`, `delta`, `include_pool`

### Example YAML

```yaml
plots:
  - name: sfxi_factorial_map
    kind: sfxi_factorial_effects
    params:
      size_by: obj__effect_scaled

  - name: sfxi_setpoint_sweep
    kind: sfxi_setpoint_sweep
    params:
      top_k: 5
      tau: 0.8

  - name: sfxi_support_diag
    kind: sfxi_support_diagnostics
    params:
      y_axis: score
      hue: effect_scaled

  - name: sfxi_uncertainty
    kind: sfxi_uncertainty

  - name: sfxi_intensity_scaling
    kind: sfxi_intensity_scaling
    params:
      include_pool: true
```


### Writing a new plot

1. Create a module in `dnadesign/opal/src/plots/` and register it:

```python
from ..registries.plots import register_plot

@register_plot("my_cool_plot")
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

## OPAL Plots

This folder contains **plot plugins** and a small `PlotContext` helper. Plots are
bespoke by design: each plugin owns its data loading, joins, and styling.

### How it works

- Prefer a separate plots config file (e.g., `plots.yaml`) and reference it from `campaign.yaml` via `plot_config`.
- Run plots with:

```bash
opal plot --config /path/to/campaign.yaml \
  [--plot-config /path/to/plots.yaml] \
  [--round latest|all|3|1,3,7|2-5] \
  [--name my_plot] \
  [--tag quick]
```

### Minimal YAML schema (recommended)

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
    tags: [quick]

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

**Built-ins injected** (resolved from the campaign config):

* `records` → resolved from `data.location` in `campaign.yaml`
* `outputs` → campaign `outputs/` directory
* `ledger_predictions_dir` → `outputs/ledger.predictions/`
* `ledger_runs_parquet` → `outputs/ledger.runs.parquet`
* `ledger_labels_parquet` → `outputs/ledger.labels.parquet`

Ledger sinks always live under `context.workspace.outputs_dir` (e.g., `outputs/ledger.*`).


### Writing a new plot

1. Create a module in `dnadesign/opal/src/plots/` and register it:

```python
from ..registries.plot import register_plot

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

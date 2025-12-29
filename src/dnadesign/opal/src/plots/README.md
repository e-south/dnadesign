## OPAL Plots

This folder contains **plot plugins** and a small `PlotContext` helper. Plots are
bespoke by design: each plugin owns its data loading, joins, and styling.

### How it works

- Add plots to your campaign YAML under a top-level `plots:` list.
- Run them with:

```bash
opal plot --config /path/to/campaign.yaml [--round latest|all|3|1,3,7|2-5] [--name my_plot]
```

### Minimal YAML schema

```yaml
plots:
  - name: score_vs_rank_latest        # unique instance label
    kind: scatter_score_vs_rank       # plugin id registered in plots registry

    # Optional extra sources (built-ins auto-injected: records, artifacts)
    data:
      - name: extra_csv
        path: ./extras/scores.csv

    # Opaque, plugin-specific params
    params:
      score_field: "score_sfxi"       # required by the starter plugin
      hue: null                       # or "round"
      highlight_selected: false

    # Optional output tuning
    output:
      dir: "{campaign}/plots/{kind}/{name}"
      filename: "{name}{round_suffix}.png"
      dpi: 600
      format: "png"                   # png|svg|pdf (png default)
      save_data: false                # save tidy CSV next to the image
```

**Built-ins injected** (resolved from the campaign config):

* `records` → resolved from `data.location` in `campaign.yaml`
* `artifacts` → `<campaign>/artifacts/`

Ledger sinks always live under `context.workspace.outputs_dir` (e.g., `outputs/ledger.*`).


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

## cruncher progressive demo (validation + live sampling)

This walkthrough highlights progressive UX features: campaign validation, live sampling status, and live metric trends during optimization. It assumes the demo workspace and a working RegulonDB connection. For the two-TF basics, see [demo.md](demo.md). For the full campaign walkthrough, see [demo_campaigns.md](demo_campaigns.md).

Captured outputs below were generated on **2026-01-10** using `CRUNCHER_LOG_LEVEL=WARNING` and `COLUMNS=200` to avoid truncated tables (unless noted otherwise).

### 1) Enter the demo workspace

```bash
cd src/dnadesign/cruncher/workspaces/demo
```

### 2) Validate the campaign definitions (no cache required)

This checks category and campaign rules without hitting the catalog:

```bash
cruncher campaign validate --campaign demo_categories_best --no-selectors --no-metrics
```

### 3) Fetch sites and validate with selectors

Fetch curated RegulonDB sites for the campaign and validate against selectors:

```bash
cruncher fetch sites --campaign demo_categories --no-selectors
cruncher campaign validate --campaign demo_categories_best --show-filtered
```

If validation fails, fix the reported issues (missing cache, ambiguous datasets, or site window lengths) and rerun.

### 4) Generate a derived config

```bash
cruncher campaign generate --campaign demo_categories_best --out config.demo_categories_best.yaml
```

### 5) Start sampling and watch live metrics

Terminal A (start a sample run):

```bash
cruncher lock config.demo_categories_best.yaml
cruncher parse config.demo_categories_best.yaml
cruncher sample config.demo_categories_best.yaml
```

Terminal B (live progress + metric trends):

```bash
cruncher runs watch config.demo_categories_best.yaml <sample_run_name>
```

Use `cruncher runs list config.demo_categories_best.yaml` to find the sample run name.

`runs watch` reads `run_status.json` and, if present, `live_metrics.jsonl` for best/current score trends. You can also write a live PNG plot during sampling:

```bash
cruncher runs watch config.demo_categories_best.yaml <sample_run_name> --plot
```

Increase the window size if you want longer history:

```bash
cruncher runs watch config.demo_categories_best.yaml <sample_run_name> --metric-points 80 --metric-width 40
```

### 6) Analyze and summarize

```bash
cruncher analyze --latest --plots pairgrid config.demo_categories_best.yaml
cruncher campaign summarize --campaign demo_categories_best --skip-missing
```

Outputs land under `runs/campaigns/<campaign_id>/` and include summary tables plus campaign plots.

## Sampling + analysis (auto‑optimize)

This guide shows the minimal end‑to‑end CLI flow for sampling and analysis, plus
how auto‑opt picks a stable optimizer by default.

### End‑to‑end (one regulator set)

From a workspace directory (so `config.yaml` is the default):

```bash
# Reproducibility pinning (required before parse/sample)
cruncher lock

# Optimization (auto‑opt is on by default)
cruncher sample
# Optional: skip pilots
cruncher sample --no-auto-opt

# Diagnostics + plots
cruncher analyze --latest

# Summarized report (JSON + Markdown)
cruncher report --latest
```

If you run via pixi, prefix each command with `pixi run cruncher --`.

### Auto‑optimize (default)

Auto‑opt runs short **Gibbs** and **parallel tempering (PT)** pilots, compares quality diagnostics, logs the decision, then runs the final sample with the best‑scoring candidate. There is **no silent fallback**: if neither pilot meets quality thresholds, auto‑opt retries with cooler settings and then errors if the runs remain unstable.

Disable auto‑opt when you want a single fixed optimizer:

```bash
cruncher sample --no-auto-opt
```

Config knobs live under `sample.auto_opt` in `config.yaml` (see the config
reference for full options).

### Diagnostics quick read

Diagnostics are written to:

- `analysis/tables/diagnostics.json`
- `analysis/tables/score_summary.csv`
- `analysis/tables/joint_metrics.csv`

Key signals:

- `diagnostics.status` — `ok|warn|fail` summary.
- `trace.rhat` ≲ 1.1 and `trace.ess_ratio` ≳ 0.10 usually indicate good mixing.
- `sequences.unique_fraction` low values suggest chain collapse or lack of
  diversity.
- PT runs: `optimizer.swap_acceptance_rate` near 0.05–0.40 is typical; very low
  values often mean a poor temperature ladder.

### Run selection + paths

Use these helpers to choose the right run and reduce table truncation:

```bash
COLUMNS=160 cruncher runs list
cruncher runs latest --set-index 1
cruncher runs best --set-index 1
```

Run artifacts live under:

```
<workspace>/runs/sample/<set_name>/<run_id>/
```

### Tuning essentials (when diagnostics warn)

If auto‑opt selects a run but diagnostics are weak, try:

- Increase `sample.draws` / `sample.tune` (more iterations).
- Increase `sample.chains` (better mixing; higher cost).
- Adjust `sample.optimiser.cooling.beta` (cooler ladders mix more stably).
- Raise `sample.pwm_sum_threshold` to keep only higher‑scoring draws.
- Tighten `sample.moves` ranges if swaps/moves are too disruptive.

For full settings, see the config reference.

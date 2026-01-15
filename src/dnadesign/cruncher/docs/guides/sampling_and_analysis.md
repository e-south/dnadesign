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
# Optional: skip pilots (requires optimizer.name=gibbs|pt in config)
cruncher sample --no-auto-opt

# Diagnostics + plots
cruncher analyze --latest

# Summarized report (JSON + Markdown)
cruncher report --latest
```

If you run via pixi, prefix each command with `pixi run cruncher --`.

### Numba cache (required for fast diagnostics)

Cruncher uses ArviZ + Numba for trace diagnostics. If `NUMBA_CACHE_DIR` is not set, cruncher
sets it to `<repo>/src/dnadesign/cruncher/.cruncher/numba_cache` (repo root discovered via
`pyproject.toml` or `.git`; falls back to `<repo>/.cruncher/numba_cache` if the cruncher dir
is missing) and requires it to be writable. To override:

```bash
export NUMBA_CACHE_DIR=/path/to/writable/cache
```

Logging defaults to concise progress. For verbose traces, use:

```bash
cruncher sample --verbose
cruncher --log-level DEBUG analyze --latest
```

### Auto‑optimize (default)

Auto‑opt runs short **Gibbs** and **parallel tempering (PT)** pilots, compares **optimization quality** signals (balance, diversity, acceptance bands, best‑score progress), logs the decision, then runs the final sample with the best‑scoring candidate. Pilot trace diagnostics (R‑hat/ESS) are still reported as warnings, but the auto‑opt scorecard is the source of truth for selection. There is **no silent fallback**: if neither pilot meets quality thresholds, auto‑opt retries with cooler settings and then proceeds with the best available candidate (logging warnings). It only errors when no pilot produced usable diagnostics.
Auto‑opt pilot runs are stored under `runs/auto_opt/`.

Disable auto‑opt when you want a single fixed optimizer (set `sample.optimizer.name`
to `gibbs` or `pt` in your config):

```bash
cruncher sample --no-auto-opt
```

Config knobs live under `sample.auto_opt` in `config.yaml` (see the config
reference for full options). Auto‑opt also writes a pilot scorecard to
`analysis/tables/auto_opt_pilots.csv` and a tradeoff plot to
`analysis/plots/auto_opt_tradeoffs.png` when available.

If enabled, `sample.auto_opt.length` probes multiple sequence lengths and
prefers the shortest candidate that clears the scorecard; when comparing
lengths, auto‑opt ranks by normalized balance (then best score) to reduce
length bias. Use
`sample.rng.deterministic=true` to lock pilot RNG streams per config.

Cooling vs soft‑min: `optimizers.gibbs.beta_schedule` controls MCMC temperature,
while `objective.softmin` controls the min‑approximation hardness. They are
independent schedules; use soft‑min to smooth early exploration and a cooler
beta to stabilize acceptance.

### Diagnostics quick read

Diagnostics are written to:

- `analysis/tables/diagnostics.json`
- `analysis/tables/score_summary.csv`
- `analysis/tables/joint_metrics.csv`

Key signals:

- `diagnostics.status` — `ok|warn|fail` summary.
- `trace.rhat` ≲ 1.1 and `trace.ess_ratio` ≳ 0.10 usually indicate good mixing
  for Gibbs runs. PT ladders skip R‑hat/ESS across temperatures.
- `sequences.unique_fraction` low values suggest chain collapse or lack of
  diversity.
- `elites.balance_median` and `elites.diversity_hamming` summarize how well the
  top‑K elites balance TF scores and remain diverse.
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
<workspace>/runs/sample/<run_name>/
```

`run_name` includes the TF slug; when multiple regulator sets are configured it is prefixed with `setN_`.

If `cruncher analyze` reports `analysis/ contains artifacts but meta/summary.json is missing`,
remove the incomplete `analysis/` folder for that run before re-running analyze.

### Tuning essentials (when diagnostics warn)

If auto‑opt selects a run but diagnostics are weak, try:

- Increase `sample.budget.draws` / `sample.budget.tune` (more iterations).
- For Gibbs, increase `sample.budget.restarts`; for PT, adjust
  `sample.optimizers.pt.beta_ladder`.
- Adjust `sample.optimizers.gibbs.beta_schedule` (cooler schedules mix more stably).
- Raise `sample.elites.filters.pwm_sum_min` to keep only higher‑scoring draws.
- Tighten `sample.moves.overrides` ranges if swaps/moves are too disruptive.

For full settings, see the config reference.

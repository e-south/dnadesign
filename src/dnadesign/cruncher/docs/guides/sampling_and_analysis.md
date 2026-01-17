## Sampling + analysis (auto‑optimize)

This guide shows the minimal end‑to‑end CLI flow for sampling and analysis, plus
how auto‑opt picks a stable optimizer by default. For a one‑page happy path,
see the top‑level cruncher README.

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
cruncher analyze

# Summarized report (JSON + Markdown)
cruncher report --latest
```

If you run via pixi, prefix each command with `pixi run cruncher --`.

Outputs are recorded under each run’s `analysis/` folder. The canonical summary
is `analysis/summary.json`, which links to plot/table manifests. A detailed
inventory with “why” each artifact was generated is in `analysis/manifest.json`.

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

Auto‑opt runs short **Gibbs** and **parallel tempering (PT)** pilots, evaluates objective‑aligned metrics from the draw phase (`combined_score_final`), and selects the best candidate using the top‑K median score (with a secondary max score tie‑breaker). Pilot diagnostics (R‑hat/ESS, acceptance summaries, diversity) are still reported as warnings, but selection is **thresholdless** and driven by objective‑comparable metrics. Auto‑opt escalates through `budget_levels` (and the configured `replicates`) until a confidence‑separated winner emerges.

`auto_opt.policy.allow_warn: true` means auto‑opt will always pick a winner by the end of the configured budgets and record low‑confidence warnings if the separation is unclear. Set `allow_warn: false` to require a confidence‑separated winner; if none emerges at the maximum budgets/replicates, auto‑opt fails fast with guidance to increase `auto_opt.budget_levels` and/or `auto_opt.replicates`. Pilots disable trim/polish by default to preserve length fidelity; override with `auto_opt.allow_trim_polish_in_pilots: true` if needed.
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
compares them using the same objective‑aligned top‑K median score; use
`sample.auto_opt.length.prefer_shortest: true` to force the shortest winning length. Use
`sample.rng.deterministic=true` to lock pilot RNG streams per config.

Length bias warning: max-over-offset scoring makes longer sequences look better
even under random background. If you sweep lengths, prefer normalized scales
(`score_scale: normalized-llr|z|logp`) and/or set a length penalty:
`objective.length_penalty_lambda > 0`.

Cooling vs soft‑min: `optimizers.gibbs.beta_schedule` controls MCMC temperature,
while `objective.softmin` controls the min‑approximation hardness. They are
independent schedules; use soft‑min to smooth early exploration and a cooler
beta to stabilize acceptance.

### Length ladder (warm start)

Ladder mode runs sequential lengths (L0..Lmax) with warm starts from the prior
length's raw samples (`sequences.parquet`, elites as fallback). This avoids a
biased "longer always wins" contest and makes length sweeps cheaper.

```yaml
sample:
  auto_opt:
    length:
      enabled: true
      mode: ladder
      step: 1
      min_length: null   # defaults to max PWM length
      max_length: null   # defaults to sum of PWM lengths
      warm_start: true
      ladder_budget_scale: 0.5
  objective:
    length_penalty_lambda: 0.25
```

Ladder runs write `analysis/tables/length_ladder.csv` under the auto-opt pilot
root with per‑length summary metrics.

### Elites: filter → rank → diversify

Elite selection is explicitly aligned with the optimization objective:

1) **Filter (representativeness)** — gates candidates using normalized per‑TF
   scores: `sample.elites.filters.min_per_tf_norm` (recommended) and optional
   `pwm_sum_min` as a secondary threshold.
2) **Rank (objective)** — sorts by the same combined score used by MCMC
   (`combined_score_final`, soft‑min across TFs plus length penalty), then
   `min_norm`, then `sum_norm`.
3) **Diversify** — Hamming‑distance filter (`elites.min_hamming`), optionally
   dsDNA‑aware via `elites.dsDNA_hamming=true`.

`pwm_sum_min` is not the objective; it is a representativeness gate.
`combined_score_final` respects `objective.combine` (default `min`, or `sum`
for `score_scale=consensus-neglop-sum` unless you set `objective.combine=min`).
When using `normalized-llr`, per‑TF scores are scaled by the PWM consensus LLR
(`0.0` = background‑like, `1.0` = consensus‑like). A `min_per_tf_norm` of 0.05–0.2
roughly corresponds to demanding 5–20% of consensus per TF; start with `null`
and tune based on `normalized_min_median` in diagnostics.

### Motif overlap metrics (feature, not a failure)

Overlap of TF motifs is expected and informative. Cruncher records overlap
patterns from elites (best-hit windows per TF) and reports:

- `analysis/tables/overlap_summary.csv` — TF-pair overlap rates, overlap bp stats,
  strand-combo counts, and `overlap_bp_hist` (bins+counts).
- `analysis/tables/elite_overlap.csv` — per-elite overlap totals and pair counts.
- `analysis/plots/plot__overlap_heatmap.png` — heatmap of overlap rates.
- `analysis/plots/plot__overlap_bp_distribution.png` — distribution of overlap bp.
- Optional: `plot__overlap_strand_combos.png` + `plot__motif_offset_rug.png`.

These are descriptive only; overlap is not penalized by default.

### Diagnostics quick read

Diagnostics are written to:

- `analysis/tables/diagnostics.json`
- `analysis/tables/score_summary.csv`
- `analysis/tables/joint_metrics.csv`
- `analysis/tables/objective_components.json`
- `analysis/plots/plot__dashboard.png`
- `analysis/plots/plot__worst_tf_trace.png` / `plot__worst_tf_identity.png`

Key signals:

- `diagnostics.status` — `ok|warn|fail` summary.
- `trace.rhat` ≲ 1.1 and `trace.ess_ratio` ≳ 0.10 usually indicate good mixing
  for Gibbs runs. PT ladders report ESS from the cold chain only.
- PT traces record **post‑swap** states (per temperature chain) for clarity.
- `trace.nc` contains draw phase only; `sample.output.trace.include_tune`
  controls whether tune samples appear in `sequences.parquet`.
- When `sample.elites.dsDNA_canonicalize=true`, `sequences.parquet` includes a
  `canonical_sequence` column and uniqueness metrics use that canonical form.
- `sequences.parquet` includes `chain_1based` and `draw_in_phase` to make
  plotting easier: `chain` remains 0‑based, `draw_idx` is the absolute sweep,
  and `draw_in_phase` is 0‑based within the phase (tune/draw).
- `sequences.unique_fraction` low values suggest chain collapse or lack of
  diversity (use `elites.dsDNA_canonicalize=true` to treat reverse complements
  as identical for uniqueness).
- `objective_components.json` includes `unique_fraction_canonical` only when
  `elites.dsDNA_canonicalize=true` is enabled.
- `elites.balance_median` and `elites.diversity_hamming` summarize how well the
  top‑K elites balance TF scores and remain diverse.
- PT runs: `optimizer.swap_acceptance_rate` near 0.05–0.40 is typical; very low
  values often mean a poor temperature ladder.
- Acceptance metrics: Gibbs `S` moves are always accepted. Auto‑opt uses the
  MH‑only aggregate rate (`optimizer.acceptance_rate_mh`) for scorecards; per‑move
  `acceptance_rate.B/M` are still reported for debugging. Diagnostics also report
  `optimizer.acceptance_rate_mh_tail` (MH acceptance over the tail window).

Plot defaults are intentionally lightweight (Tier‑0 only, including the dashboard).
Enable additional plots by setting `analysis.extra_plots=true` or use
`cruncher analyze --plots all`. Trace‑based diagnostics require
`analysis.mcmc_diagnostics=true`.

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

If `cruncher analyze` reports `analysis/ contains artifacts but summary.json is missing`,
remove the incomplete `analysis/` folder for that run before re-running analyze.

### Tuning essentials (when diagnostics warn)

If auto‑opt selects a run but diagnostics are weak, try:

- Increase `sample.budget.draws` / `sample.budget.tune` (more iterations).
- For Gibbs, increase `sample.budget.restarts`; for PT, adjust
  `sample.optimizers.pt.beta_ladder`.
- Adjust `sample.optimizers.gibbs.beta_schedule` (cooler schedules mix more stably).
- Raise `sample.elites.filters.min_per_tf_norm` to enforce per‑TF satisfaction,
  or use `pwm_sum_min` for a looser gate.
- Tighten `sample.moves.overrides` ranges if swaps/moves are too disruptive.

For full settings, see the config reference.

Quick decision guide:
- **Low unique_fraction** → increase draws/restarts or raise `elites.min_hamming`.
- **Very low swap acceptance (PT)** → adjust `pt.beta_ladder` (gentler spacing).
- **Weak worst‑TF trace** → increase `objective.softmin.beta_end` or normalize scores.

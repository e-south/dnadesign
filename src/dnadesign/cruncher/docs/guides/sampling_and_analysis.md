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
# Optional: skip pilots (requires optimizer.name=pt and auto_opt.enabled=false)
cruncher sample --no-auto-opt

# Diagnostics + plots (writes analysis/report.json + analysis/report.md)
cruncher analyze

# Optional: print a concise summary to stdout
cruncher analyze --summary
```

If you run via pixi, prefix each command with `pixi run cruncher --`.

Outputs are recorded under each run’s `analysis/` folder. The canonical summary
is `analysis/summary.json`, which links to plot/table manifests. The human‑readable
entrypoint is `analysis/report.md` (machine‑readable `analysis/report.json`). A detailed
inventory with “why” each artifact was generated is in `analysis/manifest.json`.
`analysis/objective_components.json` includes a `learning` block with best-score
draws, last improvement draw, and (when `early_stop` is configured) a per-chain
plateau simulation to help spot stalls (gated by `require_min_unique` when enabled).

### Numba cache (required for fast diagnostics)

Cruncher uses ArviZ + Numba for trace diagnostics. Cruncher sets `NUMBA_CACHE_DIR` to
`<workspace>/.cruncher/numba_cache` (relative to the config workspace) and requires it
to be writable.

Logging defaults to concise progress. For verbose traces, use:

```bash
cruncher sample --verbose
cruncher --log-level DEBUG analyze --latest
```

### Auto‑optimize (default)

Auto‑opt runs short **parallel tempering (PT)** pilots, scores each pilot with the MMR scorecard, and selects the best candidate using a deterministic tie‑break. The scorecard computes:

`pilot_score = median(relevance_raw among selected elites) + diversity_weight * mean_pairwise_distance`

Relevance defaults to `min_per_tf_norm` (consensus‑like per TF). Tie‑breakers are `median_relevance_raw`, `mean_pairwise_distance`, then a deterministic candidate ID. Pilot diagnostics (R‑hat/ESS, acceptance summaries, diversity) are still recorded, but very short pilot budgets (<200 draws) suppress mixing warnings to avoid noise. Selection is **thresholdless** and driven by objective‑comparable metrics. Auto‑opt escalates through `budget_levels` (and the configured `replicates`) until a confidence‑separated winner emerges.

`auto_opt.policy.allow_warn: true` means auto‑opt will pick a winner by the end of the configured budgets among candidates that pass diagnostics and record low‑confidence warnings if the separation is unclear. Set `allow_warn: false` to require a confidence‑separated winner; if none emerges at the maximum budgets/replicates (or only warning‑level candidates remain), auto‑opt fails fast with guidance to increase `auto_opt.budget_levels` and/or `auto_opt.replicates`. Auto‑opt pilot runs are stored under `outputs/auto_opt/`.

Disable auto‑opt when you want a single fixed PT optimizer (set `sample.optimizer.name`
to `pt` in your config):

```bash
cruncher sample --no-auto-opt
```
If `optimizer.name=auto`, `--no-auto-opt` is not allowed; set `sample.optimizer.name`
explicitly to `pt` and disable `auto_opt.enabled` in your config.

Config knobs live under `sample.auto_opt` in `config.yaml` (see the config
reference for full options). Auto‑opt also writes a pilot scorecard (including
`pilot_score`, `median_relevance_raw`, `mean_pairwise_distance`) to
`analysis/auto_opt_pilots.parquet` and a tradeoff plot to
`analysis/plot__auto_opt_tradeoffs.<plot_format>` when available.
The pilot scorecard includes `ess_ratio` and `trace_draws` so you can see
whether pilots stopped early and how mixing scales with the draw count. If most
candidates are `warn`, increase pilot budgets, relax early‑stop, or set
`auto_opt.policy.allow_warn: true` so selection can proceed with explicit warnings.

Length bias warning: max-over-offset scoring makes longer sequences look better
even under random background. If you manually compare lengths, prefer normalized scales
(`score_scale: normalized-llr|z|logp`) and/or set a length penalty:
`objective.length_penalty_lambda > 0`.

Cooling vs soft‑min: `optimizers.pt.beta_ladder` controls MCMC temperature,
while `objective.softmin` controls the min‑approximation hardness. They are
independent schedules; use soft‑min to smooth early exploration and a cooler
beta ladder to stabilize acceptance.

### Elites: filter → select (MMR by default)

Elite selection is explicitly aligned with the optimization objective:

1) **Filter (representativeness)** — gates candidates using normalized per‑TF
   scores: `sample.elites.filters.min_per_tf_norm` (recommended) and optional
   `pwm_sum_min` as a secondary threshold.
2) **Select (MMR only)** — build a filtered pool, score relevance (defaults to
   `min_per_tf_norm`), then greedily select **K** with MMR using
   `selection.alpha` (relevance vs diversity). Current MMR behavior (TFBS‑core mode) is:
   - For each sequence in the candidate pool, extract the best‑hit window for each TF
     and orient each core to its PWM.
   - When comparing two sequences, compute LexA‑core vs LexA‑core and CpxR‑core vs
     CpxR‑core Hamming distances (weighted per PWM position), then average across TFs.
   - We never compare LexA vs CpxR within the same sequence.
   “Tolerant” weights emphasize low‑information PWM positions to preserve consensus‑critical
   bases while encouraging diversity.
   When `objective.bidirectional=true`, MMR deduplicates by canonical sequence so reverse
   complements (including palindromes) count as the same identity.

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

- `analysis/overlap_summary.parquet` — TF-pair overlap rates, overlap bp stats,
  strand-combo counts, and `overlap_bp_hist` (bins+counts).
- `analysis/elite_overlap.parquet` — per-elite overlap totals and pair counts.
- `analysis/plot__overlap_heatmap.<plot_format>` — heatmap of overlap rates.
- `analysis/plot__overlap_bp_distribution.<plot_format>` — distribution of overlap bp.
- Optional: `plot__overlap_strand_combos.png` + `plot__motif_offset_rug.png`.

These are descriptive only; overlap is not penalized by default.

### Diagnostics quick read

Diagnostics are written to:

- `analysis/diagnostics.json`
- `analysis/score_summary.parquet`
- `analysis/joint_metrics.parquet`
- `analysis/objective_components.json`
- `analysis/plot__dashboard.<plot_format>`
- `analysis/plot__worst_tf_trace.<plot_format>` / `plot__worst_tf_identity.<plot_format>`

Key signals:

- `diagnostics.status` — `ok|warn|fail` summary.
- `trace.rhat` ≲ 1.1 and `trace.ess_ratio` ≳ 0.10 usually indicate good mixing;
  PT ladders report ESS from the cold chain only.
- PT traces record **post‑swap** states (per temperature chain) for clarity.
- `trace.nc` contains draw phase only; `sample.output.trace.include_tune`
  controls whether tune samples appear in `sequences.parquet`.
- When `objective.bidirectional=true`, `sequences.parquet` includes a
  `canonical_sequence` column and uniqueness metrics use that canonical form.
- `sequences.parquet` includes `chain_1based` and `draw_in_phase` to make
  plotting easier: `chain` remains 0‑based, `draw` is the absolute sweep,
  and `draw_in_phase` is 0‑based within the phase (tune/draw).
- `sequences.parquet` includes `min_per_tf_norm` (alias of `min_norm`) for the
  per‑TF minimum in normalized scales.
- `sequences.unique_fraction` low values suggest chain collapse or lack of
  diversity (increase draws or tighten selection filters).
- `objective_components.json` includes `unique_fraction_canonical` only when
  dsDNA canonicalization is enabled.
- MMR runs also write `analysis/elites_mmr_summary.parquet` with pool/alpha
  metadata and diversity summaries.
- `elites.balance_median` and `elites.diversity_hamming` summarize how well the
  top‑K elites balance TF scores and remain diverse.
- PT runs: `optimizer.swap_acceptance_rate` near 0.05–0.40 is typical; very low
  values often mean a poor temperature ladder.
- Acceptance metrics: auto‑opt uses the MH‑only aggregate rate
  (`optimizer.acceptance_rate_mh`) for scorecards; per‑move acceptance rates are
  still reported for debugging. Diagnostics also report
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
<workspace>/outputs/sample/<run_name>/
```

`run_name` includes the TF slug; when multiple regulator sets are configured it is prefixed with `setN_`.

If `cruncher analyze` reports `analysis/ contains artifacts but summary.json is missing`,
remove the incomplete `analysis/` folder for that run before re-running analyze.

### Tuning essentials (when diagnostics warn)

If auto‑opt selects a run but diagnostics are weak, try:

- Increase `sample.budget.draws` / `sample.budget.tune` (more iterations).
- For PT, adjust `sample.optimizers.pt.beta_ladder` or `swap_prob` if swaps are rare.
- Raise `sample.elites.filters.min_per_tf_norm` to enforce per‑TF satisfaction,
  or use `pwm_sum_min` for a looser gate.
- Tighten `sample.moves.overrides` ranges if swaps/moves are too disruptive.

For full settings, see the config reference.

Quick decision guide:
- **Low unique_fraction** → increase draws or relax `elites.filters.min_per_tf_norm`.
- **Very low swap acceptance (PT)** → adjust `pt.beta_ladder` (gentler spacing).
- **Weak worst‑TF trace** → increase `objective.softmin.beta_end` or normalize scores.

## Sampling + analysis


## Contents
- [Sampling + analysis](#sampling--analysis)
- [End-to-end (one regulator set)](#end-to-end-one-regulator-set)
- [Numba cache (required for fast diagnostics)](#numba-cache-required-for-fast-diagnostics)
- [Fixed-length compute model](#fixed-length-compute-model)
- [Elites: filter → select (MMR)](#elites-filter--select-mmr)
- [Motif overlap metrics (feature, not a failure)](#motif-overlap-metrics-feature-not-a-failure)
- [Diagnostics quick read](#diagnostics-quick-read)
- [Run selection + paths](#run-selection--paths)
- [Tuning essentials (when diagnostics warn)](#tuning-essentials-when-diagnostics-warn)
- [Related references](#related-references)

This guide explains how to run sampling, interpret diagnostics, and read the analysis artifacts.

### End-to-end (one regulator set)

From a workspace directory (so `config.yaml` is the default):

```bash
# Reproducibility pinning (required before parse/sample)
cruncher lock

# Optimization (parallel tempering, fixed-length)
cruncher sample

# Diagnostics + plots (writes analysis/report.json + analysis/report.md)
cruncher analyze

# Optional: print a concise summary to stdout
cruncher analyze --summary
```

If you run via pixi, prefix each command with `pixi run cruncher --`.

Outputs are recorded under each run’s `analysis/` folder. The canonical summary
is `analysis/summary.json`, which links to plot/table manifests. The human-readable
entrypoint is `analysis/report.md` (machine-readable `analysis/report.json`). A detailed
inventory with “why” each artifact was generated is in `analysis/manifest.json`.
`analysis/objective_components.json` includes a `learning` block with best-score
sweeps, last improvement sweep, and (when `early_stop` is configured) a per-chain
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

### Fixed-length compute model

Sampling uses a fixed sequence length and a single, explicit sweep budget:

- `sample.sequence_length` sets the designed DNA length.
- `sample.compute.total_sweeps` is the total number of sweeps executed per chain.
- `sample.compute.adapt_sweep_frac` controls what fraction of sweeps are used for
  adaptation (the “tune” phase).

Cruncher derives phase lengths as:

```
adapt_sweeps = ceil(total_sweeps * adapt_sweep_frac)
draw_sweeps = total_sweeps - adapt_sweeps
```

Adaptation runs first, then draw sweeps populate `trace.nc` and `sequences.parquet`.
If `objective.score_scale` is `normalized-llr`, `early_stop.min_delta` must be ≤ 0.1.

Length invariants are strict: `sequence_length` must be at least the widest PWM; if
not, sampling fails fast with the per-TF widths so you can adjust the config.

### Elites: filter → select (MMR)

Elite selection is explicitly aligned with the optimization objective:

1) **Filter (representativeness)** — gates candidates using normalized per‑TF
   scores: `sample.elites.min_per_tf_norm` (recommended) and
   `sample.elites.require_all_tfs_over_min_norm`.
2) **Select (MMR only)** — builds a filtered pool, scores relevance
   (defaults to `min_per_tf_norm`), then greedily selects **K** with MMR using
   `sample.elites.mmr_alpha` (relevance vs diversity).

Current MMR behavior (TFBS‑core mode) is:

- For each sequence in the candidate pool, extract the best‑hit window for each TF
  and orient each core to its PWM.
- When comparing two sequences, compute LexA‑core vs LexA‑core and CpxR‑core vs
  CpxR‑core Hamming distances (weighted per PWM position), then average across TFs.
- We never compare LexA vs CpxR within the same sequence.

“Tolerant” weights emphasize low‑information PWM positions to preserve consensus‑critical
bases while encouraging diversity. When `objective.bidirectional=true`, MMR deduplicates
by canonical sequence so reverse complements (including palindromes) count as the same
identity.

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
- When `analysis.dashboard_only=false`, overlap plots are also written:
  `plot__overlap_heatmap.<plot_format>` and `plot__overlap_bp_distribution.<plot_format>`.
- Optional: `plot__overlap_strand_combos.png` + `plot__motif_offset_rug.png` (when enabled).

These are descriptive only; overlap is not penalized by default.

### Diagnostics quick read

Diagnostics are written to:

- `analysis/diagnostics.json`
- `analysis/score_summary.parquet`
- `analysis/joint_metrics.parquet`
- `analysis/objective_components.json`
- `analysis/plot__dashboard.<plot_format>`
- When `analysis.dashboard_only=false`, also write
  `plot__worst_tf_trace.<plot_format>` / `plot__worst_tf_identity.<plot_format>`.

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
  and `draw_in_phase` is 0‑based within the phase (adapt/draw).
- `sequences.parquet` includes `min_per_tf_norm` (alias of `min_norm`) for the
  per‑TF minimum in normalized scales.
- `sequences.unique_fraction` low values suggest chain collapse or lack of
  diversity (increase sweeps or tighten selection filters).
- `objective_components.json` includes `unique_fraction_canonical` only when
  dsDNA canonicalization is enabled.
- MMR runs also write `analysis/elites_mmr_summary.parquet` with pool/alpha
  metadata and diversity summaries.
- `elites.balance_median` and `elites.diversity_hamming` summarize how well the
  top‑K elites balance TF scores and remain diverse.
- PT runs: `optimizer.swap_acceptance_rate` near 0.05–0.40 is typical; very low
  values often mean a poor temperature ladder.
- Acceptance metrics: `optimizer.acceptance_rate_mh` is the MH-only aggregate;
  per‑move acceptance rates are still reported for debugging.

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

If diagnostics are weak, try:

- Increase `sample.compute.total_sweeps` (more iterations).
- Reduce `sample.compute.adapt_sweep_frac` to allocate more sweeps to draws.
- Adjust `sample.elites.min_per_tf_norm` and `require_all_tfs_over_min_norm` to
  balance strictness vs yield.
- Increase `sample.sequence_length` if PWM windows are forced to overlap too tightly.
- Switch `sample.moves.profile` (for example, `balanced` → `aggressive`) if the
  chain is stuck locally.

### Related references

- [Config reference](../reference/config.md)
- [CLI reference](../reference/cli.md)
- [Two‑TF demo](../demos/demo_basics_two_tf.md)

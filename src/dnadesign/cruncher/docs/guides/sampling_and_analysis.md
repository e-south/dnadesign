## Sampling + analysis

**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-28


**Last updated by:** cruncher-maintainers on 2026-02-28

### Contents
- [Overview](#overview)
- [End-to-end](#end-to-end)
- [Sequence export layer](#sequence-export-layer)
- [Fixed-length sampling model](#fixed-length-sampling-model)
- [Fail-fast contracts](#fail-fast-contracts)
- [Gibbs annealing optimization model](#gibbs-annealing-optimization-model)
- [What The Optimizer Is Optimizing](#what-the-optimizer-is-optimizing)
- [Interpreting Noisy Trajectories](#interpreting-noisy-trajectories)
- [Elites: filter -> MMR select](#elites-mmr-select--postprocess)
- [Analysis outputs](#analysis-outputs)
- [Diagnostics quick read](#diagnostics-quick-read)
- [Run selection + paths](#run-selection--paths)
- [Related references](#related-references)

### Overview

Cruncher designs fixed-length DNA sequences that jointly satisfy multiple PWMs. Sampling uses a Gibbs annealing optimizer and returns a diverse elite set via TFBS-core MMR. Analysis is offline and reads only run artifacts.

**Important:** Cruncher is an **optimization engine**, not posterior inference. The "tune/draws/trace" terminology is used for optimizer mechanics + diagnostics; don't interpret outputs as posterior samples.

### End-to-end

From a workspace directory (so `configs/config.yaml` is the default):

```bash
cruncher lock
cruncher sample
cruncher analyze
cruncher analyze --summary
cruncher export sequences --latest
```

Outputs are written under each run directory (typically `outputs/`). The standard entrypoints are:

- `analysis/reports/summary.json`
- `analysis/reports/report.md`
- `analysis/reports/report.json`

### Sequence export layer

Use `cruncher export sequences` to emit stable, contract-first sequence tables for downstream wrappers.

```bash
cruncher export sequences --latest
cruncher export sequences --run <run_name_or_path>
cruncher export sequences --latest --table-format csv
```

Default table format is CSV. Use `--table-format parquet` only when parquet is explicitly required downstream.

Outputs per run:

- `export/table__elites.csv` (elite ids, sequence lengths, overall sequence, and rich per-regulator window metadata in `window_members_json`)
- `export/table__consensus_sites.<parquet|csv>` (one consensus sequence per TF motif)
- `export/export_manifest.json`

Fail-fast invariants:

- no duplicate `(elite_id, tf)` best-hit rows in source hits
- per-elite `window_members_json` is non-empty and includes regulator, offset, strand, k-mer, and score fields
- window spans stay within elite sequence bounds
- score columns are numeric and finite
- per-TF motif provenance (`pwm_ref`, `pwm_hash`, `pwm_width`) is internally consistent

### Fixed-length sampling model

Sampling is fixed-length and strict:

- `sample.sequence_length` sets the designed DNA length.
- `sample.budget.tune` and `sample.budget.draws` are explicit; no hidden budgets.
- `sample.motif_width.maxw` optionally trims wide PWMs by max-information windowing during sampling.
- `sample.sequence_length` must be at least the widest PWM after `sample.motif_width` constraints are applied.

If the length invariant is violated, sampling fails fast with the per-TF widths.

### Fail-fast contracts

Cruncher is intentionally strict and will error instead of silently degrading:

- Unknown config keys or missing required keys -> error (strict schema).
- Missing lockfile for `parse`/`sample` -> error.
- Existing `.cruncher/parse` or run `outputs/` directories -> error unless `--force-overwrite` is passed to `parse`/`sample`.
- `sample.sequence_length < max_pwm_width` -> error with widths.
- Elite selection cannot produce `sample.elites.k` candidates before postprocess -> error (no silent threshold relaxation).
- Invalid `sample.optimizer.*` cooling settings -> error (no implicit schedule fallbacks).
- `analyze` requires required artifacts and will not write partial reports on missing/invalid inputs.
- `analyze` enforces an in-progress lock at `<run_dir>/analysis/state/tmp`; active locks fail fast, stale interrupted locks are auto-pruned.
- `analysis.fimo_compare.enabled=true` requires MEME Suite `fimo` to be resolvable (`discover.tool_path`, `MEME_BIN`, or `PATH`).

### Gibbs annealing optimization model

Cruncher uses a hybrid Gibbs/MH optimizer with an explicit beta schedule
(`sample.optimizer.cooling.*`). Replica exchange is disabled, so chains run
independently while sharing the same objective and move policy.

What "hybrid" means in practice:

1. At each sweep, each chain samples one move kind from `sample.moves.*`.
2. Default move mix is not `P(S)=1`: `S=0.85, B=0.07, M=0.04, I=0.04, L=0, W=0`.
3. If move kind is `S`, Cruncher performs a single-site Gibbs update.
4. If move kind is `B/M/L/W/I`, Cruncher performs a Metropolis-Hastings proposal with accept/reject.
5. Optional adaptation can update move probabilities and proposal scales during tune (`sample.moves.overrides.*`).
6. Chain identity is preserved across sweeps, so trajectory outputs are chain-lineage plots (not slot/particle swap traces).

Terminology alignment:

- `gibbs_anneal` = Metropolis-within-Gibbs sequence optimization.
- With changing `beta_t` over sweeps, this is simulated annealing.
- With fixed `beta`, it is fixed-temperature hybrid MCMC (not annealing).

### What The Optimizer Is Optimizing

Each chain maintains sequence state:

```
x in {A, C, G, T}^L
```

with objective `f(x)` built from PWM best-window scores across TFs.

Scoring mechanics:

- Scan windows per TF (optionally both strands via `objective.bidirectional=true`).
- Select the best window per TF with deterministic tie-breaks.
- Combine per-TF values using `objective.combine` (`min` or `sum`), optionally
  shaped with soft-min:

```
softmin(v; beta) = -(1 / beta) * log(sum_i exp(-beta * v_i))
```

If using log-probability scoring (`score_scale=logp`), best-window p-values are
adjusted to sequence-level p-values:

```
p_seq = 1 - (1 - p_win)^n_tests
```

Move updates:

- `S` (single-site Gibbs): sample from a conditional distribution over bases:

```
P(base=b | rest) proportional to exp(beta_mcmc * f_b)
```

  `S` is accepted by construction because Gibbs samples directly from a
  conditional kernel. "Always accepted" does not mean "no beta effect":
  `beta_mcmc` controls how sharp this distribution is.
  - low `beta_mcmc`: flatter distribution (more exploratory)
  - high `beta_mcmc`: concentrated on better bases (more greedy)
- `B/M/L/W/I`: Metropolis acceptance:

```
alpha = min(1, exp(beta_mcmc * (f(x') - f(x))))
```

This is why analysis reports MH acceptance separately from `S`: including
always-accepted Gibbs updates would hide whether MH proposals are calibrated.
Trajectory stability is controlled by both cooling (`beta_mcmc`) and move
proposal scale/mix.

### Interpreting Noisy Trajectories

Raw chain traces can look jumpy even when optimization is healthy:

- Gibbs single-site updates are always accepted.
- The objective is rugged due to max-over-windows and cross-TF aggregation.
- Finite tail temperature still allows some downhill MH transitions.
- Adaptive proposal mechanics can change local behavior during a run.

For optimization narrative, prioritize:

- final/best elite quality and constraints
- tail move diagnostics (`acceptance_tail_rugged`, `downhill_accept_tail_rugged`, `gibbs_flip_rate_tail`)
- diversity tables (`elites_mmr_summary`, `elites_nn_distance`)
- sweep plots with `analysis.trajectory_sweep_mode=best_so_far` (or `all` to overlay raw + best-so-far)
- `analysis.trajectory_stride` as a readability control (deterministic decimation that preserves first/last sweeps and improvement events)
- `analysis.trajectory_summary_overlay=true` only when you explicitly want a non-parametric cross-chain narrative overlay (median + IQR across chains), using raw values in `raw` mode and best-so-far values in `best_so_far|all` modes

Treat raw trajectory plots as exploration diagnostics, not monotone progress
curves.

### Elites: MMR select + postprocess

Elite selection is aligned with the optimization objective:

1. Build the candidate pool from sampled draw states.
2. Run MMR selection on that pool using `sample.elites.select.diversity` (`0..1`) as the primary quality-vs-diversity control.
3. Postprocess selected elites with strict hit-window invariants:
   - iterative single-owner nucleotide polish,
   - contiguous uncovered prefix/suffix trim,
   - final dedupe over standard/full sequence identity.
4. Return the postprocessed elite set.

Current MMR behavior is:

- For each sequence in the candidate pool, extract the best-hit window for each TF and orient each core to its PWM.
- Compare candidates with a hybrid distance: full-sequence Hamming + motif-core weighted Hamming.
- Use direct tradeoff weights from `sample.elites.select.diversity`: score weight `1 - diversity`, diversity weight `diversity`, plus minimum-distance constraints derived from the same knob.
- We never compare LexA vs CpxR within the same sequence.

`sample.elites.select.diversity=0.0` is an explicit score-only mode:
- Selection policy switches to greedy top-k by final optimizer scalar score (`combined_score_final`).
- No diversity constraints are derived or applied.
- This gives a clear baseline for diversity-vs-score sweeps.

"Tolerant" weights are hard-coded: low-information PWM positions are weighted more (weight = `1 - info_norm`). This preserves consensus-critical positions while encouraging diversity where the motif is flexible.

When `objective.bidirectional=true`, orientation normalization is automatic: reverse complements (including palindromes) count as the same identity for uniqueness and MMR dedupe.

If Cruncher cannot produce `sample.elites.k` candidates from the available candidate pool before postprocess, the run fails fast. Postprocess dedupe may reduce the persisted elite count below `k` when two candidates converge to the same final sequence.

### Analysis outputs

Analysis writes a curated, orthogonal suite of plots and tables (no plot booleans). Key artifacts include:

- `analysis/tables/table__scores_summary.parquet`
- `analysis/tables/table__elites_topk.parquet`
- `analysis/tables/table__metrics_joint.parquet`
- `analysis/tables/table__chain_trajectory_points.parquet`
- `analysis/tables/table__chain_trajectory_lines.parquet`
- `analysis/tables/table__overlap_pair_summary.parquet`
- `analysis/tables/table__overlap_per_elite.parquet`
- `analysis/tables/table__diagnostics_summary.json`
- `analysis/tables/table__objective_components.json`
- `analysis/tables/table__elites_mmr_summary.parquet`
- `analysis/tables/table__elites_mmr_sweep.parquet` (when `analysis.mmr_sweep.enabled=true`)
- `analysis/tables/table__elites_nn_distance.parquet`

Sampling artifacts consumed by analysis:

- `optimize/tables/elites_hits.parquet` (per-elite, per-TF best-hit/core metadata)
- `optimize/tables/random_baseline.parquet` (baseline cloud for trajectory plots; default on via `sample.output.save_random_baseline=true` and `sample.output.random_baseline_n=10000`; includes seed, n_samples, length, score_scale, bidirectional, background model)
- `optimize/tables/random_baseline_hits.parquet` (baseline best-hit/core metadata for diversity context; same baseline contract as above)

Plots (always generated when data is available):

- `plots/elite_score_space_context.*` (random-baseline cloud + selected elites at true score coordinates + TF consensus anchors)
  - `analysis.pairwise=auto` uses explicit TF-pair axes for 2-TF runs; for 3+ TF runs it selects TF-specific worst/second-worst regulators from elite rankings and plots raw-LLR axes for those TFs.
  - consensus anchors are per-TF consensus sequences (`<tf> consensus`) projected into the active axis pair; they are references, not theoretical upper-bound envelopes.
  - `analysis.pairwise=all_pairs_grid` renders every TF pair in one score-space figure, with edge-only axis labels.
  - all-pairs grid shares panel limits automatically when `analysis.trajectory_scatter_scale=normalized-llr`; raw-LLR grids keep panel-local limits.
- `plots/chain_trajectory_sweep.*` (optimizer scalar score over sweeps by chain, with `best_so_far|raw|all` modes, step rendering for best-so-far, and tune/cooling boundary markers when available; scalar semantics follow `sample.objective.combine` and optional soft-min shaping)
- `plots/chain_trajectory_video.mp4` (optional via `analysis.trajectory_video.enabled=true`: selected-chain motif snapshots rendered through BaseRender, with `best_so_far` default timeline, tune+draw eligibility by default, and strict duration/fps/frame guardrails)
- `plots/elites_nn_distance.*` (elite diversity panel: y-axis is final optimizer scalar score, x-axis is full-sequence nearest-neighbor Hamming distance in bp to each elite's closest other selected elite, plus pairwise full-sequence distance matrix; core-distance context retained)
- `plots/elites_showcase.*` (baserender-backed elite panels with sense/antisense sequence rows, TF best-window placement, and per-window motif logos)
  - motif-logo matrices are sourced from the locked optimization motif library (resolved by Cruncher during analysis) rather than inferred from elite strings.
  - per-panel TF score tokens are taken from `norm_<tf>` columns and must be normalized in `[0,1]` relative to each TF consensus-theoretical max anchor.
  - Cruncher passes only minimal rendering primitives (record-shaped rows + motif primitives); baserender does not require full run artifact trees.
  - rendering uses baserender public API only (`dnadesign.baserender` package root), not internal `dnadesign.baserender.src.*` modules.
  - panel title format is terse and rank-first: `Elite #<rank>` (fallback `Elite` when rank is unavailable, hard cap 40 chars).
  - when multiple elite panels are rendered in one figure, Cruncher requests a single-row layout (left-to-right rank order).
- `plots/health_panel.*` (MH-only acceptance dynamics + move-mix over sweeps; `S` moves are excluded from acceptance rates)
- `plots/optimizer_vs_fimo.*` (optional via `analysis.fimo_compare.enabled=true`: descriptive QA scatter comparing Cruncher joint optimizer score vs FIMO weakest-TF sequence score; no-hit rows are retained at 0)

Fail-fast plotting contract:

- Required plots (`elite_score_space_context`, `chain_trajectory_sweep`, `elites_nn_distance`, `elites_showcase`) fail analysis on plotting errors; they are not silently downgraded to skipped.
- Intentional skips are only policy/data-driven cases (for example `health_panel` when trace is disabled, or `optimizer_vs_fimo` when `analysis.fimo_compare.enabled=false`).

### Diagnostics quick read

Use `report.md` for a narrative view and `summary.json` for machine-readable links.

Key signals:

- `diagnostics.status`: `ok|warn|fail` summary.
- `trace.rhat` and `trace.ess_ratio` (if `trace.nc` exists) are directional indicators of sampling health across chains.
- `optimizer.acceptance_tail_rugged` is the tail acceptance over rugged moves (`B`,`M`) and is more informative than all non-`S` acceptance when diagnosing warm tails.
- `optimizer.downhill_accept_tail_rugged` isolates downhill rugged acceptance in the tail (cold-tail indicator).
- `optimizer.gibbs_flip_rate_tail` and `optimizer.tail_step_hamming_mean` indicate whether late-chain motion is dominated by Gibbs micro-flips.
- `chain_trajectory_sweep.*` should be read as optimizer-scalar progress; `best_so_far` is the default narrative, while `all` overlays raw exploration.
- `elites_nn_distance.*` distinguishes motif-core collapse (`d_core ~ 0`) from full-sequence diversity (`d_full > 0`) so degenerate elite sets are visible.
- `objective_components.unique_fraction_standard` is present only when orientation normalization is enabled.
- `elites_mmr_summary` and `elites_nn_distance` indicate diversity strength and collapse risk.

### Run selection + paths

```bash
COLUMNS=160 cruncher runs list
cruncher runs latest --set-index 1
cruncher runs best --set-index 1
```

Run artifacts live under:

```
<workspace>/outputs/
```

### Related references

- [Config reference](../reference/config.md)
- [CLI reference](../reference/cli.md)
- [Architecture and artifacts](../reference/architecture.md)
- [Intent + lifecycle](intent_and_lifecycle.md)

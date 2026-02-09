# Sampling + analysis

## Contents
- [Overview](#overview)
- [End-to-end](#end-to-end)
- [Fixed-length sampling model](#fixed-length-sampling-model)
- [Fail-fast contracts](#fail-fast-contracts)
- [PT optimization model](#pt-optimization-model)
- [Elites: filter -> MMR select](#elites-filter--mmr-select)
- [Analysis outputs](#analysis-outputs)
- [Diagnostics quick read](#diagnostics-quick-read)
- [Run selection + paths](#run-selection--paths)
- [Related references](#related-references)

## Overview

Cruncher designs fixed-length DNA sequences that jointly satisfy multiple PWMs. Sampling uses PT-only MCMC and returns a diverse elite set via TFBS-core MMR. Analysis is offline and reads only run artifacts.

**Important:** Cruncher is an **optimization engine**, not posterior inference. The "tune/draws/trace" terminology is used for PT mechanics + diagnostics; don't interpret outputs as posterior samples.

## End-to-end

From a workspace directory (so `config.yaml` is the default):

```bash
cruncher lock
cruncher sample
cruncher analyze
cruncher analyze --summary
```

Outputs are written under each run directory (typically `outputs/`). The canonical entrypoints are:

- `analysis/summary.json`
- `analysis/report.md`
- `analysis/report.json`

## Fixed-length sampling model

Sampling is fixed-length and strict:

- `sample.sequence_length` sets the designed DNA length.
- `sample.budget.tune` and `sample.budget.draws` are explicit; no hidden budgets.
- `sample.motif_width.maxw` optionally trims wide PWMs by max-information windowing during sampling.
- `sample.sequence_length` must be at least the widest PWM after `sample.motif_width` constraints are applied.

If the length invariant is violated, sampling fails fast with the per-TF widths.

## Fail-fast contracts

Cruncher is intentionally strict and will error instead of silently degrading:

- Unknown config keys or missing required keys -> error (strict schema).
- Missing lockfile for `parse`/`sample` -> error.
- Existing `.cruncher/parse` or run `outputs/` directories -> error unless `--force-overwrite` is passed to `parse`/`sample`.
- `sample.sequence_length < max_pwm_width` -> error with widths.
- Elite constraints cannot be satisfied (after filtering/selection) -> error (no silent threshold relaxation).
- With `sample.pt.adapt.strict=true`, ladder saturation during tune is treated as a tuning failure -> error.
- `analyze` requires required artifacts and will not write partial reports on missing/invalid inputs.

## PT optimization model

Cruncher uses replica-exchange MCMC (parallel tempering) as the only optimizer.
The method follows the standard composition from the PT literature:

1. Multiple replicas are simulated at different inverse temperatures (`sample.pt.n_temps`, `sample.pt.temp_max`).
2. Each replica performs local Metropolis-Hastings updates on sequence proposals (`sample.moves.*`).
3. Adjacent replicas periodically attempt state swaps (`sample.pt.swap_stride`) to move information between hot and cold chains.
4. Optional adaptation adjusts move probabilities/proposal scales and ladder scaling during tune (`sample.moves.overrides.*`, `sample.pt.adapt.*`).
5. Each initialized state carries a persistent `particle_id`; swaps exchange particle occupancy across slots while preserving particle lineage identity.

This keeps exploration (hot replicas) and exploitation (cold replica) coupled without changing the objective definition.
In strict mode (`sample.pt.adapt.strict=true`), ladder saturation is treated as a tuning failure and the run fails fast.

Terminology alignment with common methods:

- `parallel tempering` == `replica exchange MCMC`
- swap acceptance diagnostics correspond to adjacent-replica exchange rates
- tune windows adapt proposal mechanics; draw windows produce the retained sample set

For background, this is the same class of methods often cited as Metropolis-coupled MCMC / parallel tempering / replica exchange.

## Elites: filter -> MMR select

Elite selection is aligned with the optimization objective:

1. **Filter (representativeness)**: `sample.elites.filter.min_per_tf_norm` and `require_all_tfs` gate candidates.
2. **Select (MMR)**: a filtered pool is scored for relevance and then greedily selected for diversity.

Current MMR behavior (TFBS-core mode) is:

- For each sequence in the candidate pool, extract the best-hit window for each TF and orient each core to its PWM.
- When comparing two sequences, compute LexA-core vs LexA-core and CpxR-core vs CpxR-core Hamming distances (weighted per PWM position), then average across TFs.
- We never compare LexA vs CpxR within the same sequence.

"Tolerant" weights are hard-coded: low-information PWM positions are weighted more (weight = `1 - info_norm`). This preserves consensus-critical positions while encouraging diversity where the motif is flexible.

When `objective.bidirectional=true`, canonicalization is automatic: reverse complements (including palindromes) count as the same identity for uniqueness and MMR dedupe.

If Cruncher cannot produce `sample.elites.k` elites that satisfy the configured filter gates, the run fails fast (it does not silently lower thresholds).

## Analysis outputs

Analysis writes a curated, orthogonal suite of plots and tables (no plot booleans). Key artifacts include:

- `analysis/table__scores_summary.parquet`
- `analysis/table__elites_topk.parquet`
- `analysis/table__metrics_joint.parquet`
- `analysis/table__opt_trajectory_points.parquet`
- `analysis/table__opt_trajectory_particles.parquet`
- `analysis/table__overlap_pair_summary.parquet`
- `analysis/table__overlap_per_elite.parquet`
- `analysis/table__diagnostics_summary.json`
- `analysis/table__objective_components.json`
- `analysis/table__elites_mmr_summary.parquet`
- `analysis/table__elites_nn_distance.parquet`

Sampling artifacts consumed by analysis:

- `optimize/elites_hits.parquet` (per-elite, per-TF best-hit/core metadata)
- `optimize/random_baseline.parquet` (baseline cloud for trajectory plots; includes seed, n_samples, length, score_scale, bidirectional, background model)
- `optimize/random_baseline_hits.parquet` (baseline best-hit/core metadata for diversity context)

Plots (always generated when data is available):

- `plots/plot__opt_trajectory.*` (causal particle lineage scatter in TF score-space, with random-baseline cloud and consensus anchors)
- `plots/plot__opt_trajectory_sweep.*` (cold-slot progression over sweep index for the selected objective column, with dashed lineage handoffs and bottleneck-TF coloring)
- `plots/plot__elites_nn_distance.*`
- `plots/plot__overlap_panel.*`
- `plots/plot__health_panel.*` (only if `optimize/trace.nc` exists)

## Diagnostics quick read

Use `report.md` for a narrative view and `summary.json` for machine-readable links.

Key signals:

- `diagnostics.status`: `ok|warn|fail` summary.
- `trace.rhat` and `trace.ess_ratio` (if `trace.nc` exists) are directional indicators of sampling health for the cold chain.
- `objective_components.unique_fraction_canonical` is present only when canonicalization is enabled.
- `elites_mmr_summary` and `elites_nn_distance` indicate diversity strength and collapse risk.

## Run selection + paths

```bash
COLUMNS=160 cruncher runs list
cruncher runs latest --set-index 1
cruncher runs best --set-index 1
```

Run artifacts live under:

```
<workspace>/outputs/
```

## Related references

- [Config reference](../reference/config.md)
- [CLI reference](../reference/cli.md)
- [Architecture and artifacts](../reference/architecture.md)
- [Intent + lifecycle](intent_and_lifecycle.md)

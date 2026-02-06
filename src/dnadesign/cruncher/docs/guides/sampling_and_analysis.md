# Sampling + analysis

## Contents
- [Overview](#overview)
- [End-to-end](#end-to-end)
- [Fixed-length sampling model](#fixed-length-sampling-model)
- [Elites: filter → MMR select](#elites-filter--mmr-select)
- [Analysis outputs](#analysis-outputs)
- [Diagnostics quick read](#diagnostics-quick-read)
- [Run selection + paths](#run-selection--paths)
- [Related references](#related-references)

## Overview

Cruncher designs fixed-length DNA sequences that jointly satisfy multiple PWMs. Sampling uses PT-only MCMC and returns a diverse elite set via TFBS-core MMR. Analysis is offline and reads only run artifacts.

## End-to-end

From a workspace directory (so `config.yaml` is the default):

```bash
cruncher lock
cruncher sample
cruncher analyze
cruncher analyze --summary
```

Outputs are written under each run’s `analysis/` folder. The canonical entrypoints are:

- `analysis/summary.json`
- `analysis/report.md`
- `analysis/report.json`

## Fixed-length sampling model

Sampling is fixed-length and strict:

- `sample.sequence_length` sets the designed DNA length.
- `sample.budget.tune` and `sample.budget.draws` are explicit; no hidden budgets.
- `sample.sequence_length` must be at least the widest PWM (after any `catalog.pwm_window_lengths`).

If the length invariant is violated, sampling fails fast with the per‑TF widths.

## Elites: filter → MMR select

Elite selection is aligned with the optimization objective:

1. **Filter (representativeness)**: `sample.elites.filter.min_per_tf_norm` and `require_all_tfs` gate candidates.
2. **Select (MMR)**: a filtered pool is scored for relevance and then greedily selected for diversity.

Current MMR behavior (TFBS‑core mode) is:

- For each sequence in the candidate pool, extract the best‑hit window for each TF and orient each core to its PWM.
- When comparing two sequences, compute LexA‑core vs LexA‑core and CpxR‑core vs CpxR‑core Hamming distances (weighted per PWM position), then average across TFs.
- We never compare LexA vs CpxR within the same sequence.

“Tolerant” weights are hard‑coded: low‑information PWM positions are weighted more (weight = `1 - info_norm`). This preserves consensus‑critical positions while encouraging diversity where the motif is flexible.

When `objective.bidirectional=true`, canonicalization is automatic: reverse complements (including palindromes) count as the same identity for uniqueness and MMR dedupe.

## Analysis outputs

Analysis writes a curated, orthogonal suite of plots and tables (no plot booleans). Key artifacts include:

- `table__scores__summary.parquet`
- `table__elites__topk.parquet`
- `table__metrics__joint.parquet`
- `table__overlap__pair_summary.parquet`
- `table__overlap__per_elite.parquet`
- `table__diagnostics__summary.json`
- `table__objective__components.json`
- `table__elites__mmr_summary.parquet`
- `table__elites__nn_distance.parquet`

Plots (always generated when data is available):

- `plot__run__dashboard.*`
- `plot__scores__projection.*`
- `plot__elites__nn_distance.*`
- `plot__overlap__panel.*`
- `plot__diag__panel.*` (only if `trace.nc` exists)

## Diagnostics quick read

Use `analysis/report.md` for a narrative view and `analysis/summary.json` for machine‑readable links.

Key signals:

- `diagnostics.status`: `ok|warn|fail` summary.
- `trace.rhat` and `trace.ess_ratio` (if `trace.nc` exists) indicate sampling health for the cold chain.
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
<workspace>/outputs/sample/<run_name>/
```

## Related references

- [Config reference](../reference/config.md)
- [CLI reference](../reference/cli.md)
- [Architecture and artifacts](../reference/architecture.md)

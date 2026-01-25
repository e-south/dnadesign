# Stage-A PWM Sampling: Strata-First Semantics (FIMO)

## Context
Stage-A PWM sampling currently mixes the ideas of thresholding, binning, and selection in a way that exposes too many knobs and makes the logs hard to interpret. The desired behavior is: (1) mine PWM-like sequences, (2) account for a spectrum of p-value strata for diagnostics/visualization, and (3) retain only the best strata prefix for Stage-B. Configuration should be minimal and ergonomic, with a single obvious knob to adjust strictness, while still capturing per-bin distributions for later analyses (e.g., Hamming/Levenshtein by bin).

## Goals
- Align sampling semantics with user intent: generated → eligible → retained.
- Keep configuration minimal and hard to misconfigure.
- Preserve per-bin counts for didactic plots and diagnostics.
- Make shortfalls expected and interpretable without extra debug logs.
- Ensure docs, demo config, and tests align with the new semantics.

## Non-goals
- Automatic per-regulator threshold calibration.
- Changing the FIMO backend itself or its internal scoring.
- Adding new diversity-selection algorithms (post-hoc analysis stays separate).

## Proposed Semantics
We define three counts per regulator:
- **Generated**: number of candidate sequences sampled.
- **Eligible**: candidates with a FIMO hit at or below a floor threshold.
- **Retained**: eligible hits within the best strata prefix, deduped and capped.

FIMO only reports hits under its reporting threshold, so eligibility is defined by that floor. Per-bin counts are computed for eligible hits to support plots and later analysis. Retention is a strict prefix of bins (best p-values), not an arbitrary list of indices.

## Config Changes (Breaking)
Replace `pvalue_threshold` and `mining.retain_bin_ids` with two semantic knobs:
- `pvalue_strata`: ordered p-value edges (best → worst). The **last** edge is the eligibility floor (FIMO `--thresh`).
- `retain_depth`: number of best bins to keep for Stage-B (prefix of strata).

`n_sites` remains as the **cap** on retained unique sites per regulator (not a target). The default behavior should be explicit in docs; a typical default is `pvalue_strata: [1e-8, 1e-6, 1e-4]` with `retain_depth: 2`.

## Data Flow
1. Generate candidate sequences as today.
2. Run FIMO with `--thresh = last(pvalue_strata)`.
3. Bin each reported hit by `pvalue_strata`.
4. Accumulate **eligible** counts per bin (all bins up to the floor).
5. Retain only bins `0..retain_depth-1`.
6. Dedup retained sequences; if retained > `n_sites`, keep best by `(pvalue asc, score desc)`.

This keeps accounting broad while retention remains strict and bounded.

## Reporting & UX
Stage-A recap table should show:
- `candidates` = generated/target
- `eligible` = eligible/generated
- `pool` = retained/n_sites
- `bins` = per-bin `eligible/retained` pairs (e.g., `b0 12/12 | b1 55/20 | b2 400/0`)
- `len` = `n/min/med/avg/max` for retained (pool) sequences

Zero-retained cases become interpretable without extra logs:
- `eligible=0` → no hits under floor.
- `eligible>0` but retained bins empty → hits exist, none in strict strata.

## Migration
- Remove `pvalue_threshold` and `mining.retain_bin_ids` from config schema.
- Require `pvalue_strata` and `retain_depth` for FIMO inputs.
- Update metadata fields to reflect `pvalue_strata` and `retain_depth`.
- Update demo config and docs to use the new semantics.

## Testing Plan
- Config validation rejects legacy keys and enforces `pvalue_strata` + `retain_depth`.
- Sampling tests verify:
  - FIMO floor applied from last stratum edge.
  - Eligibility counts include all bins up to floor.
  - Retention is a prefix of bins (best strata).
  - Dedup + cap are enforced on retained sites.
- CLI recap tests verify new column labels and bin formatting.

## Open Questions
- Default `retain_depth` (require explicit vs. default to full strata).
- Whether to surface the eligibility floor explicitly in metadata or derive from `pvalue_strata`.

# scar_nick Package Map

`scar_nick` models a Type IIS retained scar followed by an exact terminal nick.
It is the Cruncher surface for retron hairpin base-junction candidates, not a
phenotype predictor.

## Data Flow

1. `load.py` validates the YAML spec boundary.
2. `geometry.py` audits release-enzyme and nickase compatibility.
3. `profiles.py` calls the four scar positions in canonical `S3_S2_S1_S0`
   order.
4. `policy.py` classifies each profile as `active`, `reserve`, or `reject`.
5. `candidates.py` builds per-pair candidate records and rejection reasons.
6. `planner.py` enumerates feasible left/right scar pairs and builds the report
   model.
7. `ranking.py` performs deterministic bucket coverage and tie-breaking.
8. `tables.py` writes the CSV handoff tables.
9. `artifacts.py` owns artifact paths, manifests, snapshots, and visual bundle
   persistence.
10. `view_contracts.py` publishes the pre/post terminal-nick visual contract.
11. `reporting.py` renders the Markdown run summary.

## Core Semantics

- `S0` must be a Watson-Crick match.
- `S3`, `S2`, and `S1` are design variables.
- `W` means `G:T` or `T:G`; it is not a hard mismatch.
- `MXXM` is a reserve middle-middle hard-mismatch profile, not an active panel
  default.
- `XXMM` and `XMXM` are active S3-edge double-hard profiles when they satisfy
  the rest of the ligation policy.

## Handoff Artifacts

- `export/table__scar_nick_candidates.csv` gives one row per accepted
  candidate, including enzyme identity, recognition sites, cut boundaries,
  profile policy, and strand fields.
- `candidate_id` is a route-aware row identifier. Use `left_base`,
  `right_base`, and `profile_s3s2s1s0` when grouping by scar sequence rather
  than enzyme route.
- `export/table__scar_nick_candidate_pair_calls.csv` gives one row per
  candidate and S-site, including left/right bases, aligned right base, pair
  class, mismatch flags, and T4 mismatch tier.
- `analysis/views/scar_nick_terminal_nick.scar_nick_visual.v1.jsonl` is the
  contract source for the pre/post terminal-nick plot.

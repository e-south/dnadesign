# scar_nick Package Map

`scar_nick` models a Type IIS retained scar followed by an exact terminal nick.
It is the Cruncher surface for retron hairpin base-junction candidates, not a
phenotype predictor.

## Data Flow

1. `load.py` validates the YAML spec boundary.
2. `geometry.py` audits release-enzyme and nickase compatibility.
3. `semantics.py` owns shared ontology constants and strand-to-visual-row
   mapping.
4. `profiles.py` calls the four scar positions in canonical `S3_S2_S1_S0`
   order.
5. `policy.py` classifies each profile as `active`, `reserve`, or `reject`.
6. `candidates.py` builds per-pair candidate records and rejection reasons.
7. `planner.py` enumerates feasible left/right scar pairs and builds the report
   model.
8. `ranking.py` performs deterministic bucket coverage and tie-breaking.
9. `tables.py` writes the CSV handoff tables.
10. `artifacts.py` owns artifact paths, manifests, snapshots, and generic
   persistence helpers.
11. `visual_geometry.py` builds display-coordinate spans and aligned sequences.
12. `view_contracts.py` builds the pre/post terminal-nick visual contract.
13. `visual_publication.py` writes visual bundles and checks artifact drift.
14. `reporting.py` renders the Markdown run summary.

## Core Semantics

- `S0` must be a Watson-Crick match.
- `S3`, `S2`, and `S1` are design variables.
- `W` means `G:T` or `T:G`; it is not a hard mismatch.
- `MXXM` is a reserve middle-middle hard-mismatch profile, not an active panel
  default.
- `XXMM` and `XMXM` are active S3-edge double-hard profiles when they satisfy
  the rest of the ligation policy.
- Visual rows are mapped explicitly from strands: top -> primary, bottom ->
  complement. Post-release scar fill follows the surviving strand; the annealed
  adapter fill follows the nicked strand.

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
- `*_pair_identity` fields and `pair_identity` use the physical S-site pair
  (`left_nt:right_nt`). `aligned_right_nt` and `aligned_pair_identity` remain
  available for Watson-Crick complement audits.
- `analysis/views/scar_nick_terminal_nick.scar_nick_visual.v1.jsonl` is the
  contract source for the pre/post terminal-nick plot. The post-release junction
  row displays the raw right-hand bases in S3/S2/S1/S0 order, so visual `W`
  calls appear as physical `G:T` or `T:G` pairs.

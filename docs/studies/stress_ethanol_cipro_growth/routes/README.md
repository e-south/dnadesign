## stress_ethanol_cipro_growth Routes

**Last verified:** 2026-05-18

Use this page after the checked-in study status tells you where the record stands. Keep this file as the one-hop handoff map; put downstream detail in focused files under this directory.

- Status: `uv run ops progress show studies.stress-ethanol-cipro-growth.status --json`
- Preflight: `uv run ops progress show studies.stress-ethanol-cipro-growth.preflight --scope next --command-timeout-seconds 30 --json`
- Snapshot route inventory: `evidence.analysis_surfaces.{densegen,latentdna,cluster}`

### Route Index

| Surface | Owner | State | Detail |
| --- | --- | --- | --- |
| DenseGen EDA | `densegen` | `attention` | [DenseGen](source/densegen.md) |
| Infer lanes | `infer` | `complete` for supported Evo2 7B sidecars | [Infer](compute/infer.md) |
| Construct anchor/context refresh | `usr` plus `construct` | `complete` | [Construct](source/construct.md) |
| LatentDNA comparison surface | `latentdna` | `attention` | [LatentDNA](analysis/latentdna.md) (`routes/analysis/latentdna.md`) |
| Cluster exploration | `cluster` | `planned` | [Cluster](analysis/cluster.md) |
| OPAL campaigns | `opal` | `candidate_table_materialized_pre_assay` | [OPAL](decision/opal.md) (`routes/decision/opal.md`) |

### Terminology Guardrails

- DenseGen generation plans are biological generation conditions such as `background_only`, `ethanol`, `ciprofloxacin`, and `ethanol_ciprofloxacin`.
- OPAL campaigns are downstream objectives: ethanol factor, ciprofloxacin factor, and AND. AND is not a synonym for every `ethanol_ciprofloxacin` DenseGen row.
- OPAL reads an `opal_candidate_feature_table`, not just a matrix. The materialized table is the dense generated promoter subset in `usr_prom_eth_cip_opal_candidates` with X column `latentdna__evo2_7b__context_anchor_mean_bidir_concat`.
- SFXI state order for these campaigns is `[00, 10, 01, 11]`.
- Study lifecycle phases are record-plane state labels such as the current `latentdna_reference_normalization_audit`; they are not DenseGen generation plans.
- Infer lanes are model-family and dataset-target configs such as `anchor_only_20b` or `anchor_plus_template_7b`; they are not lifecycle phases.
- Route `Plane` values use the repo-wide enum from `ARCHITECTURE.md`. If extra nuance is needed, use `Surface role` rather than inventing a new plane name.

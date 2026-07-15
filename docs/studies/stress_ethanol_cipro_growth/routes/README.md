---
doc_id: study-stress-ethanol-cipro-growth-routes
surface: study-route-map
study_id: stress_ethanol_cipro_growth
owner: dnadesign-maintainers
last_verified: 2026-07-15
entrypoint: self
status_surface: studies.stress-ethanol-cipro-growth.status
preflight_surface: studies.stress-ethanol-cipro-growth.preflight
---

## stress_ethanol_cipro_growth Routes

**Owner:** dnadesign-maintainers

**Last verified:** 2026-07-15

Use this page after the checked-in study status tells you where the record stands. Keep this file as the one-hop handoff map; put downstream detail in focused files under this directory.

- Status: `uv run ops progress show studies.stress-ethanol-cipro-growth.status --json`
- Preflight: `uv run ops progress show studies.stress-ethanol-cipro-growth.preflight --scope next --command-timeout-seconds 30 --json`
- Snapshot route inventory: `evidence.analysis_surfaces.{densegen,latentdna,cluster}`

### Navigation Header

| Need | Surface |
| --- | --- |
| Primary route | this page |
| Status | `uv run ops progress show studies.stress-ethanol-cipro-growth.status --json` |
| Preflight | `uv run ops progress show studies.stress-ethanol-cipro-growth.preflight --scope next --command-timeout-seconds 30 --json` |
| Machine-readable contract | `../operations/ops.study.yaml` |
| Study intent and semantic guardrails | [Promoter design intent](../contexts/promoter-design-intent.md) |

### Route Index

| Surface | Owner | State | Detail |
| --- | --- | --- | --- |
| DenseGen EDA | `densegen` | `attention` | [DenseGen](source/densegen.md) |
| Infer lanes | `infer` | `complete` for supported Evo2 7B sidecars | [Infer](compute/infer.md) |
| Construct anchor/context refresh | `usr` plus `construct` | `complete` | [Construct](source/construct.md) |
| LatentDNA comparison surface | `latentdna` | `x_selected_appendix_optional` | [LatentDNA](analysis/latentdna.md) (`routes/analysis/latentdna.md`) |
| Cluster exploration | `cluster` | `planned` | [Cluster](analysis/cluster.md) |
| Reader observations and candidate identity | `stress study` | `repeat_review_required` | `src/dnadesign/studies/units/stress_ethanol_cipro_growth/{response_window_observations,promoter_candidate_bindings}/` |
| OPAL campaigns | `opal` | `round0_metric_review` | [OPAL](decision/opal/) (`routes/decision/opal/README.md`) |

### Terminology Guardrails

- The study's center of gravity is specification-driven promoter design across
  regulatory contexts; ethanol/ciprofloxacin is the tractable case study, not
  the whole contribution.
- DenseGen generation plans are biological generation conditions such as `background_only`, `ethanol`, `ciprofloxacin`, and `ethanol_ciprofloxacin`.
- OPAL campaigns are downstream objectives: ethanol factor, ciprofloxacin factor, and AND. AND is not a synonym for every `ethanol_ciprofloxacin` DenseGen row.
- OPAL reads an `opal_candidate_feature_table`, not just a matrix. The materialized table is the dense generated promoter subset plus measured pDual-10 Reader round-0 rows in `usr_prom_eth_cip_opal_candidates` with X column `latentdna__evo2_7b__context_anchor_mean_bidir_concat`.
- SFXI state order for these campaigns is `[00, 10, 01, 11]`.
- Repeat-label truth and model decision quality are independent gates. The
  observation policy remains under repeat review, and the response metastudy
  does not authorize a selection policy.
- Study lifecycle phases are record-plane state labels such as the current `opal_candidate_table_pre_assay`; they are not DenseGen generation plans.
- Infer lanes are model-family and dataset-target configs such as `anchor_only_20b` or `anchor_plus_template_7b`; they are not lifecycle phases.
- Route `Plane` values use the repo-wide enum from `ARCHITECTURE.md`. If extra nuance is needed, use `Surface role` rather than inventing a new plane name.

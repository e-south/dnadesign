---
doc_id: study-stress-ethanol-cipro-growth-routes
surface: study-route-map
study_id: stress_ethanol_cipro_growth
owner: dnadesign-maintainers
last_verified: 2026-08-01
entrypoint: self
status_surface: studies.stress-ethanol-cipro-growth.status
preflight_surface: studies.stress-ethanol-cipro-growth.preflight
---

## stress_ethanol_cipro_growth Routes

Use this page to choose one owner and one next artifact. Current facts belong in
the [checked-in status](../record/status.md); detailed scientific interpretation
belongs in `contexts/`; executable declarations belong in `operations/`.

- Status summary: `uv run ops progress show studies.stress-ethanol-cipro-growth.status --json | jq '{state, summary, opal: .evidence.opal}'`
- Preflight: `uv run ops progress show studies.stress-ethanol-cipro-growth.preflight --scope next --command-timeout-seconds 30 --json`
- Snapshot route inventory: `evidence.analysis_surfaces.{densegen,latentdna,cluster}`

### First checks

| Need | Surface |
| --- | --- |
| Primary route | This page |
| Status | `uv run ops progress show studies.stress-ethanol-cipro-growth.status --json \| jq '{state, summary, opal: .evidence.opal}'` |
| Preflight | `uv run ops progress show studies.stress-ethanol-cipro-growth.preflight --scope next --command-timeout-seconds 30 --json` |
| Machine-readable contract | `../operations/ops.study.yaml` |
| Study intent and semantic guardrails | [Promoter design intent](../contexts/promoter-design-intent.md) |

### Reader-to-OPAL path

These are sequential contracts, not interchangeable names for one dataset.
Each step verifies its input before publishing the next artifact.

| Step | Owner | Input and output | Open next |
| --- | --- | --- | --- |
| Measured records | Reader | Raw assay sources become verified measurement, event-window, and plot records. Reader does not assign study candidates or objectives. | [Study observation adapter](../../../../src/dnadesign/studies/units/stress_ethanol_cipro_growth/response_window_observations/README.md) |
| Candidate identity | Stress study | Reader aliases resolve to exact promoter candidates and sequence digests. | [Candidate bindings](../../../../src/dnadesign/studies/units/stress_ethanol_cipro_growth/promoter_candidate_bindings/README.md) |
| Candidate observations | Stress study | Verified Reader reductions plus explicit repeat decisions become objective-neutral candidate observations. | [Response-window observations](../../../../src/dnadesign/studies/units/stress_ethanol_cipro_growth/response_window_observations/README.md) |
| Candidate features | Stress study and LatentDNA | The candidate table binds each candidate to the selected fixed-length model input. | [Candidate-table contract](../contexts/opal/candidate-table.md) |
| Observed labels | Stress study | An approved observation bundle becomes an immutable OPAL label publication; this step does not score an objective. | [Label promotion](../../../../src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_window_label_promotion/README.md) |
| Objective interpretation | OPAL mathematics plus stress-study policy | SFXI, RMF, and MSRB interpret declared vectors under separate contracts. The active campaign uses MSRB; SFXI and RMF remain comparison evidence. | [OPAL context](../contexts/opal/README.md) |
| Campaign state | OPAL plus checked-in study record | Validated features and labels feed model fitting, scoring, selection, ledgers, and the recorded physical handoff. | [OPAL route and commands](decision/opal/README.md) |

Stop when the next contract is absent or fails validation. Do not infer a study
candidate from a Reader `design_id`, synthesize response-window labels from an
SFXI score, or treat an accepted physical handoff as evidence of model support.

### Other study routes

| Surface | Owner | State | Detail |
| --- | --- | --- | --- |
| DenseGen EDA | `densegen` | `attention` | [DenseGen](source/densegen.md) |
| Infer lanes | `infer` | `complete` for supported Evo2 7B sidecars | [Infer](compute/infer.md) |
| Construct anchor/context refresh | `usr` plus `construct` | `complete` | [Construct](source/construct.md) |
| LatentDNA comparison surface | `latentdna` | `x_selected_appendix_optional` | [LatentDNA](analysis/latentdna.md) (`routes/analysis/latentdna.md`) |
| Cluster exploration | `cluster` | `planned` | [Cluster](analysis/cluster.md) |
| OPAL campaigns and synthesis handoff | `opal` plus stress study | `opal_assay_b1_order_ready` | [OPAL](decision/opal/) (`routes/decision/opal/README.md`) |
| Objective semantics | `opal` mathematics plus `stress study` masks, scales, and decisions | MSRB active learning probe; SFXI and RMF comparison evidence | [MSRB symbol walkthrough](../contexts/opal/multistate-response-behavior-walkthrough.html), [MSRB study binding](../contexts/opal/multistate-response-behavior.md), [SFXI](../contexts/opal/sfxi-round0-source-evidence.md), and [RMF](../contexts/opal/response-magnitude-feasibility.md) |

### Terminology Guardrails

- The study's center of gravity is specification-driven promoter design across regulatory contexts; ethanol/ciprofloxacin is the tractable case study, not the whole contribution.
- DenseGen generation plans are biological generation conditions such as `background_only`, `ethanol`, `ciprofloxacin`, and `ethanol_ciprofloxacin`.
- OPAL campaigns are downstream objectives: ethanol factor, ciprofloxacin factor, and AND. AND is not a synonym for every `ethanol_ciprofloxacin` DenseGen row.
- OPAL reads an `opal_candidate_feature_table`, not just a matrix. The materialized table is the dense generated promoter subset plus measured pDual-10 Reader round-0 rows in `usr_prom_eth_cip_opal_candidates` with X column `latentdna__evo2_7b__context_anchor_mean_bidir_concat`.
- The shared assay state order is `[00, 10, 01, 11]`; each objective owns its own vector interpretation, masks, diagnostics, and claim boundary.
- Repeat-label truth, model decision quality, and physical handoff acceptance are independent. The observation policy publishes 27 exact labels and eight exclusions; `model_support_ready` remains false even though the exact 18-candidate handoff is accepted for order.
- Study lifecycle phases are record-plane state labels such as the current `opal_assay_b1_order_ready`; they are not DenseGen generation plans.
- Infer lanes are model-family and dataset-target configs such as `anchor_only_20b` or `anchor_plus_template_7b`; they are not lifecycle phases.
- Route `Plane` values use the repo-wide enum from `ARCHITECTURE.md`. If extra nuance is needed, use `Surface role` rather than inventing a new plane name.

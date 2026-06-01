# Study Surfaces

Keep ownership boundaries clear.

## Checked-in study surfaces

- `docs/studies/<study-id>/record/status.md`: factual current datasets,
  current phase, current row counts, downstream posture, and concise next
  actions
- `docs/studies/<study-id>/routes/README.md`: one-hop study handoff page for
  DenseGen, Construct, Infer, LatentDNA, Cluster, and OPAL
- `docs/studies/<study-id>/routes/`: focused route details for owner surfaces
  that would otherwise make the one-hop map monolithic
- `docs/studies/<study-id>/record/datasets.yaml`: affiliated dataset registry,
  root semantics, and sync posture
- `docs/studies/<study-id>/operations/ops.study.yaml`: lifecycle order,
  snapshot scope, execution surfaces, and next-scope preflight grouping
- `docs/studies/<study-id>/operations/runtime/command-groups/README.md`:
  progressive-disclosure lane map for command-group navigation
- `docs/studies/<study-id>/operations/runtime/command-groups/lanes/`: bounded
  owner-lane sidecars for DenseGen, Infer, LatentDNA, Cluster, and OPAL
- `docs/studies/<study-id>/operations/runtime/command-groups/pipeline.yaml`: structural workspace,
  config, and downstream surface bindings that complement
  `operations/ops.study.yaml`, including any study-bound exploratory-analysis
  route inventory surfaced through snapshot `analysis_surfaces`; keep this as
  the canonical machine payload loaded by status and docs-contract checks
- `docs/studies/<study-id>/record/campaign.yaml`: tracked procedure set for
  `ops progress campaign`

## Tool-owned operational detail

- DenseGen analysis details stay in the DenseGen workspace README or runbook.
- Construct runtime and lineage details stay in the Construct workspace README
  or runbook.
- Infer cold-start, GPU tuning, notify handling, and rollback stay in the Infer
  workspace README or preflight-owning docs.
- LatentDNA workflow detail stays in the study-bound workspace README and
  workflow doc.
- Cluster and OPAL detail stays in their tool-owned workflow docs until the
  study owns a concrete results root or campaign config.
- OPAL campaign notebook viewing is routed through the study `routes/README.md` OPAL
  section, then the study-owned OPAL route-detail page; this skill should not
  grow a parallel OPAL command walkthrough.
- OPAL batch-0 candidate-table creation is study-owned generated data. The
  current shared table is materialized, but the contract audit remains in
  `opal_batch0/candidate_table.py`; route through `routes/decision/opal/README.md` instead of
  treating the full LatentDNA review view as the OPAL universe.
- OPAL candidate ID provenance is study-owned. Route per-ID questions through
  `opal_batch0/provenance.py`, which joins DenseGen sidecars, anchor records,
  Construct views, Infer aliases, LatentDNA rows, and OPAL records by stable
  `id`.

## Study-Owned Source Routing

- `src/dnadesign/studies/units/stress_ethanol_cipro_growth/operations/status/service.py`:
  OPS service orchestration and contract binding.
- `src/dnadesign/studies/units/stress_ethanol_cipro_growth/operations/status/snapshot.py`:
  record-backed status assembly.
- `src/dnadesign/studies/units/stress_ethanol_cipro_growth/operations/status/preflight.py`:
  preflight context resolution and check coordination.
- `src/dnadesign/studies/units/stress_ethanol_cipro_growth/operations/status/probes/`:
  semantic completeness, sequence-view, and runtime probes. Deep Infer feature
  completion stays under preflight command checks.
- `src/dnadesign/studies/units/stress_ethanol_cipro_growth/operations/status/ops/`:
  OPS provider entrypoints and checked-in status registry metadata.

## Router rule

If a follow-up question needs commands, tuning, or tool-local troubleshooting,
leave this skill and route to the owning study or tool surface.

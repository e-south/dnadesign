# Study Surfaces

Keep ownership boundaries clear.

## Checked-in study surfaces

- `docs/studies/<study-id>/status.md`: factual current datasets, current phase,
  current row counts, downstream posture, and concise next actions
- `docs/studies/<study-id>/routes.md`: one-hop study handoff page for DenseGen,
  Construct, Infer, LatentDNA, Cluster, and OPAL
- `docs/studies/<study-id>/datasets.yaml`: affiliated dataset registry, root
  semantics, and sync posture
- `docs/studies/<study-id>/ops.study.yaml`: lifecycle order, snapshot scope,
  execution surfaces, and next-scope preflight grouping
- `docs/studies/<study-id>/pipeline.yaml`: structural workspace, config, and
  downstream surface bindings that complement `ops.study.yaml`, including any
  study-bound exploratory-analysis route inventory surfaced through snapshot
  `analysis_surfaces`
- `docs/studies/<study-id>/campaign.yaml`: tracked procedure set for
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
- OPAL campaign notebook viewing is routed through the study `routes.md` OPAL
  section and `opal notebook generate/run`; this skill should not grow a
  parallel OPAL command walkthrough.

## Study-Owned Source Routing

- `src/dnadesign/studies/studies/stress_ethanol_cipro_growth/status/service.py`:
  OPS service orchestration and contract binding.
- `src/dnadesign/studies/studies/stress_ethanol_cipro_growth/status/snapshot.py`:
  record-backed status assembly.
- `src/dnadesign/studies/studies/stress_ethanol_cipro_growth/status/preflight.py`:
  preflight context resolution and check coordination.
- `src/dnadesign/studies/studies/stress_ethanol_cipro_growth/status/probes/`:
  semantic completeness, sequence-view, and runtime probes. Deep Infer feature
  completion stays under preflight command checks.
- `src/dnadesign/studies/studies/stress_ethanol_cipro_growth/status/ops/`:
  OPS provider entrypoints and checked-in status registry metadata.

## Router rule

If a follow-up question needs commands, tuning, or tool-local troubleshooting,
leave this skill and route to the owning study or tool surface.

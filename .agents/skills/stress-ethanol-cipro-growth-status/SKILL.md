---
name: stress-ethanol-cipro-growth-status
description: Report record-backed status for stress_ethanol_cipro_growth. Use for current phase, active datasets, preflight, or LatentDNA/OPAL handoff. Do not use for another study or for family-level routing.
metadata:
  version: 1.0.7
  category: workflow-automation
  tags: [studies, stress-ethanol-cipro-growth, status, routes, preflight]
---

# Stress Ethanol Cipro Growth Status

## Purpose

Answer `where is stress_ethanol_cipro_growth now?` from the checked-in study
record and the study-owned OPS status provider.

## Scope

In scope:
- `docs/studies/stress_ethanol_cipro_growth/`
- `uv run ops progress show studies.stress-ethanol-cipro-growth.status --json`
- `uv run ops progress show studies.stress-ethanol-cipro-growth.preflight --scope next --json --command-timeout-seconds 30`
- one-hop routing through `docs/studies/stress_ethanol_cipro_growth/routes/README.md`,
  with OPAL and LatentDNA detail under `docs/studies/stress_ethanol_cipro_growth/routes/`
- OPAL candidate-table and DenseGen axis-probe context pages linked from the
  OPAL route after that route is selected

Out of scope:
- family-level or cross-study status routing
- status for `regulondb_native_promoter_panel` or any other study
- reconstructing current state from workspaces when the checked-in record is missing

## Success Criteria

- Status answers come from the checked-in study record and OPS provider.
- Missing or mismatched study ids fail visibly instead of being reconstructed
  from workspace scratch outputs.
- OPAL and LatentDNA details stay behind the one-hop route map.
- Preflight evidence is machine-readable JSON when blocker/readiness state is
  requested.

## Workflow

1. Read `docs/studies/stress_ethanol_cipro_growth/operations/ops.study.yaml`,
   `record/datasets.yaml`, `operations/runtime/command-groups/README.md`,
   `operations/runtime/command-groups/pipeline.yaml`, `record/status.md`, and
   `routes/README.md`.
2. Run the status command for record posture:
   `uv run ops progress show studies.stress-ethanol-cipro-growth.status --json`.
3. Run the preflight command only for blocker or next-run readiness:
   `uv run ops progress show studies.stress-ethanol-cipro-growth.preflight --scope next --json --command-timeout-seconds 30`.
4. Route downstream work through `routes/README.md`. Open `routes/decision/opal/README.md` or
   `routes/analysis/latentdna.md` only after the one-hop route selects that owner
   surface.

## Guardrails

- This skill is study-specific. Do not generalize it to another study.
- Missing or mismatched `study_id` should fail visibly.
- OPAL candidate-table details are meaningful only in this study's candidate
  feature-table context.
- When this skill changes, run
  `bash .agents/skills/stress-ethanol-cipro-growth-status/scripts/audit-stress-ethanol-cipro-growth-status-skill.sh`.

## Required Deliverables

- current phase and next declared surface
- key dataset ids and row counts from the checked-in record
- downstream posture for LatentDNA, Cluster, and OPAL when present in status evidence
- preflight blockers only when preflight was requested
- exact missing-record or mismatch errors when the record is incomplete

## Trigger Tests

Should trigger:
- "Where is stress_ethanol_cipro_growth now?"
- "Run the stress ethanol cipro preflight for the next phase."
- "Which LatentDNA or OPAL route owns the current stress-study handoff?"

Should not trigger:
- "Where is regulondb_native_promoter_panel now?"
- "Run a generic OPAL campaign walkthrough."
- "Reconstruct current state from workspace outputs instead of the checked-in study record."

## References

- [route-matrix.md](references/route-matrix.md)
- [refresh-loop.md](references/refresh-loop.md)
- [study-surfaces.md](references/study-surfaces.md)
- [external-sources.md](references/external-sources.md)
- [test-matrix.md](references/test-matrix.md)

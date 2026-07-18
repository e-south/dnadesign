---
name: stress-ethanol-cipro-growth-status
description: Report record-backed status for stress_ethanol_cipro_growth. Use for current phase, active datasets, preflight, or LatentDNA/OPAL handoff. Do not use for another study or for family-level routing.
metadata:
  version: 1.2.1
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
- the study-owned response metric metastudy, MSRB source protocol, activation
  audit, and active campaign record when the question concerns metric validity,
  campaign state, or synthesis posture

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
5. When OPAL is in `round0_selection_review`, read
   `contexts/opal/response-metastudy.md` for comparison evidence and
   `contexts/opal/multistate-response-behavior.md` for the active scientific
   contract. Treat generated evidence as current only when its manifest and
   source run IDs match the campaign state being discussed.
6. For active-campaign readiness, verify the study-owned
   `decision/opal/multistate_response_behavior/protocol.yaml`,
   `activation_audit.json`, and the `secg_msrb_greedy` campaign route. For RMF
   questions, use `contexts/opal/response-magnitude-feasibility.md` only as the
   frozen comparator contract; `secg_rmf_greedy` is not executable.

## Guardrails

- This skill is study-specific. Do not generalize it to another study.
- Missing or mismatched `study_id` should fail visibly.
- OPAL candidate-table details are meaningful only in this study's candidate
  feature-table context.
- Do not turn a metastudy policy comparison into synthesis authorization.
- Report `secg_msrb_greedy` as the sole executable stress-study OPAL campaign
  only when the checked-in record, source protocol, and activation audit agree.
- Do not report `secg_rmf_greedy` as executable; it is immutable comparator
  evidence under the study workbench.
- Campaign activation and completed round state do not authorize synthesis or
  establish prospective optimization efficacy.
- Keep SFXI metric evidence, RF held-out support, and biological validation as
  separate claims.
- When this skill changes, run
  `bash .agents/skills/stress-ethanol-cipro-growth-status/scripts/audit-stress-ethanol-cipro-growth-status-skill.sh`.

## Required Deliverables

- current phase and next declared surface
- key dataset ids and row counts from the checked-in record
- downstream posture for LatentDNA, Cluster, and OPAL when present in status evidence
- SFXI source-run verdict, frozen RMF comparator posture, MSRB activation and
  runtime posture, and the separate model-support boundary when OPAL is in
  selection review
- preflight blockers only when preflight was requested
- exact missing-record or mismatch errors when the record is incomplete

## Output Contract

Return:

1. Checked-in study phase and next declared surface.
2. Dataset identities, row counts, and owner-specific downstream posture.
3. OPAL status separated into label truth, MSRB activation, runtime completion,
   model support, and synthesis authorization.
4. Exact record, protocol, activation-audit, campaign, or preflight evidence
   supporting each claim.
5. A visible failure or missing-artifact route instead of reconstructed state.

## Trigger Tests

Should trigger:
- "Where is stress_ethanol_cipro_growth now?"
- "Run the stress ethanol cipro preflight for the next phase."
- "Which LatentDNA or OPAL route owns the current stress-study handoff?"
- "Does the current response metric metastudy support synthesis?"
- "Is the active MSRB campaign bound to the approved study protocol?"

Should not trigger:
- "Where is regulondb_native_promoter_panel now?"
- "Run a generic OPAL campaign walkthrough."
- "Reconstruct current state from workspace outputs instead of the checked-in study record."

## Progressive Disclosure Resources

- `references/route-matrix.md`
- `references/refresh-loop.md`
- `references/study-surfaces.md`
- `references/external-sources.md`
- `references/test-matrix.md`

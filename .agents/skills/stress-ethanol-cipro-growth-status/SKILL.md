---
name: stress-ethanol-cipro-growth-status
description: Report record-backed status for the concrete stress_ethanol_cipro_growth study. Use when the user asks where this study stands, which checked-in files are authoritative, what current datasets or phase are active, or what study-owned handoff should be opened next. Do not use for another study or for family-level routing; create that study's own skill/status provider only when needed.
metadata:
  version: 1.0.0
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
- `uv run ops progress show studies.stress-ethanol-cipro-growth.preflight --scope next --json`
- one-hop routing through `docs/studies/stress_ethanol_cipro_growth/routes.md`

Out of scope:
- family-level or cross-study status routing
- status for `regulondb_native_promoter_panel` or any other study
- reconstructing current state from workspaces when the checked-in record is missing

## Workflow

1. Read `docs/studies/stress_ethanol_cipro_growth/ops.study.yaml`,
   `datasets.yaml`, `pipeline.yaml`, `status.md`, and `routes.md`.
2. Run the status command for record posture:
   `uv run ops progress show studies.stress-ethanol-cipro-growth.status --json`.
3. Run the preflight command only for blocker or next-run readiness:
   `uv run ops progress show studies.stress-ethanol-cipro-growth.preflight --scope next --json`.
4. Route downstream work through `routes.md` or the owning tool docs.

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

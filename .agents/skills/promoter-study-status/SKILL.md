---
name: promoter-study-status
description: Report record-backed promoter-study status for one live study. Use when the user asks where the promoter study stands, what phase or datasets are current, whether the checked-in study record needs a refresh, which study files are authoritative, which dataset sync posture is current, or whether the study is still source-phase versus downstream. Do not use for blockers or next-run readiness; use the Ops `usr.data-plane.promoter-study-preflight` status command. Do not use for tool-local operational walkthroughs or generic workflow discovery when no checked-in study record exists.
metadata:
  version: 0.5.1
  category: workflow-automation
  tags: [usr, promoter-study, status, routes, preflight]
---

# Promoter Study Status

## Purpose

Answer `where is the live promoter study now?` from the checked-in study
record. Route blocker questions to the Ops `usr.data-plane.promoter-study-preflight`
status command and owner-surface questions to the study's `routes.md`.

## Scope

In scope:
- active-study selection from `docs/studies/index.yaml`
- checked-in snapshot refresh through
  `ops progress show usr.data-plane.promoter-study-status --json`
- study-record questions about authoritative files, sync posture, or
  source/handoff mode
- short routing to `docs/studies/<study-id>/routes.md` and the owning study or
  tool docs

Out of scope:
- blocker or next-run readiness questions that belong to the Ops
  `usr.data-plane.promoter-study-preflight` status command
- tool-local DenseGen, Construct, Infer, LatentDNA, Cluster, or OPAL
  operational walkthroughs
- reconstructing live study state from generic runbooks, workspaces, or
  journals
- inventing current state when the checked-in study record is missing or
  inconsistent

## Success Criteria

- one checked-in study record is selected or ambiguity fails visibly
- answers separate snapshot posture from execution-readiness posture
- current dataset ids, row counts, phase, and downstream posture come from the
  checked-in study record plus the snapshot command
- cross-tool handoffs go through `routes.md` or the owning surface instead of
  expanding this skill
- missing records or stale selectors fail visibly

## Workflow

1. Select the active study record.
- Read `docs/studies/README.md` and `docs/studies/index.yaml`.
- Require `campaign.yaml`, `datasets.yaml`, `status.md`, and `ops.study.yaml`
  in `docs/studies/<study-id>/`.
- Load `pipeline.yaml` and `routes.md` when present.
- Use [study-surfaces.md](references/study-surfaces.md) for ownership
  boundaries.

2. Route the question through one ladder.
- Run `uv run ops progress show usr.data-plane.promoter-study-status --json`
  for `where is the study now?`
- Use `evidence.analysis_surfaces` in that same snapshot when the follow-up is
  really `which DenseGen plots, LatentDNA deliverables, notebooks, or Cluster
  artifact paths are available?`
- Route `what blocks execution here?` or `what should run next on this host?`
  to
  `uv run ops progress show usr.data-plane.promoter-study-preflight --scope next --json`
  instead of answering from this skill.
- Use `docs/studies/<study-id>/routes.md` for DenseGen, Construct, Infer,
  LatentDNA, Cluster, and OPAL handoff after the state or blocker question is
  answered.
- Use [route-matrix.md](references/route-matrix.md) and
  [refresh-loop.md](references/refresh-loop.md) for authoritative files, sync
  posture, source/handoff mode, and blank-thread bootstrap.

3. If the checked-in record is missing or inconsistent, fail visibly.
- Report which record files or selector fields are missing.
- Route the user to
  `src/dnadesign/usr/docs/operations/promoter-study-status-contract.md`
  instead of guessing.

## Guardrails

- `promoter-study-status` is the record-plane router.
- `uv run ops progress show usr.data-plane.promoter-study-preflight --scope next --json`
  is the execution-readiness router; it is not a repo-local skill name.
- Keep `status.md` factual and short; keep lifecycle and preflight authority in
  `ops.study.yaml`; keep structural workspace or config bindings in
  `pipeline.yaml`.
- Keep `campaign.yaml`, `datasets.yaml`, `status.md`, and `ops.study.yaml` as
  the core checked-in evidence bundle.
- Treat `datasets.yaml` as the source of affiliated dataset ids, root
  semantics, and sync posture.
- Treat `routes.md` as the study handoff page for downstream analysis and
  campaigns.
- Use repo-backed status commands only; do not reconstruct current state from
  stale notes.
- When this skill changes, run
  `bash .agents/skills/promoter-study-status/scripts/audit-promoter-study-status-skill.sh`.

## Required Deliverables

- selected study id and record root, or an explicit missing-record failure
- snapshot vs preflight distinction for the answer path used
- current phase, key datasets, row counts, and downstream posture from the
  checked-in record
- exploratory-analysis route inventory only through snapshot
  `analysis_surfaces` plus `routes.md`, not by inventing tool-local state
- next route surface: `routes.md`, the Ops `usr.data-plane.promoter-study-preflight`
  status command, or the owning tool doc or workspace
- explicit assumptions or missing artifacts when the record is incomplete

## Output

Return:
- study id
- whether the answer came from snapshot posture or preflight readiness
- current phase and next declared study surface
- current dataset ids and row counts from the checked-in record
- current downstream posture for `latentdna`, `cluster`, and `opal`
- the next owning doc, workspace, or route doc to open
- explicit blockers only when preflight was requested
- explicit missing-record errors when the study record is incomplete

## Trigger Tests

Should trigger:
- "Check where the promoter study stands right now."
- "Refresh the checked-in promoter study status."
- "Which study files are authoritative for this promoter study?"
- "Which dataset sync posture is current for the live study?"
- "Which study route should I open next for LatentDNA or OPAL?"
- "Is the live promoter study still in infer preparation or already downstream?"
- "Show the current promoter-study record and the next owner surface."

Should not trigger:
- "What blocks execution here?"
- "What should run next on this host?"
- "Which runbook should I use for infer?"
- "Show me DenseGen workspace commands."
- "Explain the cluster workflow."
- "Debug why notify profile doctor fails on this host."

## References

- [route-matrix.md](references/route-matrix.md)
- [refresh-loop.md](references/refresh-loop.md)
- [study-surfaces.md](references/study-surfaces.md)
- [external-sources.md](references/external-sources.md)

---
name: promoter-study-status
description: Answer or refresh the checked-in promoter-study status for one live study, then route agents to preflight or the study-owned route map when the question is really about blockers or the next owner surface. Use when the user asks where the promoter study stands, what phase or datasets are current, whether the checked-in study record needs a refresh, or which study-owned route doc to open next for DenseGen, Construct, Infer, LatentDNA, Cluster, or OPAL. Do not use for tool-local operational walkthroughs or generic workflow discovery when no checked-in study record exists.
metadata:
  version: 0.4.0
  category: workflow-automation
  tags: [usr, promoter-study, status, routes, preflight]
---

# Promoter Study Status

## Purpose

Give one repo-backed answer to `where is the live promoter study now?` and one
clean handoff to `promoter-study-preflight` or the study's `routes.md` when the
user is really asking about blockers or downstream branching.

## Scope

In scope:
- active-study selection from `docs/studies/index.yaml`
- checked-in snapshot refresh through
  `ops progress show usr.data-plane.promoter-study-status --json`
- explicit escalation to `promoter-study-preflight` for blockers or next-run
  readiness
- short routing to `docs/studies/<study-id>/routes.md` and the owning study or
  tool docs

Out of scope:
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

2. Refresh the cheap record-plane snapshot first.
- Run `uv run ops progress show usr.data-plane.promoter-study-status --json`.
- Use [refresh-loop.md](references/refresh-loop.md) for the exact refresh
  contract.
- Do not infer live status from demo workspaces, journal notes, or generic tool
  docs.

3. Escalate only when the user is asking about blockers or next-run readiness.
- Run
  `uv run ops progress show usr.data-plane.promoter-study-preflight --scope next --json`
  when the question is really `what blocks execution here?` or `what should run
  next on this host?`.
- Keep snapshot and preflight separate. Snapshot is repo-backed study posture;
  preflight is command-level execution readiness.

4. Route follow-up intent to the study-owned handoff surfaces.
- Use `docs/studies/<study-id>/routes.md` for DenseGen, Construct, Infer,
  LatentDNA, Cluster, and OPAL entrypoints.
- Use [route-matrix.md](references/route-matrix.md) when the user intent is
  ambiguous.
- Keep tool-owned operational detail in the owning workspace README, runbook,
  or workflow doc.

5. If the checked-in record is missing or inconsistent, fail visibly.
- Report which record files or selector fields are missing.
- Route the user to
  `src/dnadesign/usr/docs/operations/promoter-study-status-contract.md`
  instead of guessing.

## Guardrails

- `promoter-study-status` is the record-plane router;
  `promoter-study-preflight` is the execution-readiness router.
- Keep `status.md` factual and short; keep lifecycle and preflight authority in
  `ops.study.yaml`; keep structural workspace or config bindings in
  `pipeline.yaml`.
- Treat `datasets.yaml` as the source of affiliated dataset ids, root
  semantics, and sync posture.
- Treat `routes.md` as the one-hop study handoff page for downstream analysis
  and campaigns.
- Use repo-backed status commands only; do not reconstruct current state from
  stale notes.
- When this skill changes, run
  `bash .agents/skills/promoter-study-status/scripts/audit-promoter-study-status-skill.sh`.

## Required Deliverables

- selected study id and record root, or an explicit missing-record failure
- snapshot vs preflight distinction for the answer path used
- current phase, key datasets, row counts, and downstream posture from the
  checked-in record
- next route surface: `routes.md`, `promoter-study-preflight`, or the owning
  tool doc or workspace
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
- "Which study route should I open next for LatentDNA or OPAL?"
- "Is the live promoter study still in infer preparation or already downstream?"
- "Show the current promoter-study record and the next owner surface."

Should not trigger:
- "Which runbook should I use for infer?"
- "Show me DenseGen workspace commands."
- "Explain the cluster workflow."
- "Debug why notify profile doctor fails on this host."

## References

- [route-matrix.md](references/route-matrix.md)
- [refresh-loop.md](references/refresh-loop.md)
- [study-surfaces.md](references/study-surfaces.md)
- [external-sources.md](references/external-sources.md)

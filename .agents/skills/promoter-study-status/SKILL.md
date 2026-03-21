---
name: promoter-study-status
description: Answer or refresh the dnadesign promoter-study status across DenseGen, USR, Construct, Infer, Cluster or OPAL, and optional Notify by reading the checked-in study record first, including the affiliated-dataset registry for local and remote sync posture, then running the explicit status commands. Use when the user asks where the promoter study stands, which datasets and row counts are current, which infer slices are done or pending, whether the downstream dataset is ready, how affiliated datasets sync between locations such as BU SCC and local roots, or what batch step should run next. Do not use for generic workflow discovery when no checked-in study record exists, or for tool-local questions that do not need cross-tool study status.
metadata:
  version: 0.3.0
  category: workflow-automation
  tags: [usr, promoter-study, densegen, construct, infer, cluster, notify, status]
---

# Promoter Study Status

## Purpose

Give naive agents one deterministic repo-level route to answer "where is the
real promoter study right now?" without reconstructing the whole DenseGen ->
USR -> Construct -> Infer -> Cluster or OPAL path from scratch.

## Scope

In scope:
- repo-level promoter-study status questions that cross DenseGen, USR,
  Construct, Infer, Cluster or OPAL, and optional Notify
- refreshing checked-in study records from explicit CLI status surfaces
- reporting blockers when the repo lacks a usable live study record

Out of scope:
- generic workflow discovery with no live study record
- tool-local questions that do not need cross-tool study state
- inventing status from demo workspaces, journals, or stale notes

## Success Criteria

- exactly one checked-in study record is selected or the ambiguity is reported
- the answer names explicit dataset ids, row counts, infer slices, and next
  actions from the checked-in record
- the answer includes affiliated dataset sync posture from `datasets.yaml`
- every freshness claim is backed by an explicit command run in the refresh loop
- missing records or ambiguous study selection fail visibly instead of causing
  guessed status

## Guardrails

- Start with `docs/studies/README.md` and
  `src/dnadesign/usr/docs/operations/promoter-study-status-contract.md`.
- Treat `docs/studies/promoter/<study-id>/campaign.yaml`,
  `docs/studies/promoter/<study-id>/datasets.yaml`, and `status.md` as the only
  valid source for live dataset ids, local-vs-remote sync posture, row targets,
  completed infer slices, and next actions.
- Use `ops progress ...` and `usr ...` commands only to refresh those checked-in
  study records; do not infer live study state from demo workspaces, journal
  notes, or generic runbooks.
- If no checked-in study record exists, say so explicitly and route the user to
  the status contract instead of inventing current state.

## Workflow

1. Locate the active study record
- Look under `docs/studies/promoter/`.
- If exactly one study directory contains `campaign.yaml`, `datasets.yaml`, and
  `status.md`, treat it as the active study.
- If more than one candidate exists, require an explicit study id or path.

2. Refresh the shared status surface
- Run:
  `uv run ops progress campaign --repo-root <repo-root> --manifest docs/studies/promoter/<study-id>/campaign.yaml`
- Run:
  `uv run ops progress show usr.data-plane.promoter-feature-matrix --repo-root <repo-root> --usr-root <usr-root> --dataset <feature-dataset>`

3. Refresh the data-plane evidence
- Run `uv run usr --root <usr-root> validate <feature-dataset> --strict`.
- Inspect one small sample with `usr head`, including source labels, construct
  lineage, and one explicit `infer__...` column.

4. Refresh affiliated-dataset sync posture
- For each sync-enabled entry in `docs/studies/promoter/<study-id>/datasets.yaml`, run:
  `uv run usr --root <usr-root> info <dataset-id> --format json`
- Then run:
  `uv run usr --root <usr-root> diff <dataset-id> <remote-name> --audit-json-out docs/studies/promoter/<study-id>/audits/<dataset-id>--<remote-name>-diff.json`
- Then run:
  `uv run ops progress show usr.data-plane.hpc-sync --sync-audit-json docs/studies/promoter/<study-id>/audits/<dataset-id>--<remote-name>-diff.json`
- If `onboard_mode: existing_remote`, keep `strict_bootstrap_id: true` and
  require an explicit dataset id for the first pull.

5. Refresh pending infer slices
- For each pending config in `docs/studies/promoter/<study-id>/status.md`, run:
  `uv run infer validate config --config <infer-config>`
- Then run:
  `uv run infer run --config <infer-config> --dry-run`

6. Refresh batch or notify evidence only when asked
- DenseGen or Infer batch posture: read the recorded runbook or `qsub` command
  in `status.md`.
- Notify posture: use `notify setup resolve-events` plus
  `notify usr-events watch --dry-run`.

## Output

Return:
- study id
- live feature dataset and row count
- source datasets named in the checked-in study record
- affiliated dataset registry entries and sync posture
- completed versus pending infer slices
- rollback paths (`infer prune`, `usr maintenance overlay-remove`,
  `usr maintenance overlay-compact`)
- batch and notify readiness
- next actions
- explicit blockers when the repo lacks the required study record

## Trigger Tests

Should trigger:
- "Check where the promoter study stands right now."
- "Which infer slices are already written for the current DenseGen to USR study?"
- "Is the current promoter feature dataset ready for cluster or OPAL?"
- "Refresh the checked-in promoter study status and tell me the next batch step."
- "Which study datasets live on BU SCC versus locally, and how do they sync?"

Should not trigger:
- "Which runbook should I use for infer?"
- "Explain the construct CLI."
- "Show me DenseGen workspace commands."

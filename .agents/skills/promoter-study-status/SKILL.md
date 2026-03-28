---
name: promoter-study-status
description: Answer or refresh the dnadesign promoter-study status across DenseGen, USR, Construct, Infer, Cluster or OPAL, and optional Notify by reading the checked-in study record first, including the affiliated-dataset registry and optional pipeline execution map for local, remote, and batch posture, then running the explicit status commands. Use when the user asks where the promoter study stands, which datasets and row counts are current, which infer slices are done or pending, whether the downstream dataset is ready, how affiliated datasets sync between locations such as BU SCC and local roots, or what batch step should run next. Do not use for generic workflow discovery when no checked-in study record exists, or for tool-local questions that do not need cross-tool study status.
metadata:
  version: 0.3.3
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
- the answer distinguishes canonical shared datasets from workspace-local export
  roots
- every freshness claim is backed by an explicit command run in the refresh loop
- missing records or ambiguous study selection fail visibly instead of causing
  guessed status

## Guardrails

- Start with `docs/studies/README.md`, `docs/studies/index.yaml`, and
  `src/dnadesign/usr/docs/operations/promoter-study-status-contract.md`.
- Keep the OPS mental model in `docs/operations/ops-mental-model.md` in sync
  with this skill: snapshot is the record-plane summary, preflight is the
  execution-readiness summary, and `missing > attention > ok` is the global
  severity order.
- For the active `stress_ethanol_cipro_growth` study, treat
  `promoter-study-preflight` as strict submit-readiness for the default
  notify-enabled Infer presets: missing notify env/TLS wiring, failed notify
  profile or event-path checks, and failed notify-enabled runbook plans are
  blockers, not advisories. Use batch-only routes only when the user explicitly
  opts out of notify.
- Treat `docs/studies/index.yaml`,
  `docs/studies/<study-id>/campaign.yaml`,
  `docs/studies/<study-id>/datasets.yaml`,
  `docs/studies/<study-id>/status.md`,
  `docs/studies/<study-id>/ops.study.yaml`, and optional
  `docs/studies/<study-id>/pipeline.yaml` as the checked-in source set for
  live dataset ids, local-vs-remote sync posture, row targets, completed infer
  slices, study-owned Construct or Infer surfaces, and next actions.
  `ops.study.yaml` is the OPS-facing source of lifecycle order, declared
  execution surfaces, and next-scope preflight grouping. `pipeline.yaml`
  remains supplemental study-owned runtime context when the study needs exact
  Construct, Infer, or Notify mappings beyond that contract. Infer Notify
  profile paths should be derived from the checked-in Infer lane configs rather
  than stored separately in `pipeline.yaml`.
- Use `root_kind` and `status` in `datasets.yaml` to tell canonical shared USR
  roots apart from workspace-local export roots and planned-but-not-yet-created
  datasets.
- Use `ops progress ...` and `usr ...` commands only to refresh those checked-in
  study records; do not infer live study state from demo workspaces, journal
  notes, or generic runbooks.
- Treat `promoter-study-status` as the repo-backed snapshot surface and
  `promoter-study-preflight` as the execution-readiness surface. Do not answer
  "what should run next?" from snapshot alone when preflight blockers are in
  scope.
- When this skill changes, run the deterministic repo-local audit:
  `bash .agents/skills/promoter-study-status/scripts/audit-promoter-study-status-skill.sh`
- If no checked-in study record exists, say so explicitly and route the user to
  the status contract instead of inventing current state.

## Workflow

1. Locate the active study record
- Read `docs/studies/index.yaml`.
- If `docs/studies/index.yaml` is missing or invalid, report that no live
  promoter study record is checked in yet.
- Require `active_study_id`, `family`, and `record_root` for the selected
  study entry.
- Require `campaign.yaml`, `datasets.yaml`, `status.md`, and `ops.study.yaml`
  in the matching study directory.
- If `pipeline.yaml` exists, load it as supplemental runtime context before
  answering exact Construct, Infer, batch, or Notify next-step questions that
  need more detail than the declared `ops.study.yaml` surfaces.
- If the registry and checked-in study directory disagree, fail visibly instead
  of scanning for a best guess.

2. Refresh the shared status surface
- Run:
  `uv run ops progress show usr.data-plane.promoter-study-status --json`
- Use that output as the repo-scoped one-command summary for current phase,
  declared datasets, next ready phase, and missing execution surfaces before
  deeper probes. Treat host-local advisories there as advisory only.
- Snapshot means repo-backed study posture: declared datasets, row targets,
  lifecycle state, study-owned execution surfaces, and sync evidence already
  checked in.
- Preflight is the escalation path when a user asks for blockers, failed
  commands, missing artifacts, or scheduler/readiness posture right now.
- When the user needs command-level blockers rather than the cheap snapshot,
  run:
  `uv run ops progress show usr.data-plane.promoter-study-preflight --scope next --json`
- Use the preflight output when the question is "what fails right now?" across
  DenseGen, Construct, Infer, Notify, and batch-plan surfaces.
- `--scope next` narrows blocker reporting first. It can still run every
  declared check attached to a broad preparation phase such as
  `infer_batch_preparation`, so do not describe it as cheap.
- Preflight means execution-readiness blockers: generic checks, grouped by the
  declared study contract, with ontology fields such as `observes_plane`,
  `summary_scope`, `scope`, `phase_id`, `check_group`, `kind`, `surface_id`,
  and `artifact_id`.
- Use `next_in_progress_phase`, `next_ready_phase`, and the declared lifecycle
  fields from `ops.study.yaml` when answering "what should run next?" instead
  of reconstructing that ordering from generic runbooks.
- This is a repo-local project skill. If Codex was launched outside the
  `dnadesign` checkout, the skill may not be auto-advertised even though this
  file exists. In that case, use the `ops progress` commands directly instead
  of assuming project-scope skill discovery is active.
- If you are outside the repo checkout or you need a non-active study, rerun with:
  `uv run ops progress show usr.data-plane.promoter-study-status --repo-root <repo-root> --study-dir docs/studies/<study-id> --json`
- Use the same `--repo-root ... --study-dir ...` shape for `promoter-study-preflight`.
- Run:
  `uv run ops progress campaign --repo-root <repo-root> --manifest docs/studies/<study-id>/campaign.yaml`
- If `status.md` names a current canonical feature dataset, run:
  `uv run ops progress show usr.data-plane.promoter-feature-matrix --repo-root <repo-root> --usr-root <usr-root> --dataset <feature-dataset>`
- If the study record says the canonical feature dataset is still `n/a`, report
  that the study is still in source-assembly mode and skip the feature-matrix
  refresh instead of inventing one.

3. Refresh the data-plane evidence
- Run `uv run usr --root <usr-root> validate <feature-dataset> --strict`.
- Inspect one small sample with `usr head`, including source labels, construct
  lineage, and one explicit `infer__...` column.

4. Refresh affiliated-dataset sync posture
- For each sync-enabled entry in `docs/studies/<study-id>/datasets.yaml`, run:
  `uv run usr --root <usr-root> info <dataset-id> --format json` when that
  entry is locally present
- Then run:
  `uv run usr --root <usr-root> diff <dataset-id> <remote-name> --audit-json-out docs/studies/<study-id>/audits/<dataset-id>--<remote-name>-diff.json`
- Then run:
  `uv run ops progress show usr.data-plane.hpc-sync --sync-audit-json docs/studies/<study-id>/audits/<dataset-id>--<remote-name>-diff.json`
- If `onboard_mode: existing_remote`, keep `strict_bootstrap_id: true` and
  require an explicit dataset id for the first pull.

5. Refresh pending infer slices
- For each pending config in `docs/studies/<study-id>/status.md`, run:
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
- whether each answer came from snapshot posture or preflight readiness
- live feature dataset and row count, or an explicit statement that the study is
  still source-phase with no canonical feature dataset yet
- source datasets named in the checked-in study record
- affiliated dataset registry entries and sync posture
- study-owned Construct, Infer, and batch surfaces from `ops.study.yaml`, plus
  any supplemental runtime detail from `pipeline.yaml` and derived Infer Notify
  profile paths from the checked-in lane configs
- completed versus pending infer slices
- rollback paths (`infer prune`, `usr maintenance overlay-remove`,
  `usr maintenance overlay-compact`)
- batch and notify readiness
- preflight ontology fields when blockers are reported (`scope`, `phase_id`,
  `kind`, `surface_id`, `artifact_id`)
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

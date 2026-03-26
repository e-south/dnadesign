## Study Records

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-25

Use this index when the question is "where is the real study right now?" rather
than "which generic workflow should I use?"

Study records are checked-in status artifacts for one live effort. They are not
runbooks, and they are not generated outputs.
They are the record plane: `ops progress` reads these checked-in artifacts as
observation surfaces, while `ops runbook` stays in the control plane for
planning and execution.

Authority chain: `docs/studies/index.yaml` selects the active study,
the matching `docs/studies/<study-id>/` directory holds the required
`campaign.yaml`, `datasets.yaml`, `status.md`, and `ops.study.yaml`, and may
also carry an optional `pipeline.yaml` when the study owns checked-in
Construct, Infer, batch, or Notify execution surfaces.

Keep four complementary artifacts for each real study:

- `campaign.yaml`: workflow progress and registered procedure evidence
- `datasets.yaml`: machine-readable registry of affiliated USR datasets and
  sync posture across local and remote locations
- `status.md`: human-readable current state, row targets, and next actions
- `ops.study.yaml`: OPS-facing study contract for lifecycle order, record
  sources, artifacts, execution surfaces, and explicit preflight scope/check
  planning

Keep the code boundary equally explicit: study-family implementation code lives
under `src/dnadesign/studies/`, not under `src/dnadesign/ops/`. OPS reads the
checked-in record and dispatches the provider, but the family-owned snapshot
and preflight logic stays with the family package.

When a study already owns concrete execution surfaces, add one optional fifth
artifact:

- `pipeline.yaml`: machine-readable map of the study-owned workspace, config,
  batch, and runtime surfaces that a naive agent should follow next without
  duplicating study narrative or lifecycle authority

Use the study record even when the effort is still in the source-assembly phase.
An active study does not need to wait until the final feature matrix already
exists, but the record must say explicitly whether the current shared feature
dataset is materialized or still pending.

Use `docs/studies/index.yaml` to declare which checked-in study record is
active and which study family owns its adapter.

### Declared layout

Keep promoter-study records under:

```text
docs/studies/<study-id>/
  campaign.yaml
  datasets.yaml
  status.md
  ops.study.yaml
  pipeline.yaml  # optional but recommended once the study owns execution surfaces
  audits/
```

and keep the study selector at:

```text
docs/studies/index.yaml
```

- `campaign.yaml` is the explicit multi-step manifest generated from
  `uv run ops progress scaffold ...` and then filled with the real artifact
  paths. Use v2 semantics with explicit `version`, `path_base`, and per-step
  `inputs:` mappings.
- `datasets.yaml` declares which USR datasets belong to the study, whether each
  location is a shared USR root or a workspace-local export root, how
  it should be onboarded, and how it syncs to remotes such as `cluster` or a
  study-specific workspace-export remote.
- `status.md` is the human-readable note that records row targets, source
  datasets, infer slice status, rollback paths, and next actions.
- `ops.study.yaml` is the machine-readable OPS contract for lifecycle
  ordering, record sources, artifacts, execution surfaces, repo-scoped
  snapshot posture, and explicit preflight scope/check planning.
- `pipeline.yaml`, when present, records the exact Construct workspace,
  Infer configs, batch presets, and other runtime surfaces that belong to the
  real study rather than to a generic demo. Keep study meaning and lifecycle in
  `status.md` and `ops.study.yaml`.
- `audits/` stores machine-readable sync audit JSON files referenced from
  `datasets.yaml`.

### Create or refresh a promoter-study record

1. Create the study directory:
   `mkdir -p docs/studies/<study-id>`
2. If `docs/studies/index.yaml` is missing, bootstrap it once:
   `cp docs/templates/promoter-study-index.yaml docs/studies/index.yaml`
3. Generate the manifest:
   `uv run ops progress scaffold --related-to usr.data-plane.promoter-feature-matrix --repo-root <repo-root> > docs/studies/<study-id>/campaign.yaml`
4. Copy the dataset registry template:
   `cp docs/templates/promoter-study-datasets.yaml docs/studies/<study-id>/datasets.yaml`
5. Copy the status-note template:
   `cp docs/templates/promoter-study-status.md docs/studies/<study-id>/status.md`
6. Copy the OPS-facing study contract template:
   `cp docs/templates/promoter-study-ops.study.yaml docs/studies/<study-id>/ops.study.yaml`
7. Create the audit directory:
   `mkdir -p docs/studies/<study-id>/audits`
8. Edit the checked-in `index.yaml` plus the new `campaign.yaml`, `datasets.yaml`, `status.md`, and `ops.study.yaml` so they point at the real study ids, paths, and commands.
9. If the study already has concrete Construct, Infer, or batch surfaces,
   add `docs/studies/<study-id>/pipeline.yaml` and record those exact
   paths there.
10. Refresh evidence with:
   `uv run ops progress campaign --repo-root <repo-root> --manifest docs/studies/<study-id>/campaign.yaml`

### Dataset registry contract

Use `datasets.yaml` to keep one DRY declaration of study-affiliated datasets.
Every dataset entry should answer:

- which role the dataset plays in the study
- which `usr_root` and dataset id own that artifact locally
- whether that root is the shared root, a workspace-local export root, or
  `external_usr`
- whether the dataset is already `present` or still `planned`
- whether the dataset is being onboarded as `existing_local`,
  `existing_remote`, `existing_both`, or `create_new`
- whether the source of truth is `local`, `remote`, or `shared`
- whether sync is required, which remote profile is used, and where the sync
  audit JSON should be written

Keep the terminology strict:

- `workspace_local_export`: a producer-owned USR-shaped dataset rooted under a
  tool workspace rather than the shared study root
- `shared`: the cross-tool study copy rooted under an explicit shared
  USR root such as `src/dnadesign/usr/datasets`
- `external_usr`: an operator-owned but non-repo USR root that is still
  explicit in the study record

For a sync-enabled dataset, refresh evidence with explicit commands such as:

```bash
# Confirm the local dataset exists and inspect its metadata shape.
uv run usr --root <usr-root> info <dataset-id> --format json
# Capture remote drift against the study's declared remote profile.
uv run usr --root <usr-root> diff <dataset-id> <remote-name> \
  --audit-json-out docs/studies/<study-id>/audits/<dataset-id>--<remote-name>-diff.json
# Summarize the same sync audit through the registered status view.
uv run ops progress show usr.data-plane.hpc-sync \
  --sync-audit-json docs/studies/<study-id>/audits/<dataset-id>--<remote-name>-diff.json
```

If a dataset is being onboarded from a remote-only starting point, keep
`strict_bootstrap_id: true` in `datasets.yaml` and use an explicit dataset id
for the first pull rather than relying on local name guessing.

### Status lookup rules

- One-command checked-in status summary:
  `uv run ops progress show usr.data-plane.promoter-study-status`
- One-command checked-in command preflight:
  `uv run ops progress show usr.data-plane.promoter-study-preflight`
- To pin a non-active study or run from outside the repo checkout, add:
  `--repo-root <repo-root> --study-dir docs/studies/<study-id>`
- The repo-local promoter-study skill lives at `.agents/skills/promoter-study-status/SKILL.md`, but native project-scope skill discovery only picks it up when Codex is launched from this repo root or another path inside this checkout. If the session started elsewhere, use the two `ops progress` commands above directly.
- Read `docs/studies/index.yaml` first.
- `active_study_id` must name a study declared under `studies:`.
- The selected study entry must declare `family` and `record_root`.
- The selected study directory must contain `campaign.yaml`, `datasets.yaml`,
  `status.md`, and `ops.study.yaml`.
- `ops.study.yaml` is the OPS-facing source of lifecycle phase order, record
  sources, execution surfaces, repo snapshot summary scope, and preflight
  scope/check posture.
  `pipeline.yaml` remains the
  exact execution-surface map when the study owns concrete Construct, Infer, or
  batch assets.
- If `pipeline.yaml` exists, treat it as the study-owned execution map for the
  next Construct, Infer, batch, or Notify step; do not reconstruct that path
  from generic workspace docs.
- Use `ops progress show usr.data-plane.promoter-study-status` for the cheap
  record-plane snapshot, then escalate to
  `ops progress show usr.data-plane.promoter-study-preflight --scope next --json`
  when the question is "what should run next?" or "which execution-readiness
  blockers remain right now?"
- If the registry and directory contents disagree, fail visibly and fix the
  registry before asking agents for live study status.

### Related docs

- [Promoter study status contract](../../src/dnadesign/usr/docs/operations/promoter-study-status-contract.md)
- [Promoter study preflight contract](../../src/dnadesign/usr/docs/operations/promoter-study-preflight.md)
- [Promoter study index template](../templates/promoter-study-index.yaml)
- [Promoter study datasets template](../templates/promoter-study-datasets.yaml)
- [Promoter study status template](../templates/promoter-study-status.md)
- [Promoter study OPS contract template](../templates/promoter-study-ops.study.yaml)
- [Documentation index](../README.md)

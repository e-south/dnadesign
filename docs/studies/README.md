## Study Records

**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-29

Use this index for `where is the real study right now?`, not for generic
workflow routing.

Study records are checked-in status artifacts for one live effort. They are not
runbooks or generated outputs.
They are the record plane: `ops progress` reads them as observation surfaces,
while `ops runbook` stays in the control plane for planning and execution.

Family note:

- `promoter` studies use `usr.data-plane.promoter-study-status` and
  `usr.data-plane.promoter-study-preflight`.
- `cruncher` studies use `cruncher.data-plane.cruncher-study-status` and
  `cruncher.data-plane.cruncher-study-preflight`.
- `docs/studies/index.yaml` is a repo-wide selector, not a family-wide one. If
  the request names a checked-in study that is not `active_study_id`, keep the
  selector untouched and pin that study with `--study-dir docs/studies/<study-id>`.
- If the active checked-in study belongs to another family, pin the desired
  study with `--study-dir docs/studies/<study-id>`.
- The worked examples below remain promoter-oriented; Cruncher studies lean
  harder on `routes.md` and `pipeline.yaml` because command grouping and native-agent
  bootstrap are part of the family contract.

### Quick route

Use these surfaces in order:

| Need | Surface | Why |
| --- | --- | --- |
| Where is the live study right now? | `uv run ops progress show usr.data-plane.promoter-study-status --json` | Cheap checked-in snapshot of the active study record. |
| What blocks the next execution step here? | `uv run ops progress show usr.data-plane.promoter-study-preflight --scope next --command-timeout-seconds 30 --json` | Command-level readiness for the next actionable phase on this host. |
| Which owner doc or workspace should I open next? | `docs/studies/<study-id>/routes.md` | Study-owned one-hop handoff for DenseGen, Construct, Infer, LatentDNA, Cluster, and OPAL. |

### Family shortcuts

- Active promoter-study snapshot:
  `uv run ops progress show usr.data-plane.promoter-study-status --json`
- Checked-in Cruncher shortening-study snapshot:
  `uv run ops progress show cruncher.data-plane.cruncher-study-status --study-dir docs/studies/snapback_shortening_effort --json`
- Checked-in Cruncher shortening-study preflight:
  `uv run ops progress show cruncher.data-plane.cruncher-study-preflight --study-dir docs/studies/snapback_shortening_effort --scope next --json`
- Checked-in Cruncher shortening-study route handoff:
  `docs/studies/snapback_shortening_effort/routes.md`
- Repo-local shortening-study skill:
  `.agents/skills/snapback-hairpin-study/SKILL.md`

### Fresh-thread bootstrap

Use this sequence when a new thread starts cold or the repo-local skill is not
visible:

1. Read `docs/studies/index.yaml`.
2. If the request names a checked-in study that is not `active_study_id`, pin
   that study immediately with the family-specific `--study-dir` command and
   treat the registry as discovery only.
3. Run `uv run ops progress show usr.data-plane.promoter-study-status --json`
   only when the question is about the repo-wide active promoter study.
4. Run
   `uv run ops progress show usr.data-plane.promoter-study-preflight --scope next --command-timeout-seconds 30 --json`
   only when the question is blocker or next-run readiness for that same active
   promoter study.
5. Open `docs/studies/<study-id>/routes.md` after the state or blocker question
   is answered and the next owner surface is the real need.

For the checked-in Cruncher shortening example, pin the study explicitly:

- `uv run ops progress show cruncher.data-plane.cruncher-study-status --study-dir docs/studies/snapback_shortening_effort --json`
- `uv run ops progress show cruncher.data-plane.cruncher-study-preflight --study-dir docs/studies/snapback_shortening_effort --scope next --json`
- Open `docs/studies/snapback_shortening_effort/routes.md` for the canonical
  post-probe handoff after the state or blocker answer is settled.

Authority chain: `docs/studies/index.yaml` selects the active study,
the matching `docs/studies/<study-id>/` directory holds the required
`campaign.yaml`, `datasets.yaml`, `status.md`, and `ops.study.yaml`, and may
also carry optional `routes.md` and `pipeline.yaml` surfaces when the study
needs explicit cross-tool handoff navigation plus study-owned runtime context
that should not be reconstructed from generic tool docs.
For a named non-active study, the pinned `--study-dir` path overrides the
registry selector for that lookup without changing the repo-wide active-study
record.

Keep four complementary artifacts for each real study:

- `campaign.yaml`: workflow progress and registered procedure evidence
- `datasets.yaml`: machine-readable registry of affiliated USR datasets and
  sync posture across local and remote locations
- `status.md`: maintainer-facing current datasets, current phase, current row
  counts, downstream posture, and next actions
- `ops.study.yaml`: OPS-facing study contract for lifecycle order, record
  sources, artifacts, execution surfaces, and explicit preflight scope/check
  planning. Declare generic readiness kinds there, such as `path_exists`,
  `dataset_snapshot`, `workspace_layout`, `environment`, `gpu_availability`,
  `command`, `scheduler_queue`, `sequence_view_contract`, and `runbook_plan`,
  then bind them to explicit artifact ids and execution-surface ids.

Keep the code boundary clear: study-family implementation code lives under
`src/dnadesign/studies/`, not under `src/dnadesign/ops/`. OPS reads the
checked-in record and dispatches the provider, but the snapshot and preflight
logic stay with the family package.

When a study already owns concrete execution surfaces, add one optional fifth
artifact:

- `pipeline.yaml`: optional study-owned runtime context for exact Construct,
  Infer, batch, or Notify paths that complement `ops.study.yaml` without
  replacing its lifecycle or preflight authority

When a study already spans several owner surfaces, add one optional sixth
artifact:

- `routes.md`: optional study-owned one-hop route map for DenseGen, Construct,
  Infer, LatentDNA, Cluster, and OPAL handoffs

Use the study record even while the effort is still in source assembly.
An active study does not need to wait for the final feature matrix, but the
record must say whether the current shared feature dataset is materialized or
still pending.

Use `docs/studies/index.yaml` to declare the active study record and the study
family that owns its adapter.

### Declared layout

Keep promoter-study records under:

```text
docs/studies/<study-id>/
  campaign.yaml
  datasets.yaml
  status.md
  ops.study.yaml
  routes.md    # optional but recommended once the study spans multiple owner surfaces
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
- `status.md` is the maintainer-facing note that records current datasets, current
  phase, current row counts, downstream posture, and concise next actions.
- `ops.study.yaml` is the machine-readable OPS contract for lifecycle
  ordering, record sources, artifacts, execution surfaces, repo-scoped
  snapshot posture, and explicit preflight scope/check planning. Snapshot stays
  repo-backed and cheap; preflight is the execution-readiness surface.
- `routes.md`, when present, is the study-owned one-hop handoff page for the
  current DenseGen, Construct, Infer, LatentDNA, Cluster, and OPAL surfaces.
- `pipeline.yaml`, when present, records study-owned runtime context that is
  useful outside the OPS preflight contract, such as exact Construct workspace
  mappings or lane-specific Infer details. Keep OPS-facing lifecycle order,
  declared execution surfaces, and preflight shape in `ops.study.yaml`.
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
9. If the study already spans several owner surfaces, add
   `docs/studies/<study-id>/routes.md` and use it as the study-owned handoff
   page.
10. If the study already has concrete Construct, Infer, or batch surfaces,
   add `docs/studies/<study-id>/pipeline.yaml` and record those exact
   paths there.
11. Refresh evidence with:
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

- Status summary:
  `uv run ops progress show usr.data-plane.promoter-study-status --json`
- Command preflight:
  `uv run ops progress show usr.data-plane.promoter-study-preflight --scope next --command-timeout-seconds 30 --json`
- To pin a non-active study or run from outside the repo checkout, add:
  `--repo-root <repo-root> --study-dir docs/studies/<study-id>`
- The repo-local promoter-study skill lives at `.agents/skills/promoter-study-status/SKILL.md`, but native project-scope skill discovery only picks it up when Codex is launched from this repo root or another path inside this checkout. If the session started elsewhere, use the two `ops progress` commands above directly.
- The repo-local shortening-study skill lives at `.agents/skills/snapback-hairpin-study/SKILL.md`, but native project-scope skill discovery only picks it up when Codex is launched from this repo root or another path inside this checkout. If the session started elsewhere, use the pinned `cruncher-study-status` and `cruncher-study-preflight` commands above directly.
- Read `docs/studies/index.yaml` first.
- If the request names a checked-in study that is not `active_study_id`, pin it
  with `--study-dir docs/studies/<study-id>` instead of treating the registry
  selector as a redirect.
- `active_study_id` must name a study declared under `studies:`.
- The selected study entry must declare `family` and `record_root`.
- The selected study directory must contain `campaign.yaml`, `datasets.yaml`,
  `status.md`, and `ops.study.yaml`.
- `ops.study.yaml` is the OPS-facing source of lifecycle phase order, record
  sources, declared execution surfaces, repo snapshot summary scope, and
  preflight scope/check posture.
- If `routes.md` exists, treat it as the study-owned cross-tool handoff page
  rather than expanding the status note into a workflow encyclopedia.
- If `pipeline.yaml` exists, treat it as supplemental study-owned runtime
  context for exact Construct, Infer, batch, or Notify details that are not
  already declared in `ops.study.yaml`; do not reconstruct those paths from
  generic workspace docs.
- Use `ops progress show usr.data-plane.promoter-study-status` for the cheap
  record-plane snapshot, then escalate to
  `ops progress show usr.data-plane.promoter-study-preflight --scope next --command-timeout-seconds 30 --json`
  when the question is "what should run next?" or "which execution-readiness
  blockers remain right now?" Use the returned ontology fields such as
  `observes_plane`, `summary_scope`, `scope`, `phase_id`, `group_id`, `kind`,
  `surface_id`, and `artifact_id` when you summarize blockers.
- If the registry and directory contents disagree, fail visibly and fix the
  registry before asking agents for live study status.

### Related docs

- [Promoter study status contract](../../src/dnadesign/usr/docs/operations/promoter-study-status-contract.md)
- [Promoter study preflight contract](../../src/dnadesign/usr/docs/operations/promoter-study-preflight.md)
- [stress_ethanol_cipro_growth route map](stress_ethanol_cipro_growth/routes.md)
- [Promoter study index template](../templates/promoter-study-index.yaml)
- [Promoter study datasets template](../templates/promoter-study-datasets.yaml)
- [Promoter study status template](../templates/promoter-study-status.md)
- [Promoter study OPS contract template](../templates/promoter-study-ops.study.yaml)
- [Documentation index](../README.md)

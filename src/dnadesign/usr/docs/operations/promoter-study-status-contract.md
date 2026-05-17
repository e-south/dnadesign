## Promoter Study Status Contract

**Type:** contract
**Plane:** data-plane
**Owner-boundary:** usr
**Entry artifact:** one checked-in promoter-study directory chosen from docs/studies/index.yaml
**Exit artifact:** a read-only snapshot of the current phase, dataset posture, and study-owned handoff surfaces
**Registry-id:** usr.data-plane.promoter-study-status
**Summary:** Read one checked-in promoter-study record and report the current phase, datasets, and handoff surfaces without reconstructing the workflow by hand.
**Execution-kind:** iterative
**Status-kind:** promoter-study-status

**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-17

Use this for the current state of the live DenseGen/manual/wildtype ->
optional Construct -> Infer -> LatentDNA, Cluster, or OPAL study.
It reads the checked-in study record. It does not replace the control-plane
`ops runbook` routes that plan or execute batch work.

### Choose the next surface

Use this page for the cheap snapshot.

| Need | Surface | Why |
| --- | --- | --- |
| Where is the live study right now? | `uv run ops progress show usr.data-plane.promoter-study-status --json` | Cheap checked-in snapshot of study phase, datasets, row counts, and downstream posture. |
| Are sequence products visible at a glance? | `usr.data-plane.promoter-study-status --json` and read `evidence.sequence_view_contract_state` | Summarized product-kind/orientation/pooling contract health and generated sidecar freshness without replacing preflight. |
| Are Infer feature-completion lanes reusable or stale? | [Promoter Study Preflight](promoter-study-preflight.md) | The sequence-view feature-completion inventory scans large Infer sidecars, so it belongs in the deeper preflight surface rather than the record snapshot. |
| What blocks execution on this host? | [Promoter Study Preflight](promoter-study-preflight.md) | Command-level readiness for the next actionable phase. |
| Which owner doc or workspace should I open next? | `docs/studies/<study-id>/routes.md` | Study-owned one-hop handoff for DenseGen, Construct, Infer, LatentDNA, Cluster, and OPAL. |
| Which plots, notebooks, deliverables, or artifact roots are available? | `uv run ops progress show usr.data-plane.promoter-study-status --json` and read `evidence.analysis_surfaces` | One snapshot now exposes DenseGen contract-governed current inventory and freshness, LatentDNA deliverable ids plus artifact roots, and Cluster artifact-layout templates without guessing. |

This page is a router, not the full workflow doc.

Shared runbooks explain the procedure. They do not carry the live dataset ids,
root semantics, sync posture, row targets, completed infer slices, or the next
batch call. Without the study record, they tell you how the workflow works but
not where the study stands today.

### First-thread bootstrap

Use this order when a thread starts cold or the repo-local skill is not
visible:

1. Read [Study records](../../../../../docs/studies/README.md), then
   `docs/studies/index.yaml`.
2. Run `uv run ops progress show usr.data-plane.promoter-study-status --json`.
3. Run
   `uv run ops progress show usr.data-plane.promoter-study-preflight --scope next --json`
   only when the question is blocker or next-run readiness.
4. Open `docs/studies/<study-id>/routes.md` after the state or blocker question
   is answered and the next owner surface is the real need.

### Canonical checked-in location

Keep the study selector at:

```text
docs/studies/index.yaml
```

Keep promoter-study records under:

```text
docs/studies/<study-id>/
  campaign.yaml
  datasets.yaml
  ops.study.yaml
  routes.md    # optional when the study spans owner surfaces
  pipeline.yaml  # optional when the study owns execution surfaces
  status.md
  audits/
```

Read [Study records](../../../../../docs/studies/README.md) first, then
`docs/studies/index.yaml`, when the question is `which study record should I
trust?` rather than `which workflow route applies?`
`index.yaml` selects the active study. The matching study directory holds the
record, and this contract explains how to refresh it.

Fastest read-only summary once the checked-in directory exists:

```bash
# Read the active checked-in promoter-study status summary.
uv run ops progress show usr.data-plane.promoter-study-status
```

If you need to pin a non-active study or you are invoking the command from
outside the repo checkout, add:

```bash
# Pin a specific checked-in study directory when you are outside the repo root.
uv run ops progress show usr.data-plane.promoter-study-status \
  --repo-root <repo-root> \
  --study-dir docs/studies/<study-id>
```

That surface reads the study directory, checks the declared source datasets and
execution surfaces, reads phase order and repo-summary scope from
`ops.study.yaml`, reports the current phase, and highlights the next ready
phase without submitting jobs or mutating USR. Host-local readiness such as GPU
visibility stays advisory here and moves into preflight for hard blockers.
Large feature-completion inventories also stay out of this record snapshot; use
preflight when you need reusable/stale/missing Infer feature counts.
The snapshot keeps each evidence axis separate: source growth, shared handoff
readiness, semantic completeness for critical downstream metadata, and planned
outputs. Operators should not have to infer readiness from one overloaded
`attention` summary.
Snapshot `ok` means the checked-in record is coherent for the current phase; it
does not mean every future phase is finished. Future planned outputs stay in
evidence without forcing `attention`, and an upstream DenseGen row target only
drives `attention` while it is still a live gate for the current study phase.
Once the study has advanced and the canonical shared handoff datasets already
exceed that source threshold, the source target remains visible as historical
context instead of overriding the current phase summary.
For promoter studies that depend on DenseGen design metadata downstream, the
snapshot should also treat missing or stale `densegen__*` metadata on shared
handoff datasets as semantic incompleteness even when the row counts are green.
Describe GPU or notify readiness from the checked-in study contract and live
operator evidence, not from guessed queue names or shared-doc examples. If the
study record pins one lane or environment contract, report that exact selector
or required check from the study-owned surface instead of hard-coding it here.
The snapshot now also calls out stale downstream handoffs when merged anchor or
Construct context datasets trail the upstream DenseGen source even though those
datasets still exist.
OPS resolves the registered provider. The promoter status adapter code that
assembles the snapshot lives under
`src/dnadesign/studies/status_adapters/promoter_status/`.

### Exploratory-analysis route inventory

The snapshot keeps exploratory-analysis discovery in a separate
`evidence.analysis_surfaces` section so record-plane status stays distinct from
tool-local execution. It is still a record-plane summary, not a scheduler-health
surface and not a blanket feature-matrix completeness guarantee.

- `analysis_surfaces.densegen` exposes the checked-in DenseGen workspace,
  public analysis-surface contract reference, generated/operator-visible/
  optional/hidden plot taxonomy, authoritative current inventory from
  `outputs/plots/current_inventory.json`, optional historical ledger from
  `outputs/plots/artifact_ledger.json`, freshness state, degraded-state
  diagnostics, and the generated notebook path when present.
- DenseGen may still keep `outputs/plots/plot_manifest.json` as a compatibility
  ledger mirror, but it is not the authoritative current snapshot for operator
  status.
- `analysis_surfaces.latentdna` exposes the checked-in workspace, notebook id,
  plot ids, deliverable ids, any declared export ids, and the
  `outputs/<artifact-kind>/<artifact-id>/manifest.json` path
  contract plus `outputs/notebooks/<notebook-id>/notebook.py` for the generated
  workspace notebook.
- `analysis_surfaces.cluster` exposes the study entry artifact, workflow doc,
  one workspace example, and the explicit results-root layout template because
  the current study does not yet own a checked-in cluster results root.

This surface is intentionally route-oriented:

- it reports explicit ids, commands, and path conventions
- it does not fabricate missing LatentDNA or Cluster artifacts
- it keeps blocker and host-readiness questions in preflight instead of mixing
  them into exploratory-analysis discovery

Need command-level blockers for the same study? Open
[Promoter Study Preflight](promoter-study-preflight.md):

```bash
# Escalate from the cheap snapshot to the command-level preflight surface.
uv run ops progress show usr.data-plane.promoter-study-preflight --scope next --json
```

### Keep these artifacts for every real study

1. A campaign manifest that names the real artifacts.
2. A machine-readable dataset registry that names the affiliated USR datasets,
   onboarding mode, root semantics, and sync posture.
3. A status note that answers the human questions the manifest and
   dataset registry do not encode.
4. `ops.study.yaml`, which tells OPS the explicit study lifecycle, record
   sources, execution surfaces, snapshot scope, and next-scope preflight
   posture without hard-coding workflow taxonomy in core code.
5. When downstream execution surfaces exist, an optional `routes.md` that points to the current DenseGen, Construct, Infer,
   LatentDNA, Cluster, and OPAL owner surfaces without bloating `status.md`.
6. When downstream execution surfaces exist, an optional `pipeline.yaml` that
   names the exact Construct, Infer, and batch surfaces plus any minimal
   runtime mappings the live study still needs. Infer Notify profile paths
   should derive from the checked-in lane configs instead of being duplicated
   there.

Recommended bootstrap for the manifest:

```bash
# Bootstrap the promoter-study registry only if it is missing.
cp docs/templates/promoter-study-index.yaml docs/studies/index.yaml
# Create the checked-in directory for one real promoter study.
mkdir -p docs/studies/<study-id>
# Emit a related-procedure campaign skeleton for the checked-in study record.
uv run ops progress scaffold --related-to usr.data-plane.promoter-feature-matrix --repo-root <repo-root> > docs/studies/<study-id>/campaign.yaml
# Copy the machine-readable affiliated-dataset registry template.
cp docs/templates/promoter-study-datasets.yaml docs/studies/<study-id>/datasets.yaml
# Copy the maintained status-note template into the same study directory.
cp docs/templates/promoter-study-status.md docs/studies/<study-id>/status.md
# Copy the OPS-facing study contract template into the same study directory.
cp docs/templates/promoter-study-ops.study.yaml docs/studies/<study-id>/ops.study.yaml
# Create the audit directory referenced by sync-enabled dataset entries.
mkdir -p docs/studies/<study-id>/audits
```

If the registry already exists, edit it in place instead of replacing it.
Then replace the placeholders with the real `usr_root`, `dataset`,
`cluster_results_root`, and `opal_config` values for the current study.
Delete steps that are not part of the active branch of work. If the study is
still only a source-growth effort, omit `pipeline.yaml` until there is a real
downstream Construct or Infer surface to record.

### Dataset registry, status template, and discovery rules

Keep `datasets.yaml` next to the manifest. Copy the template from
[docs/templates/promoter-study-datasets.yaml](../../../../../docs/templates/promoter-study-datasets.yaml).

Each dataset entry should declare:

- `role`: the study role such as anchor source, wildtype/manual source,
  construct context, or feature matrix
- `dataset` plus `usr_root`: the explicit local USR location
- `root_kind`: `shared`, `workspace_local_export`, or `external_usr`
- `status`: whether the dataset is already `present` or still `planned`
- `onboard_mode`: `existing_local`, `existing_remote`, `existing_both`, or
  `create_new`
- `authority`: whether local, remote, or both sides are the current source of
  truth
- `notes`: the operator-facing explanation for why this location exists
- `sync`: whether sync is enabled, which remote profile is used, the default
  direction, the audit JSON path, whether `strict_bootstrap_id` must stay on,
  which remote root kind applies, and whether explicit `remote_path` mapping is
  needed

Semantics:

- `workspace_local_export` means the dataset is owned by a tool workspace such
  as `outputs/usr_datasets/` and is not automatically the cross-tool study
  source of truth
- `shared` means the dataset is the shared study copy intended for
  downstream status, infer, cluster, or OPAL routes
- `external_usr` means the dataset is still a USR root, but lives outside the
  repo-owned shared path and must stay explicit in the study record

Keep `status.md` next to the manifest. Read it first for `where are we now?`
Copy the template from
[docs/templates/promoter-study-status.md](../../../../../docs/templates/promoter-study-status.md).
Keep it factual and short: current datasets, current phase, current row counts,
current downstream posture, and concise next actions.
When the study spans several owner surfaces, add `routes.md` next to the status
note and use it as the one-hop handoff page.

Discovery rules:

- Read `docs/studies/index.yaml` first.
- `active_study_id` must name a study declared under `studies:`.
- The selected study entry must declare `record_root`; the study's
  `ops.study.yaml` must declare `ops_surfaces.status_kind` and
  `ops_surfaces.preflight_kind`.
- The corresponding study directory must contain `campaign.yaml`,
  `datasets.yaml`, `status.md`, and `ops.study.yaml`.
- If `routes.md` exists in the study directory, treat it as the study-owned
  one-hop handoff page for DenseGen, Construct, Infer, LatentDNA, Cluster, and
  OPAL instead of expanding `status.md` into a workflow encyclopedia.
- `ops.study.yaml` is the OPS-facing source of lifecycle phase order, record
  sources, execution surfaces, repo snapshot scope, and next-scope routing.
  Keep it checked in with the study record.
- If `pipeline.yaml` exists in the study directory, treat it as the canonical
  execution-map surface for exact Construct, Infer, and batch paths plus any
  minimal runtime mappings still needed by the live study. Infer Notify profile
  paths derive from the checked-in lane configs.
- Snapshot output keeps `ops.study.yaml` labels authoritative under
  `execution_surfaces`. If `pipeline.yaml` expands convenience aliases beyond
  that contract, OPS reports those under `derived_execution_surfaces` instead
  of mixing them into the canonical surface list.
- If the registry and study directory contents disagree, fail visibly and fix
  the registry before asking for live status.

### Refresh loop

1. Refresh the campaign summary:

```bash
# Summarize every tracked step in the study manifest.
uv run ops progress campaign --repo-root <repo-root> --manifest docs/studies/<study-id>/campaign.yaml
```

2. Refresh the current consolidated feature-dataset summary only when a
   canonical consolidated feature dataset already exists:

```bash
# Summarize the live feature dataset from explicit USR artifacts.
uv run ops progress show usr.data-plane.promoter-feature-matrix --repo-root <repo-root> --usr-root <usr-root> --dataset <feature-dataset>
```

If `status.md` still marks the canonical consolidated feature dataset as
`planned` or `n/a`, skip this step and report that the study is still in
source/handoff mode.

### Minimum evidence by question

| Question | Primary surface | Minimum evidence | Fail visibly when |
| --- | --- | --- | --- |
| Where is the live study right now? | `usr.data-plane.promoter-study-status --json` | study id, current phase, dataset ids, row counts, downstream posture, next surface | selector fields or required record files are missing |
| What blocks execution on this host? | `usr.data-plane.promoter-study-preflight --scope next --json` | `scope`, `phase_id`, `check_group`, `kind`, `surface_id`, `artifact_id` | `ops.study.yaml` or declared execution surfaces are missing |
| Which owner doc or workspace should I open next? | `docs/studies/<study-id>/routes.md` | owner tool, entry artifact, primary doc or workspace, first command | the study spans owner surfaces but no route map is checked in |
| Which dataset sync posture is current? | `datasets.yaml` plus `usr.data-plane.hpc-sync` | dataset id, remote profile, audit JSON path, explicit drift summary | sync-enabled dataset entries or audit evidence are missing |
| Which exploratory-analysis artifacts exist or are declared? | `usr.data-plane.promoter-study-status --json` `analysis_surfaces` plus `docs/studies/<study-id>/routes.md` | DenseGen operator-visible/current inventory plus freshness and degradation state, LatentDNA deliverable/export ids plus artifact roots, Cluster results-layout template | the study record omits the owning workspace/doc path or the tool-local contract is missing |

3. Validate the dataset and inspect lineage plus one explicit infer column:

```bash
# Confirm the feature dataset still satisfies the active USR registry.
uv run usr --root <usr-root> validate <feature-dataset> --strict
# Inspect source labels, construct lineage, and one explicit infer column together.
uv run usr --root <usr-root> head <feature-dataset> -n 5 --columns id,usr_label__primary,construct__input_dataset,construct__template_id,<one explicit infer__... column>
```

4. Refresh any pending infer slice before writing:

```bash
# Validate the next infer config before any write-back.
uv run infer validate config --config <infer-config>
# Dry-run the same infer slice against the real study dataset.
uv run infer run --config <infer-config> --dry-run
```

5. Refresh watcher status when delivery matters:

```bash
# Resolve the infer-managed events path from the real study config.
uv run notify setup resolve-events --tool infer --config <infer-config> --json
# Dry-run watcher delivery against the same study dataset event stream.
uv run notify usr-events watch --events <usr-root>/<feature-dataset>/.events.log --dry-run --no-advance-cursor-on-dry-run
```

6. Refresh affiliated-dataset sync posture from `datasets.yaml`:

```bash
# Inspect the local dataset named by one datasets.yaml entry.
uv run usr --root <usr-root> info <dataset-id> --format json
# Capture local-vs-remote drift for one sync-enabled datasets.yaml entry.
uv run usr --root <usr-root> diff <dataset-id> <remote-name> --audit-json-out docs/studies/<study-id>/audits/<dataset-id>--<remote-name>-diff.json
# Summarize that same sync audit through the registered status view.
uv run ops progress show usr.data-plane.hpc-sync --sync-audit-json docs/studies/<study-id>/audits/<dataset-id>--<remote-name>-diff.json
```

If a dataset starts remote-only, keep `strict_bootstrap_id: true` and use an
explicit dataset id on the first `usr pull` so bootstrap never relies on local
name guessing. When the producer writes to a workspace-local export root on
SCC, declare that root explicitly in `datasets.yaml` instead of pretending it
is the shared USR root.

### What each artifact answers

From `campaign.yaml` plus `ops progress ...`:

- which registered procedure applies
- the current dataset row count
- whether infer-derived namespaces exist
- whether cluster or OPAL steps have explicit artifact roots

From `datasets.yaml`:

- which USR datasets are affiliated with the study across source, handoff, and
  downstream stages
- whether each declared location is shared storage or only a
  workspace-local export
- whether each dataset is already present locally, remotely, or both
- which remote profile and sync direction apply for each dataset
- which sync audit JSON file should be refreshed and summarized
- whether explicit path-mode sync is required instead of dataset-id sync

From `status.md`:

- current source and handoff dataset ids
- current phase
- current row counts
- current downstream posture
- concise next actions

From `ops.study.yaml`:

- the study phase order that OPS should treat as canonical
- whether the cheap snapshot is repo-scoped or broader
- which phase groups belong in `--scope next` preflight rather than `--scope full`

From `pipeline.yaml` when present:

- the canonical Construct workspace/config paths for the live study
- the canonical Infer workspace/config paths for the live study
- which Notify-backed batch presets belong to the study
- the expected phase order from source assembly through Infer write-back
- whether anchor-only and template-backed Infer lanes are modeled as one plane
  or as explicit separate dataset planes
- study-bound downstream structural bindings such as LatentDNA workspace,
  Cluster results root, or OPAL config

### Failure and rollback reminders

- DenseGen accumulation should use `--mode auto` or `--resume --extend-quota`,
  not implicit fresh resets when existing rows should survive.
- Infer write-back is reversible at the namespace level with `infer prune` or
  `usr maintenance overlay-remove`.
- USR overlay parts and sync contracts are explicit. If `usr pull` or
  `usr push` is interrupted, rerun the command instead of copying files by
  hand.
- Strict `pull` or `push` can still spend time proving `_derived` parity before
  concluding `NO-OP`. Use `usr diff` first for a quick drift preview, then run
  `pull` or `push` when you want transfer or final fidelity verification.
- Remote-only bootstrap should stay explicit: prefer `--strict-bootstrap-id`
  for the first pull when `datasets.yaml` marks a dataset as `existing_remote`.
- Notify recovery depends on workspace-scoped `cursor` and `spool` paths. Keep
  those artifacts specific to one run family.

### Related docs

- [Study records](../../../../../docs/studies/README.md)
- [Study records index](../../../../../docs/studies/README.md)
- [Promoter study index template](../../../../../docs/templates/promoter-study-index.yaml)
- [Promoter study datasets template](../../../../../docs/templates/promoter-study-datasets.yaml)
- [Promoter study status template](../../../../../docs/templates/promoter-study-status.md)
- [Promoter study Evo2 workflow journey](promoter-evo2-journey.md)
- [Promoter characterization feature matrix](promoter-characterization-feature-matrix.md)
- [Chained DenseGen and Infer sync runbook](chained-densegen-infer-sync-runbook.md)
- [Docs index](../../../../../docs/README.md)
- [Runbook catalog](../../../../../docs/runbooks/README.md)
- [Ops orchestration index](../../../../../docs/operations/README.md)
- [Notify USR events operator guide](../../../../../docs/notify/usr-events.md)

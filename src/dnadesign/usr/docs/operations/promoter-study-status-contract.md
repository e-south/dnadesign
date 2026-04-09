## Promoter Study Status Contract

**Type:** contract
**Plane:** data-plane
**Owner-boundary:** usr
**Entry artifact:** one checked-in promoter-study directory selected through docs/studies/index.yaml
**Exit artifact:** one read-only summary of dataset presence, phase posture, and study-owned execution surfaces
**Registry-id:** usr.data-plane.promoter-study-status
**Summary:** Read one checked-in promoter-study record and summarize dataset, phase, and execution-surface readiness without reconstructing the workflow by hand.
**Execution-kind:** iterative
**Status-kind:** promoter-study-status

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-25

Use this contract when the question is not only "which runbook applies?" but
"what is the current status of the actual DenseGen/manual/wildtype ->
optional Construct -> Infer -> Cluster or OPAL study?"
This is an observation-plane surface: it reads the checked-in study record and
does not replace the control-plane `ops runbook` routes that plan or execute
batch work.

The shared runbooks explain procedure. They do not carry your real dataset ids,
shared-root versus workspace-export semantics, local-vs-remote sync posture,
row targets, completed infer slices, or the next batch call. Without a
maintained study record, the docs explain how the workflow works but not where
the live study stands.

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
  pipeline.yaml
  status.md
  audits/
```

Read [Study records](../../../../../docs/studies/README.md) first, then
`docs/studies/index.yaml`, when the question is "which study record
should I trust?" rather than "which workflow route applies?"
`index.yaml` selects the active study, the matching study directory holds the
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

That surface reads the checked-in study directory, validates that the declared
source datasets and study-owned execution surfaces exist, reads phase order and
repo-summary scope from `ops.study.yaml`, reports the current phase from the
checked-in record, and highlights the next ready phase without submitting jobs
or mutating USR. Host-local readiness such as GPU visibility remains advisory
here and moves into preflight for hard blockers.
For `evo2_20b`, describe GPU readiness from the checked-in runbook contract,
not from a guessed BU queue name. If the study record or live operator
evidence says the current `.venv` is GPU-family-pinned, report the exact
selector as part of readiness. For the active stress-ethanol-cipro study on
BU SCC, the current Blackwell-family lane is `gpu_t=RTXP6000` with
`gpu_capability=12.0`; a generic `gpu_capability >= 9.0` request is a looser
model-fit floor and can still land on H200.
The snapshot now also calls out stale downstream handoffs when merged anchor or
Construct context datasets trail the upstream DenseGen source even though those
datasets still exist.
The implementation boundary is explicit: OPS resolves the registered provider,
and the stress-promoter family code that assembles the snapshot lives under
`src/dnadesign/studies/families/promoter/`.

When you need deeper, command-level blockers across the same checked-in study,
continue to [Promoter Study Preflight](promoter-study-preflight.md):

```bash
# Escalate from the cheap snapshot to the command-level preflight surface.
uv run ops progress show usr.data-plane.promoter-study-preflight --json
```

### Keep these artifacts for every real study

1. A checked-in campaign manifest that names the real artifacts.
2. A machine-readable dataset registry that names the affiliated USR datasets,
   onboarding mode, root semantics, and sync posture.
3. A checked-in status note that answers the human questions the manifest and
   dataset registry do not encode.
4. A checked-in `ops.study.yaml` that tells OPS the study-family lifecycle,
   record sources, execution surfaces, snapshot scope, and next-scope
   preflight posture without hard-coding workflow taxonomy in core code.
5. When the study has real downstream execution surfaces, an optional
   `pipeline.yaml` that names the exact Construct, Infer, and batch surfaces
   plus any minimal runtime mappings the live study still needs. Infer Notify
   profile paths should derive from the checked-in lane configs instead of
   being duplicated there.

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

Keep a filled `datasets.yaml` next to the manifest. Copy the template from
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

Keep the semantics explicit:

- `workspace_local_export` means the dataset is owned by a tool workspace such
  as `outputs/usr_datasets/` and is not automatically the cross-tool study
  source of truth
- `shared` means the dataset is the shared study copy intended for
  downstream status, infer, cluster, or OPAL routes
- `external_usr` means the dataset is still a USR root, but lives outside the
  repo-owned shared path and must stay explicit in the study record

Keep a filled `status.md` next to the manifest. Read it first when the question
is "where are we now?" or "what should run next?" Copy the template from
[docs/templates/promoter-study-status.md](../../../../../docs/templates/promoter-study-status.md).

Discovery rules:

- Read `docs/studies/index.yaml` first.
- `active_study_id` must name a study declared under `studies:`.
- The selected study entry must declare `family` and `record_root`.
- The corresponding study directory must contain `campaign.yaml`,
  `datasets.yaml`, `status.md`, and `ops.study.yaml`.
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
# Summarize every tracked step in the checked-in study manifest.
uv run ops progress campaign --repo-root <repo-root> --manifest docs/studies/<study-id>/campaign.yaml
```

2. Refresh the current feature-dataset summary only when a shared feature
   dataset already exists:

```bash
# Summarize the live feature dataset from explicit USR artifacts.
uv run ops progress show usr.data-plane.promoter-feature-matrix --repo-root <repo-root> --usr-root <usr-root> --dataset <feature-dataset>
```

If `status.md` still marks the shared feature dataset as `n/a`, skip this
step and report that the study is still in source-assembly mode.

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

- which DenseGen dataset is being grown toward the current row target
- which wildtype or manual dataset is merged into the study
- whether construct expansion is required or optional
- which infer slices are already written versus only preflighted
- the next concrete batch call to run

From `ops.study.yaml`:

- the study-family phase order that OPS should treat as canonical
- whether the cheap snapshot is repo-scoped or broader
- which phase groups belong in `--scope next` preflight rather than `--scope full`

From `pipeline.yaml` when present:

- the canonical Construct workspace/config paths for the live study
- the canonical Infer workspace/config paths for the live study
- which Notify-backed batch presets belong to the study
- the expected phase order from source assembly through Infer write-back
- whether anchor-only and template-backed Infer lanes are modeled as one plane
  or as explicit separate dataset planes

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

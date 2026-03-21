## Promoter Study Status Contract

**Type:** contract
**Plane:** data-plane
**Owner-boundary:** usr
**Entry artifact:** one real promoter-study effort that needs agent-readable status
**Exit artifact:** one checked-in study record under `docs/studies/promoter/<study-id>/`

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-21

Use this contract when the question is not only "which runbook applies?" but
"what is the current status of our actual DenseGen/manual/wildtype ->
optional Construct -> Infer -> Cluster or OPAL study?"

The shared runbooks explain procedure. They do not know your real dataset ids,
local-vs-remote sync posture, row targets, completed infer slices, or the next
batch call. Without a maintained study-status record, a naive agent can
reconstruct mechanics but cannot answer current-study questions honestly.

### Canonical checked-in location

Keep promoter-study records under:

```text
docs/studies/promoter/<study-id>/
  campaign.yaml
  datasets.yaml
  status.md
  audits/
```

Read [Study records](../../../../../docs/studies/README.md) first when the
question is "which study record should I trust?" rather than "which workflow
route applies?"

### Keep these three artifacts for every real study

1. A checked-in campaign manifest that names the real artifacts.
2. A machine-readable dataset registry that names the affiliated USR datasets,
   onboarding mode, and sync posture.
3. A checked-in status note that answers the human questions the manifest and
   dataset registry do not encode.

Recommended bootstrap for the manifest:

```bash
# Create the canonical checked-in directory for one real promoter study.
mkdir -p docs/studies/promoter/<study-id>
# Emit a related-procedure campaign skeleton for the checked-in study record.
uv run ops progress scaffold --related-to usr.data-plane.promoter-feature-matrix --repo-root <repo-root> > docs/studies/promoter/<study-id>/campaign.yaml
# Copy the machine-readable affiliated-dataset registry template.
cp docs/templates/promoter-study-datasets.yaml docs/studies/promoter/<study-id>/datasets.yaml
# Copy the maintained status-note template into the same study directory.
cp docs/templates/promoter-study-status.md docs/studies/promoter/<study-id>/status.md
# Create the audit directory referenced by sync-enabled dataset entries.
mkdir -p docs/studies/promoter/<study-id>/audits
```

Then replace the placeholders with the real `usr_root`, `dataset`,
`cluster_results_root`, and `opal_config` values for the current study.
Delete steps that are not part of the active branch of work.

### Dataset registry, status template, and discovery rules

Keep a filled `datasets.yaml` next to the manifest. Copy the template from
[docs/templates/promoter-study-datasets.yaml](../../../../../docs/templates/promoter-study-datasets.yaml).

Each dataset entry should declare:

- `role`: the study role such as anchor source, wildtype/manual source,
  construct context, or feature matrix
- `dataset` plus `usr_root`: the explicit local USR location
- `onboard_mode`: `existing_local`, `existing_remote`, `existing_both`, or
  `create_new`
- `authority`: whether local, remote, or both sides are the current source of
  truth
- `sync`: whether sync is enabled, which remote profile is used, the default
  direction, the audit JSON path, whether `strict_bootstrap_id` must stay on,
  and whether explicit `remote_path` mapping is needed

Keep a filled `status.md` next to the manifest. This is the document a naive
agent should read first when asked for "where are we now?" or "what should run
next?" Copy the template from
[docs/templates/promoter-study-status.md](../../../../../docs/templates/promoter-study-status.md).

Discovery rules:

- If exactly one `docs/studies/promoter/<study-id>/` directory contains both
  `campaign.yaml`, `datasets.yaml`, and `status.md`, a naive agent may treat it
  as the active study.
- If more than one candidate exists, the user or a higher-level record must
  identify the intended study.
- If no study record exists, answer that the live study record is missing
  instead of inferring current status from generic runbooks.

### Refresh loop

1. Refresh the campaign summary:

```bash
# Summarize every tracked step in the checked-in study manifest.
uv run ops progress campaign --repo-root <repo-root> --manifest docs/studies/promoter/<study-id>/campaign.yaml
```

2. Refresh the current feature-dataset summary:

```bash
# Summarize the live feature dataset from explicit USR artifacts.
uv run ops progress show usr.data-plane.promoter-feature-matrix --repo-root <repo-root> --usr-root <usr-root> --dataset <feature-dataset>
```

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
uv run usr --root <usr-root> diff <dataset-id> <remote-name> --audit-json-out docs/studies/promoter/<study-id>/audits/<dataset-id>--<remote-name>-diff.json
# Summarize that same sync audit through the registered progress surface.
uv run ops progress show usr.data-plane.hpc-sync --sync-audit-json docs/studies/promoter/<study-id>/audits/<dataset-id>--<remote-name>-diff.json
```

If a dataset starts remote-only, keep `strict_bootstrap_id: true` and use an
explicit dataset id on the first `usr pull` so bootstrap never relies on local
name guessing.

### What each artifact answers

From `campaign.yaml` plus `ops progress ...`:

- which registered procedure applies
- the current dataset row count
- whether infer-derived namespaces exist
- whether cluster or OPAL steps have explicit artifact roots

From `datasets.yaml`:

- which USR datasets are affiliated with the study across source, handoff, and
  downstream stages
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

### Failure and rollback reminders

- DenseGen accumulation should use `--mode auto` or `--resume --extend-quota`,
  not implicit fresh resets when existing rows should survive.
- Infer write-back is reversible at the namespace level with `infer prune` or
  `usr maintenance overlay-remove`.
- USR overlay parts and sync contracts are explicit. If `usr pull` or
  `usr push` is interrupted, rerun the command instead of copying files by
  hand.
- Remote-only bootstrap should stay explicit: prefer `--strict-bootstrap-id`
  for the first pull when `datasets.yaml` marks a dataset as `existing_remote`.
- Notify recovery depends on workspace-scoped `cursor` and `spool` paths. Keep
  those artifacts specific to one run family.

### Related docs

- [Study records](../../../../../docs/studies/README.md)
- [Promoter study datasets template](../../../../../docs/templates/promoter-study-datasets.yaml)
- [Promoter study status template](../../../../../docs/templates/promoter-study-status.md)
- [Promoter study Evo2 workflow journey](promoter-evo2-journey.md)
- [Promoter characterization feature matrix](promoter-characterization-feature-matrix.md)
- [Chained DenseGen and Infer sync runbook](chained-densegen-infer-sync-runbook.md)
- [Docs index](../../../../../docs/README.md)
- [Runbook catalog](../../../../../docs/runbooks/README.md)
- [Ops orchestration index](../../../../../docs/operations/README.md)
- [Notify USR events operator guide](../../../../../docs/notify/usr-events.md)

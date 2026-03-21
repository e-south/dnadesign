## Study Records

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-21

Use this index when the question is "where is the real study right now?" rather
than "which generic workflow should I use?"

Study records are checked-in status artifacts for one live effort. They are not
runbooks, and they are not generated outputs.

Keep three complementary artifacts for each real study:

- `campaign.yaml`: workflow progress and registered procedure evidence
- `datasets.yaml`: machine-readable registry of affiliated USR datasets and
  sync posture across local and remote locations
- `status.md`: human-readable current state, row targets, and next actions

Use the study record even when the effort is still in the source-assembly phase.
An active study does not need to wait until the final feature matrix already
exists, but the record must say explicitly whether the current shared feature
dataset is materialized or still pending.

Use [Promoter study registry](promoter/README.md) plus
`docs/studies/promoter/index.yaml` to declare whether one live study is active.

### Declared layout

Keep promoter-study records under:

```text
docs/studies/promoter/<study-id>/
  campaign.yaml
  datasets.yaml
  status.md
  audits/
```

and keep the study selector at:

```text
docs/studies/promoter/index.yaml
```

- `campaign.yaml` is the explicit multi-step manifest generated from
  `uv run ops progress scaffold ...` and then filled with the real artifact
  paths.
- `datasets.yaml` declares which USR datasets belong to the study, whether each
  location is a shared USR root or a workspace-local export root, how
  it should be onboarded, and how it syncs to remotes such as `cluster` or a
  study-specific workspace-export remote.
- `status.md` is the human-readable note that records row targets, source
  datasets, infer slice status, rollback paths, and next actions.
- `audits/` stores machine-readable sync audit JSON files referenced from
  `datasets.yaml`.

### Create or refresh a promoter-study record

1. Create the study directory:
   `mkdir -p docs/studies/promoter/<study-id>`
2. If `docs/studies/promoter/index.yaml` is missing, bootstrap it once:
   `cp docs/templates/promoter-study-index.yaml docs/studies/promoter/index.yaml`
3. Generate the manifest:
   `uv run ops progress scaffold --related-to usr.data-plane.promoter-feature-matrix --repo-root <repo-root> > docs/studies/promoter/<study-id>/campaign.yaml`
4. Copy the dataset registry template:
   `cp docs/templates/promoter-study-datasets.yaml docs/studies/promoter/<study-id>/datasets.yaml`
5. Copy the status-note template:
   `cp docs/templates/promoter-study-status.md docs/studies/promoter/<study-id>/status.md`
6. Create the audit directory:
   `mkdir -p docs/studies/promoter/<study-id>/audits`
7. Edit the checked-in `index.yaml` plus the new `campaign.yaml`, `datasets.yaml`, and `status.md` so they point at the real study ids, paths, and commands.
8. Refresh evidence with:
   `uv run ops progress campaign --repo-root <repo-root> --manifest docs/studies/promoter/<study-id>/campaign.yaml`

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
  --audit-json-out docs/studies/promoter/<study-id>/audits/<dataset-id>--<remote-name>-diff.json
# Summarize the same sync audit through the registered status surface.
uv run ops progress show usr.data-plane.hpc-sync \
  --sync-audit-json docs/studies/promoter/<study-id>/audits/<dataset-id>--<remote-name>-diff.json
```

If a dataset is being onboarded from a remote-only starting point, keep
`strict_bootstrap_id: true` in `datasets.yaml` and use an explicit dataset id
for the first pull rather than relying on local name guessing.

### Agent discovery rules

- Read `docs/studies/promoter/index.yaml` first.
- If `active_study: null`, agents should say the live study record is missing
  and route the user to the promoter-study status contract instead of guessing.
- If `active_study` names a study, that same id must appear under `studies:`
  and its directory must contain `campaign.yaml`, `datasets.yaml`, and
  `status.md`.
- If the registry and directory contents disagree, fail visibly and fix the
  registry before asking agents for live study status.

### Related docs

- [Promoter study status contract](../../src/dnadesign/usr/docs/operations/promoter-study-status-contract.md)
- [Promoter study registry](promoter/README.md)
- [Promoter study index template](../templates/promoter-study-index.yaml)
- [Promoter study datasets template](../templates/promoter-study-datasets.yaml)
- [Promoter study status template](../templates/promoter-study-status.md)
- [Documentation index](../README.md)

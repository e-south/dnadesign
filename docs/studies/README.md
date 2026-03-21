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

### Canonical layout

Keep promoter-study records under:

```text
docs/studies/promoter/<study-id>/
  campaign.yaml
  datasets.yaml
  status.md
  audits/
```

- `campaign.yaml` is the explicit multi-step manifest generated from
  `uv run ops progress scaffold ...` and then filled with the real artifact
  paths.
- `datasets.yaml` declares which USR datasets belong to the study, how they
  should be onboarded, and how they sync to remotes such as `bu-scc`.
- `status.md` is the human-readable note that records row targets, source
  datasets, infer slice status, rollback paths, and next actions.
- `audits/` stores machine-readable sync audit JSON files referenced from
  `datasets.yaml`.

### Create or refresh a promoter-study record

1. Create the study directory:
   `mkdir -p docs/studies/promoter/<study-id>`
2. Generate the manifest:
   `uv run ops progress scaffold --related-to usr.data-plane.promoter-feature-matrix --repo-root <repo-root> > docs/studies/promoter/<study-id>/campaign.yaml`
3. Copy the dataset registry template:
   `cp docs/templates/promoter-study-datasets.yaml docs/studies/promoter/<study-id>/datasets.yaml`
4. Copy the status-note template:
   `cp docs/templates/promoter-study-status.md docs/studies/promoter/<study-id>/status.md`
5. Create the audit directory:
   `mkdir -p docs/studies/promoter/<study-id>/audits`
6. Replace placeholders with the real study ids, paths, and commands.
7. Refresh evidence with:
   `uv run ops progress campaign --repo-root <repo-root> --manifest docs/studies/promoter/<study-id>/campaign.yaml`

### Dataset registry contract

Use `datasets.yaml` to keep one DRY declaration of study-affiliated datasets.
Every dataset entry should answer:

- which role the dataset plays in the study
- which `usr_root` and dataset id own that artifact locally
- whether the dataset is being onboarded as `existing_local`,
  `existing_remote`, `existing_both`, or `create_new`
- whether the source of truth is `local`, `remote`, or `shared`
- whether sync is required, which remote profile is used, and where the sync
  audit JSON should be written

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

- If exactly one `docs/studies/promoter/<study-id>/` directory contains both
  `campaign.yaml`, `datasets.yaml`, and `status.md`, naive agents may treat it
  as the active study.
- If more than one candidate exists, the user or a higher-level record must
  specify which study is in scope.
- If no such record exists, agents should say the live study record is missing
  and route the user to the promoter-study status contract instead of guessing.

### Related docs

- [Promoter study status contract](../../src/dnadesign/usr/docs/operations/promoter-study-status-contract.md)
- [Promoter study datasets template](../templates/promoter-study-datasets.yaml)
- [Promoter study status template](../templates/promoter-study-status.md)
- [Documentation index](../README.md)

## Study Record Authoring

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-14

Checked-in study records keep long-running project context discoverable without
turning the top-level [Study Records](README.md) index into a status/preflight
ladder.

### Required record shape

```text
docs/studies/<study-id>/
  campaign.yaml
  datasets.yaml
  status.md
  ops.study.yaml
  routes.md      # optional, preferred once the study spans owner surfaces
  pipeline.yaml  # optional, for exact command groups/runtime context
  audits/
```

- `campaign.yaml` is the explicit multi-step manifest generated from
  `uv run ops progress scaffold ...` and then filled with real artifact paths.
- `datasets.yaml` declares which USR datasets belong to the study and how they
  sync across local and remote roots.
- `status.md` records current state and concise next actions.
- `ops.study.yaml` is the OPS-facing contract for lifecycle ordering or track
  maps, record sources, artifacts, execution surfaces, and preflight planning.
- `routes.md`, when present, is the study-owned one-hop handoff page.
- `pipeline.yaml`, when present, records exact command groups or runtime
  context that should not be reconstructed from generic tool docs.
- `audits/` stores machine-readable sync audit JSON files referenced from
  `datasets.yaml`.

### Promoter-study bootstrap

Use this sequence for promoter-style studies that need the full
status/preflight contract:

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
8. Edit the checked-in `index.yaml`, `campaign.yaml`, `datasets.yaml`,
   `status.md`, and `ops.study.yaml` so they point at real ids, paths, and
   commands.
9. Add `routes.md` when the study spans several owner surfaces.
10. Add `pipeline.yaml` when the study has exact Construct, Infer, batch, or
    Notify command groups.
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
- whether the source of truth is `local`, `remote`, or `shared`
- whether sync is required, which remote profile is used, and where the sync
  audit JSON should be written

Terminology:

- `workspace_local_export`: a producer-owned USR-shaped dataset rooted under a
  tool workspace rather than the shared study root
- `shared`: the cross-tool study copy rooted under an explicit shared USR root
  such as `src/dnadesign/usr/datasets`
- `external_usr`: an operator-owned but non-repo USR root that is still
  explicit in the study record

For a sync-enabled dataset, refresh evidence with explicit commands such as:

```bash
uv run usr --root <usr-root> info <dataset-id> --format json
uv run usr --root <usr-root> diff <dataset-id> <remote-name> \
  --audit-json-out docs/studies/<study-id>/audits/<dataset-id>--<remote-name>-diff.json
uv run ops progress show usr.data-plane.hpc-sync \
  --sync-audit-json docs/studies/<study-id>/audits/<dataset-id>--<remote-name>-diff.json
```

If a dataset is being onboarded from a remote-only starting point, keep
`strict_bootstrap_id: true` in `datasets.yaml` and use an explicit dataset id
for the first pull rather than relying on local name guessing.

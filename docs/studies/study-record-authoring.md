## Study Record Authoring

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-14

Checked-in study records keep long-running project context discoverable without
turning the top-level [Study Records](README.md) index into a status/preflight
ladder.

### Required record shape

```text
docs/studies/<study-id>/
  campaign.yaml  # optional explicit progress manifest
  datasets.yaml
  status.md
  ops.study.yaml
  routes.md      # optional, preferred once the study spans owner surfaces
  pipeline.yaml  # optional, for exact command groups/runtime context
  audits/
```

- `campaign.yaml` is an optional explicit multi-step manifest generated from
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

### Study Bootstrap

Use this sequence for studies that need the shared checked-in record shape.
Only add study status/preflight providers when that specific study owns the
implementation.

1. Create the study directory:
   `mkdir -p docs/studies/<study-id>`
2. If `docs/studies/index.yaml` is missing, author it from the active study id
   and explicit `record_root` entries; do not copy a family-specific template.
3. Generate a campaign manifest only when a registered progress route is useful:
   `uv run ops progress scaffold --related-to usr.data-plane.promoter-feature-matrix --repo-root <repo-root> > docs/studies/<study-id>/campaign.yaml`
4. Create `datasets.yaml` from real affiliated dataset ids and sync posture.
5. Create `status.md` from the current study facts rather than a status-provider template.
6. Create `ops.study.yaml` with lifecycle and execution surfaces. Add
   `ops_surfaces` only after this study owns concrete status/preflight providers.
7. Create the audit directory:
   `mkdir -p docs/studies/<study-id>/audits`
8. Edit the checked-in `index.yaml`, `datasets.yaml`, `status.md`, and
   `ops.study.yaml` so they point at real ids, paths, and commands.
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

## Study Record Authoring

**Owner:** dnadesign-maintainers
**Last verified:** 2026-08-08

Study records keep long-running project context discoverable without turning
the top-level [Study Records](../README.md) index into a status/preflight
ladder. The same shape works in this public repository or in an explicit
private study repository; the public/private boundary decides which one owns
the record.

### Required record shape

```text
docs/studies/<study-id>/
  README.md      # directory ontology and first-hop usage
  record/
    campaign.yaml  # optional explicit progress manifest
    datasets.yaml
    status.md
  operations/
    ops.study.yaml
    catalog/      # optional OPS catalog index
      contracts/
        registry/
    contract/      # optional split parts loaded by ops.study.yaml
      lifecycle/
      surfaces/
        execution/
      status/
      readiness/
        checks/
    runtime/
      command-groups/
        pipeline.yaml  # optional exact command groups/runtime context
  routes/
    README.md      # optional, preferred once the study spans owner surfaces
    ...            # optional focused route details for bulky owner surfaces
  contexts/      # optional, long-form rationale and handoff notes
  compiler/      # optional, study-owned compiler inputs/config
  workbench/     # optional, durable ontology, design-set, and provenance records
  audits/
```

- `README.md`, when present, names the study-local directory ontology and the
  first-hop usage rules. Use YAML frontmatter for agent routing fields:
  `doc_id`, `surface`, `study_id`, `owner`, `last_verified`, and `first_hop`
  or `entrypoint`.
- `record/campaign.yaml` is an optional explicit multi-step manifest generated from
  `uv run ops progress scaffold ...` and then filled with real artifact paths.
- `record/datasets.yaml` declares which USR datasets belong to the study and how they
  sync across local and remote roots.
- `record/status.md` records current state and concise next actions.
- `operations/ops.study.yaml` is the OPS-facing contract for lifecycle ordering
  or track maps, record sources, artifacts, execution surfaces, and preflight
  planning.
- `operations/contract/`, when present, stores split YAML sections referenced
  by `operations/ops.study.yaml` `parts`. A part can point at one file or a
  short ordered list of files. Use lists when execution surfaces or preflight
  checks would otherwise become a flat shelf. Keep fragments under
  `lifecycle/`, `surfaces/execution/{runbooks,commands}/`, `status/`, and
  `readiness/checks/`. If one owner lane grows beyond scan-friendly size,
  split it into a nested semantic directory and keep `ops.study.yaml` as the
  one-hop index.
- `operations/catalog/`, when present, stores status/preflight catalog docs for
  studies with concrete OPS providers. Put catalog contract pages under
  `operations/catalog/contracts/` and their registry sidecars under
  `operations/catalog/contracts/registry/`.
- `operations/runtime/command-groups/pipeline.yaml`, when present, records exact command groups or
  runtime context that should not be reconstructed from generic tool docs.
- `routes/README.md`, when present, is the study-owned one-hop handoff page.
  Use YAML frontmatter with `surface: study-route-map`, `study_id`, and the
  status/preflight posture so blank-thread agents can classify it before
  reading the full page.
- `routes/`, when present, keeps focused owner-surface details out of the
  one-hop router.
- `contexts/`, when present, stores long-form rationale or handoff notes that
  should not crowd the router. Tool bindings that are durable study context,
  such as LatentDNA binding files, live under context-specific subdirectories.
- `compiler/`, when present, stores study-owned compiler input/config records
  such as registries or convenience label lists.
- `workbench/`, when present, stores study-specific hypotheses, ontology terms,
  design sets, compiler runs, and materialization provenance that should
  outlive transient tool outputs. Prefer `workbench/ontology/`,
  `workbench/design_sets/`, and `workbench/provenance/` once more than one
  record family exists.
- `audits/` stores machine-readable sync audit JSON files referenced from
  `record/datasets.yaml`.

### Study Bootstrap

Use this sequence for studies that need the shared checked-in record shape.
Only add study status/preflight providers when that specific study owns the
implementation.

1. Create the study directory:
   `mkdir -p docs/studies/<study-id>`
2. If `docs/studies/index.yaml` is missing, author it from the active study id
   and explicit `record_root` entries; do not copy a family-specific template.
3. Create `README.md` when the study has subdirectories or multiple owner
   surfaces.
4. Generate a campaign manifest only when a registered progress route is useful:
   `uv run ops progress scaffold --related-to usr.data-plane.promoter-feature-matrix --repo-root <repo-root> > docs/studies/<study-id>/record/campaign.yaml`
5. Create `record/datasets.yaml` from real affiliated dataset ids and sync posture.
6. Create `record/status.md` from the current study facts rather than a status-provider template.
7. Create `operations/ops.study.yaml` with lifecycle and execution surfaces.
   If it grows beyond a short entrypoint, keep the entrypoint thin and place
   section bodies under `operations/contract/`. Add `ops_surfaces` only after
   this study owns concrete status/preflight providers.
8. Create the audit directory:
   `mkdir -p docs/studies/<study-id>/audits`
9. Edit the checked-in `index.yaml`, `record/datasets.yaml`,
   `record/status.md`, and `operations/ops.study.yaml` so they point at real
   ids, paths, and commands.
10. Add `routes/README.md` when the study spans several owner surfaces.
11. Add `routes/` detail pages when route detail would otherwise bloat the
    one-hop router.
12. Add `operations/catalog/` when the study owns concrete status/preflight
    providers.
13. Add `contexts/<tool>/` when cross-tool study context needs a durable home
    outside the root control plane.
14. Add broader `contexts/` pages when long-form rationale or handoff notes need a durable
    home outside the router.
15. Add `compiler/` when the study owns narrow compiler inputs or normalization
    metadata.
16. Add `workbench/` when hypotheses, design sets, and run provenance need a
    durable study-owned home; split it into `ontology/`, `design_sets/`, and
    `provenance/` when the records would otherwise flatten at the workbench
    root.
17. Add `operations/runtime/command-groups/pipeline.yaml` when the study has exact Construct, Infer,
    batch, or Notify command groups.
18. Refresh evidence with:
    `uv run ops progress campaign --repo-root <repo-root> --manifest docs/studies/<study-id>/record/campaign.yaml`

### Dataset registry contract

Use `record/datasets.yaml` to keep one DRY declaration of study-affiliated datasets.
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
`strict_bootstrap_id: true` in `record/datasets.yaml` and use an explicit dataset id
for the first pull rather than relying on local name guessing.

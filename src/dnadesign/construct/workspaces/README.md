## construct workspaces

Use this directory for construct studies. Each workspace keeps its configs, registry, and default outputs together.

### Start with one of two paths

- [Packaged demo](demo_promoter_swap_pdual10/README.md): curated pDual-10 promoter-swap tracer bullet with both 1 kb window and full-plasmid configs.
- [Packaged source-of-truth demo](demo_promoter_swap_pdual10_source_of_truth/README.md): curated pDual-10 promoter-swap workspace that writes both window projects into one shared USR dataset for infer and downstream handoff.
- Blank workspace: scaffold your own study and import your own input/template datasets.

### Quick start

```bash
# Blank workspace for a custom construct study.
uv run construct workspace init --id demo_construct # Create a blank construct workspace scaffold.
uv run construct workspace show --workspace src/dnadesign/construct/workspaces/demo_construct # Review workspace registry/config paths.
uv run construct workspace doctor --workspace src/dnadesign/construct/workspaces/demo_construct # Verify workspace registry/config health before editing or running.

# Packaged promoter-swap demo copied into a new workspace id.
uv run construct workspace init --id demo_promoter_swap --profile promoter-swap-demo # Copy the packaged promoter-swap demo into a new workspace id.
# The cd path below assumes the default workspace root from repo root.
# If you used --root or CONSTRUCT_WORKSPACE_ROOT, cd into the printed workspace path instead.
cd src/dnadesign/construct/workspaces/demo_promoter_swap # Enter the newly initialized demo workspace.
./runbook.sh --mode dry-run --config config.slot_a.window.yaml # Execute the packaged tracer-bullet dry run for the slot_a window config.

# Packaged shared-dataset source-of-truth demo.
uv run construct workspace init --id demo_construct_source_of_truth --profile promoter-swap-source-of-truth-demo # Copy the packaged shared-dataset demo into a new workspace id.
cd src/dnadesign/construct/workspaces/demo_construct_source_of_truth # Enter the newly initialized source-of-truth workspace.
./runbook.sh --mode dry-run-all # Dry-run both packaged projects into one semantic USR dataset.
```

If you initialize a workspace outside the repo tree, reuse the `uv run --project <repo-root> construct ...` commands printed by `workspace init` or the packaged `runbook.sh` wrapper.

### What `workspace init` creates

- blank profile:
  - `workspaces/<id>/construct.workspace.yaml`
  - `workspaces/<id>/config.yaml`
  - `workspaces/<id>/inputs/README.md`
  - `workspaces/<id>/inputs/import_manifest.template.yaml`
- `promoter-swap-demo` profile:
  - `workspaces/<id>/construct.workspace.yaml`
  - `workspaces/<id>/README.md`
  - `workspaces/<id>/runbook.md`
  - `workspaces/<id>/runbook.sh`
  - `workspaces/<id>/config.slot_a.window.yaml`
  - `workspaces/<id>/config.slot_a.full.yaml`
  - `workspaces/<id>/config.slot_b.window.yaml`
  - `workspaces/<id>/config.slot_b.full.yaml`
  - `workspaces/<id>/inputs/README.md`
- `promoter-swap-source-of-truth-demo` profile:
  - `workspaces/<id>/construct.workspace.yaml`
  - `workspaces/<id>/README.md`
  - `workspaces/<id>/runbook.md`
  - `workspaces/<id>/runbook.sh`
  - `workspaces/<id>/config.slot_a.window.yaml`
  - `workspaces/<id>/config.slot_b.window.yaml`
  - `workspaces/<id>/inputs/README.md`
- all profiles:
  - `workspaces/<id>/outputs/logs/ops/audit/`

### Edit these first

- safe edit set for a blank workspace:
  - `construct.workspace.yaml`
  - `config.yaml`
  - `inputs/import_manifest.template.yaml`
- safe edit set for the packaged demo copy:
  - `construct.workspace.yaml`
  - `config.*.yaml`
  - `inputs/seed_manifest.yaml`
- operator helpers, not primary contract surfaces:
  - `README.md`
  - `runbook.md`
  - `runbook.sh`
- generated/run outputs:
  - `outputs/**`

### Workspace contract

- Workspace ids must be directory names, not paths.
- Existing workspaces are never overwritten.
- Every workspace carries `construct.workspace.yaml` as the project registry and provenance surface.
- `construct workspace doctor` is the contract check for registry/config drift before project execution.
- Packaged workspaces default construct IO to `outputs/usr_datasets`, consistent with repo workspace-scoping guidance.
- Blank workspaces also scaffold explicit `root: outputs/usr_datasets` entries so custom studies do not fall back to repo-package datasets implicitly.
- External/shared USR roots remain allowed, but only through explicit `root:` fields or `construct seed --root <path>`.
- One construct job uses one template; multi-template or slot-matrix studies are represented as multiple project entries and config files in the workspace registry.
- The packaged promoter-swap demo exposes `./runbook.sh --mode seed|validate|dry-run|run|validate-all` as the local workspace entrypoint.
- The packaged source-of-truth demo exposes `./runbook.sh --mode seed|validate-all|dry-run-all|run-all` as the local workspace entrypoint for the shared-dataset flow; the authoritative cross-tool handoff still lives in `../../usr/docs/operations/construct-infer-source-of-truth-demo.md`.

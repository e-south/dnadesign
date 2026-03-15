## construct workspaces

Use this directory for construct studies. Each workspace keeps its configs, registry, and default outputs together.

### Start with one of two paths

- [Packaged demo](demo_promoter_swap_pdual10/README.md): curated pDual-10 promoter-swap tracer bullet with both 1 kb window and full-plasmid configs.
- Blank workspace: scaffold your own study and import your own input/template datasets.

### Quick start

```bash
# Blank workspace for a custom construct study.
uv run construct workspace init --id demo_construct
uv run construct workspace show --workspace src/dnadesign/construct/workspaces/demo_construct
uv run construct workspace doctor --workspace src/dnadesign/construct/workspaces/demo_construct

# Packaged promoter-swap demo copied into a new workspace id.
uv run construct workspace init --id demo_promoter_swap --profile promoter-swap-demo
# The cd path below assumes the default workspace root from repo root.
# If you used --root or CONSTRUCT_WORKSPACE_ROOT, cd into the printed workspace path instead.
cd src/dnadesign/construct/workspaces/demo_promoter_swap
./runbook.sh --mode dry-run --config config.slot_a.window.yaml
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
- External/shared USR roots remain allowed, but only through explicit `root:` fields or `construct seed --root <path>`.
- One construct job uses one template; multi-template or slot-matrix studies are represented as multiple project entries and config files in the workspace registry.
- The packaged promoter-swap demo exposes `./runbook.sh --mode seed|validate|dry-run|run|validate-all` as the canonical operator entrypoint.

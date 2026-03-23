## Construct workspaces

Use this directory for packaged construct workspace templates and examples. New workspaces default to the current working directory unless `--root` or `CONSTRUCT_WORKSPACE_ROOT` is set.

List the local construct workspaces in the active root plus the packaged templates with `uv run construct workspace list`.

### Start with one of three paths

- [Packaged local demo](demo_anchor_template_local/README.md): a didactic anchor-into-template tracer bullet with both 1 kb window and full-context configs.
- [Packaged shared-dataset demo](demo_anchor_template_shared_dataset/README.md): the same anchor/template contract, but with two audited projects writing into one downstream USR dataset.
- [Study-owned pDual-10 surface](study_stress_ethanol_cipro_pdual10/README.md): the real `stress_ethanol_cipro_growth` Construct handoff, reading merged anchors from the shared USR root and writing template-backed contexts into a shared downstream dataset.
- Blank workspace: scaffold your own study and import your own input/template datasets.

### Quick start

```bash
# Inspect the packaged construct workspaces before copying one.
uv run construct workspace list

# Blank workspace for a custom construct study.
uv run construct workspace init --id demo_construct # Create a blank construct workspace scaffold.
uv run construct workspace show --workspace demo_construct # Review workspace registry/config paths.
uv run construct workspace doctor --workspace demo_construct # Verify workspace registry/config health before editing or running.

# Packaged local anchor/template demo copied into a new workspace id.
uv run construct workspace init --id demo_anchor_template --profile anchor-template-demo # Copy the packaged local demo into a new workspace id.
# The cd path below assumes the default workspace root from the current working directory.
# If you used --root or CONSTRUCT_WORKSPACE_ROOT, cd into the printed workspace path instead.
cd demo_anchor_template # Enter the newly initialized demo workspace.
./runbook.sh --mode dry-run --config config.slot_a.window.yaml # Execute the packaged tracer-bullet dry run for the slot_a window config.

# Packaged shared-dataset demo.
uv run construct workspace init --id demo_construct_shared_dataset --profile anchor-template-shared-dataset-demo # Copy the packaged shared-dataset demo into a new workspace id.
cd demo_construct_shared_dataset # Enter the newly initialized shared-dataset workspace.
./runbook.sh --mode dry-run-all # Dry-run both packaged projects into one semantic USR dataset.
```

If you initialize a workspace outside the repo tree, reuse the `uv run --project <repo-root> construct ...` commands printed by `workspace init` or the packaged `runbook.sh` wrapper.

### What `workspace init` creates

- blank profile:
  - `workspaces/<id>/construct.workspace.yaml`
  - `workspaces/<id>/config.yaml`
  - `workspaces/<id>/inputs/README.md`
  - `workspaces/<id>/inputs/import_manifest.template.yaml`
- `anchor-template-demo` profile:
  - `workspaces/<id>/construct.workspace.yaml`
  - `workspaces/<id>/README.md`
  - `workspaces/<id>/runbook.md`
  - `workspaces/<id>/runbook.sh`
  - `workspaces/<id>/config.slot_a.window.yaml`
  - `workspaces/<id>/config.slot_a.full.yaml`
  - `workspaces/<id>/config.slot_b.window.yaml`
  - `workspaces/<id>/config.slot_b.full.yaml`
  - `workspaces/<id>/inputs/README.md`
- `anchor-template-shared-dataset-demo` profile:
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
- That workspace-local root is a construct-owned export only.
  The shared study dataset exists only after an explicit copy or sync into a
  named shared USR root.
- External/shared USR roots remain allowed, but only through explicit `root:` fields or `construct seed --root <path>`.
- One construct job uses one template; multi-template or slot-matrix studies are represented as multiple project entries and config files in the workspace registry.
- The packaged local demo exposes `./runbook.sh --mode seed|validate|dry-run|run|validate-all` as the local workspace entrypoint.
- The packaged shared-dataset demo exposes `./runbook.sh --mode seed|validate-all|dry-run-all|run-all` as the local workspace entrypoint for the shared-dataset flow; the authoritative cross-tool handoff still lives in `../../usr/docs/operations/construct-infer-shared-dataset-runbook.md`.
- The study-owned pDual-10 surface is a checked-in execution surface, not a demo profile; use its tracked workspace/config paths when the question is about the live promoter study rather than generic Construct behavior.

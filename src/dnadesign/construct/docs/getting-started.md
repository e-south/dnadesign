---
doc_id: construct-getting-started
title: Construct getting started
owner: dnadesign-maintainers
status: active
last_verified: 2026-08-08
---

# Construct getting started

This page gets you from zero to a validated Construct run with the fewest moving parts.

### Path 1: packaged isolated demo

```bash
uv run construct workspace init --id demo_anchor_template --profile anchor-template-demo
cd demo_anchor_template
uv run construct workspace doctor --workspace .
./runbook.sh --mode dry-run --config config.slot_a.window.yaml
```

Use this path when you want a known-good anchor/template tracer bullet. By default, `workspace init` creates the workspace under your current working directory and keeps construct IO inside `outputs/usr_datasets` in that workspace. If you use `--root` or `CONSTRUCT_WORKSPACE_ROOT`, `cd` into the printed workspace path instead. If the workspace lives outside the repo tree, reuse the `uv run --project /path/to/dnadesign construct ...` commands printed by `workspace init`.

### Path 2: packaged shared-dataset demo

```bash
uv run construct workspace init --id demo_construct_shared_dataset --profile anchor-template-shared-dataset-demo
cd demo_construct_shared_dataset
uv run construct workspace doctor --workspace .
./runbook.sh --mode dry-run-all
```

Use this path when both packaged window projects should accumulate into one semantic USR dataset before infer or Notify pick it up. The profile keeps the shared-dataset contract explicit in `construct.workspace.yaml` by tracking both config artifacts and routing assertions instead of relying on manual config repointing.

### Path 3: blank custom workspace

```bash
uv run construct workspace init --id my_construct_workspace
cd my_construct_workspace
uv run construct workspace doctor --workspace .
uv run construct seed import-manifest \
  --manifest inputs/import_manifest.template.yaml \
  --root outputs/usr_datasets
```

Then edit `config.yaml`, update the matching `project.artifacts.config` and `project.contract` entry in `construct.workspace.yaml`, run `workspace validate-project --runtime`, and finish with `workspace run-project --dry-run`.

The blank scaffold writes an explicit workspace-local
`root: outputs/usr_datasets` entry into `config.yaml`, so an incomplete or
misrouted workspace fails before materialization.

### Keep the model simple

- Packaged demos use local semantic datasets such as `anchor_parts_demo` and `template_parts_demo`.
- Real projects choose their own stable dataset IDs; study vocabulary belongs
  in the caller's config and provenance.
- `anchor`, `template`, and `helper` are placement roles in the current config,
  not biological classes or dataset path categories.
- One construct job uses one template plus one or more placed parts.
- Multi-template or slot-matrix studies are represented as multiple project entries in `construct.workspace.yaml`, each with its own tracked config artifact.

### Continue reading

- [Construct docs](README.md)
- [Workspaces guide](../workspaces/README.md)
- [Construct -> USR -> Infer shared dataset runbook](../../usr/docs/operations/assembly/construct-infer-shared-dataset-runbook.md)
- [Config reference](reference/config.md)
- [Workspace registry reference](reference/workspace-registry.md)

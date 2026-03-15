## construct getting started

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-14

This page gets you from zero to a validated construct run with the fewest moving parts.

### Path 1: packaged demo

```bash
uv run construct workspace init --id demo_promoter_swap --profile promoter-swap-demo
cd src/dnadesign/construct/workspaces/demo_promoter_swap
uv run construct workspace doctor --workspace .
./runbook.sh --mode dry-run --config config.slot_a.window.yaml
```

Use this path when you want a known-good tracer bullet. By default it keeps construct IO inside `outputs/usr_datasets` in the workspace. If the workspace lives outside the repo tree, reuse the `uv run --project /path/to/dnadesign construct ...` commands printed by `workspace init`.

### Path 2: blank custom workspace

```bash
uv run construct workspace init --id my_construct_study
cd src/dnadesign/construct/workspaces/my_construct_study
uv run construct workspace doctor --workspace .
uv run construct seed import-manifest \
  --manifest inputs/import_manifest.template.yaml \
  --root outputs/usr_datasets
```

Then edit `config.yaml`, update `construct.workspace.yaml`, run `workspace validate-project --runtime`, and finish with `workspace run-project --dry-run`.

### Keep the model simple

- USR dataset ids stay biological and semantic, for example `mg1655_promoters` or `plasmids`.
- `anchor`, `template`, and `helper` are construct roles assigned inside the config, not dataset path categories.
- One construct job uses one template plus one or more placed parts.
- Multi-template or slot-matrix studies are represented as multiple project configs in `construct.workspace.yaml`.

### Continue reading

- [Docs overview](README.md)
- [Docs index](index.md)
- [Workspaces guide](../workspaces/README.md)
- [Config reference](reference/config.md)
- [Workspace registry reference](reference/workspace-registry.md)

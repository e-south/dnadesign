## Construct getting started

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-15

This page gets you from zero to a validated Construct run with the fewest moving parts.

### Path 1: packaged isolated demo

```bash
uv run construct workspace init --id demo_promoter_swap --profile promoter-swap-demo
cd demo_promoter_swap
uv run construct workspace doctor --workspace .
./runbook.sh --mode dry-run --config config.slot_a.window.yaml
```

Use this path when you want a known-good tracer bullet. By default, `workspace init` creates the workspace under your current working directory and keeps construct IO inside `outputs/usr_datasets` in that workspace. If you use `--root` or `CONSTRUCT_WORKSPACE_ROOT`, `cd` into the printed workspace path instead. If the workspace lives outside the repo tree, reuse the `uv run --project /path/to/dnadesign construct ...` commands printed by `workspace init`.

### Path 2: packaged shared source-of-truth demo

```bash
uv run construct workspace init --id demo_construct_source_of_truth --profile promoter-swap-source-of-truth-demo
cd demo_construct_source_of_truth
uv run construct workspace doctor --workspace .
./runbook.sh --mode dry-run-all
```

Use this path when both packaged window projects should accumulate into one semantic USR dataset before infer or Notify pick it up. The profile keeps the source-of-truth contract explicit in `construct.workspace.yaml` instead of relying on manual config repointing.

### Path 3: blank custom workspace

```bash
uv run construct workspace init --id my_construct_study
cd my_construct_study
uv run construct workspace doctor --workspace .
uv run construct seed import-manifest \
  --manifest inputs/import_manifest.template.yaml \
  --root outputs/usr_datasets
```

Then edit `config.yaml`, update `construct.workspace.yaml`, run `workspace validate-project --runtime`, and finish with `workspace run-project --dry-run`.

The blank scaffold now writes explicit workspace-local `root: outputs/usr_datasets` entries into `config.yaml` so custom studies stay fail-fast and workspace-scoped by default.

### Keep the model simple

- USR dataset ids stay biological and semantic, for example `mg1655_promoters` or `plasmids`.
- `anchor`, `template`, and `helper` are construct roles assigned inside the config, not dataset path categories.
- One construct job uses one template plus one or more placed parts.
- Multi-template or slot-matrix studies are represented as multiple project configs in `construct.workspace.yaml`.

### Continue reading

- [Construct docs](README.md)
- [Workspaces guide](../workspaces/README.md)
- [Construct -> USR -> Infer source-of-truth runbook](../../usr/docs/operations/construct-infer-source-of-truth-runbook.md)
- [Config reference](reference/config.md)
- [Workspace registry reference](reference/workspace-registry.md)

## Workspace layout

DenseGen is most ergonomic when each run is self-contained and uses config-relative paths.

```
workspace/
  config.yaml
  inputs/
  outputs/
    meta/
    logs/
    pools/
    libraries/
    tables/
    plots/
    report/
```

Notes:
- Config resolution policy is canonical in `../reference/cli.md#config-resolution`.
- Outputs live under `densegen.run.root` (typically `.`).
- If run outputs already exist, `dense run` auto-resumes (same as `--resume`);
  use `dense run --fresh` to clear outputs.
- `dense workspace init` defaults to `src/dnadesign/densegen/workspaces` (or
  `$DENSEGEN_WORKSPACE_ROOT` when set).

Tip: `dense workspace init --id <workspace_name> --from-workspace <demo_workspace>` scaffolds a workspace.
Use `--output-mode local|usr|both` to choose parquet output, USR output, or both.
Use `dense workspace where --format json` to confirm the effective workspace root and source workspace root.

---

@e-south

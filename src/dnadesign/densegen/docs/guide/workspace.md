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
- `dense workspace init` defaults to `src/dnadesign/densegen/workspaces/runs` (or
  `$DENSEGEN_WORKSPACE_ROOT` when set) to keep local run outputs centralized.

Tip: `dense workspace init --id <run_name> --template-id <template>` scaffolds a workspace.
Use `--output-mode local|usr|both` to choose parquet output, USR output, or both.
Use `dense workspace where --format json` to confirm the effective run root and template root.

---

@e-south

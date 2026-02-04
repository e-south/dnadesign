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
- CLI requires an explicit config path: pass `-c/--config` or set `DENSEGEN_CONFIG_PATH`.
- Outputs live under `densegen.run.root` (typically `.`).
- If run outputs already exist, `dense run` auto-resumes (same as `--resume`);
  use `dense run --fresh` to clear outputs.

Tip: `dense workspace init --id <run_name> --template-id <template>` scaffolds a workspace.

---

@e-south

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
- CLI resolves config from `-c`, `DENSEGEN_CONFIG_PATH`, `./config.yaml`, then nearest parent.
- Outputs live under `densegen.run.root` (typically `.`).
- If run outputs already exist, choose explicitly:
  `dense run --resume` to continue, or `dense run --fresh` to clear outputs.

Tip: `dense workspace init --id <run_name> --template-id <template>` scaffolds a workspace.

---

@e-south

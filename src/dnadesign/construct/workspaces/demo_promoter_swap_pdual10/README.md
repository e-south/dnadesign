## demo_promoter_swap_pdual10 workspace

Copy this packaged workspace into a new local workspace id with:

```bash
uv run construct workspace init --id my_promoter_swap --profile promoter-swap-demo
cd src/dnadesign/construct/workspaces/my_promoter_swap
uv run construct workspace show --workspace .
./runbook.sh --mode dry-run --config config.slot_a.window.yaml
```

This workspace defaults all construct IO to `outputs/usr_datasets` so the study stays self-contained. If you want a shared USR root instead, edit the config `root:` fields deliberately and re-run `construct workspace show` to verify the project inventory.

- Workspace registry: [construct.workspace.yaml](construct.workspace.yaml)
- Runbook: [runbook.md](runbook.md)
- Runbook wrapper: [runbook.sh](runbook.sh)
- Configs:
  - [config.slot_a.window.yaml](config.slot_a.window.yaml)
  - [config.slot_a.full.yaml](config.slot_a.full.yaml)
  - [config.slot_b.window.yaml](config.slot_b.window.yaml)
  - [config.slot_b.full.yaml](config.slot_b.full.yaml)
- Inputs notes: [inputs/README.md](inputs/README.md)
- All workspaces: [../README.md](../README.md)

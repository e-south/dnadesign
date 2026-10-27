## demo_anchor_template_local workspace

Use this packaged workspace when you want the smallest Construct demo:
one anchor part placed into one larger template, with the resulting sequence
and `construct__*` provenance written into workspace-local USR datasets.

Copy it into a new workspace id with:

```bash
uv run construct workspace init --id my_anchor_template_demo --profile anchor-template-demo
cd my_anchor_template_demo
uv run construct workspace show --workspace .
./runbook.sh --mode dry-run --config config.slot_a.window.yaml
```

This workspace keeps all IO in `outputs/usr_datasets` by default so the demo
stays self-contained. The configs illustrate two independent choices:

- which slot in the template should receive the anchor (`slot_a` or `slot_b`)
- whether the realized output should be a 1 kb window or the full template context

If you want a shared USR root instead, edit the config `root:` fields
deliberately and re-run `construct workspace show` to verify the project
inventory.

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

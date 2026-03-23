## demo_anchor_template_shared_dataset workspace

Use this packaged workspace when you want to show the second Construct pattern:
multiple audited projects accumulating into one downstream dataset.

Copy it into a new workspace id with:

```bash
uv run construct workspace init --id my_anchor_template_shared_demo --profile anchor-template-shared-dataset-demo
cd my_anchor_template_shared_demo
uv run construct workspace show --workspace .
./runbook.sh --mode dry-run-all
```

This workspace keeps construct IO inside `outputs/usr_datasets` by default and
routes both packaged window projects into one semantic dataset,
`anchor_template_shared_dataset_demo`. Use it when Construct should hand one
USR-backed dataset to downstream consumers without relying on implicit config
repointing.

- Workspace registry: [construct.workspace.yaml](construct.workspace.yaml)
- Runbook: [runbook.md](runbook.md)
- Runbook wrapper: [runbook.sh](runbook.sh)
- Configs:
  - [config.slot_a.window.yaml](config.slot_a.window.yaml)
  - [config.slot_b.window.yaml](config.slot_b.window.yaml)
- Inputs notes: [inputs/README.md](inputs/README.md)
- Shared cross-tool runbook: [../../../usr/docs/operations/construct-infer-shared-dataset-runbook.md](../../../usr/docs/operations/construct-infer-shared-dataset-runbook.md)
- Broader feature-matrix runbook: [../../../usr/docs/operations/promoter-characterization-feature-matrix.md](../../../usr/docs/operations/promoter-characterization-feature-matrix.md)
- All workspaces: [../README.md](../README.md)

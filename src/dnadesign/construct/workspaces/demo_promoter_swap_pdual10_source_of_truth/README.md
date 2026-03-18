## demo_promoter_swap_pdual10_source_of_truth workspace

Copy this packaged workspace into a new local workspace id with:

```bash
uv run construct workspace init --id my_promoter_source_of_truth --profile promoter-swap-source-of-truth-demo
cd my_promoter_source_of_truth
uv run construct workspace show --workspace .
./runbook.sh --mode dry-run-all
```

This workspace keeps construct IO inside `outputs/usr_datasets` by default and routes both packaged window projects into one semantic dataset, `pdual10_source_of_truth_demo`. Use it when construct should hand one USR-backed source-of-truth dataset into infer and downstream consumers without manual config repointing.

- Workspace registry: [construct.workspace.yaml](construct.workspace.yaml)
- Runbook: [runbook.md](runbook.md)
- Runbook wrapper: [runbook.sh](runbook.sh)
- Configs:
  - [config.slot_a.window.yaml](config.slot_a.window.yaml)
  - [config.slot_b.window.yaml](config.slot_b.window.yaml)
- Inputs notes: [inputs/README.md](inputs/README.md)
- Shared cross-tool runbook: [../../../usr/docs/operations/construct-infer-source-of-truth-runbook.md](../../../usr/docs/operations/construct-infer-source-of-truth-runbook.md)
- Broader feature-matrix runbook: [../../../usr/docs/operations/promoter-characterization-feature-matrix.md](../../../usr/docs/operations/promoter-characterization-feature-matrix.md)
- All workspaces: [../README.md](../README.md)

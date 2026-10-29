[Construct workspaces](../README.md)

# Combine placements in one dataset

This demo places the same input parts at two template sites and saves both
sets of 1,000-base windows in one SeqLedger dataset. Downstream tools can read
that dataset while using the saved lineage to distinguish the placements.

With the [Construct environment active](../../docs/getting-started.md#install-and-activate):

```bash
# Copy the shared-dataset example into a new local workspace.
construct workspace init --id my_anchor_template_shared_demo \
  --profile anchor-template-shared-dataset-demo
cd my_anchor_template_shared_demo
construct workspace show --workspace .
```

Continue with the [runbook](runbook.md) to seed inputs, preview both projects
and save the results. Both projects write to `anchor_template_shared_dataset_demo`
under the workspace's `outputs/usr_datasets` root.

## Files and next steps

- [Workspace registry](construct.workspace.yaml): project names and expected dataset identities.
- [Slot A configuration](config.slot_a.window.yaml) and [slot B configuration](config.slot_b.window.yaml): placement and output settings.
- [Input notes](inputs/README.md): records created by the seed command.
- [Construct → SeqLedger → Infer guide](../../../seqledger/docs/operations/assembly/construct-infer-shared-dataset-runbook.md): calculate sequence-model outputs from the assembled windows.
- [Promoter characterization example](../../../seqledger/docs/operations/promoter/characterization-feature-matrix.md): use the shared dataset in a larger analysis.

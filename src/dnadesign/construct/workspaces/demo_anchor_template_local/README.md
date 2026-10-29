[Construct workspaces](../README.md)

# Place parts in a template

This demo places each supplied DNA part at one of two sites in a circular
template. Each configuration saves either a 1,000-base window around the part
or the full assembled construct to a separate SeqLedger dataset.

For installation and a complete first run, follow
[Build your first construct](../../docs/getting-started.md). To explore all
four configurations, copy the demo with your Construct environment active:

```bash
# Create a local workspace in the current directory.
construct workspace init --id my_anchor_template_demo \
  --profile anchor-template-demo
cd my_anchor_template_demo
# List its project names and configuration files.
construct workspace show --workspace .
```

Continue with the [step-by-step runbook](runbook.md) to seed the inputs,
preview a project and save its results. All demo data are stored under
`outputs/usr_datasets` inside the copied workspace.

## Choose an output

| Project | Configuration | Output |
| --- | --- | --- |
| `slot_a_window` | [Slot A window](config.slot_a.window.yaml) | 1,000 bases around the inserted part. |
| `slot_a_full` | [Slot A full construct](config.slot_a.full.yaml) | The entire assembled template. |
| `slot_b_window` | [Slot B window](config.slot_b.window.yaml) | 1,000 bases around the inserted part. |
| `slot_b_full` | [Slot B full construct](config.slot_b.full.yaml) | The entire assembled template. |

The [workspace registry](construct.workspace.yaml) lists the expected input,
template and output identities for each project. See the [input notes](inputs/README.md)
for the seed receipt, or the [shared-dataset demo](../demo_anchor_template_shared_dataset/README.md)
to combine placements in one output dataset.

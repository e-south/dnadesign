<!--
**Type:** route
**Plane:** data-plane
**Owner-boundary:** seqledger
**Entry artifact:** operator intent for a SeqLedger-backed workflow branch
**Exit artifact:** chosen runbook link plus summary command chain
**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-28
-->

[SeqLedger guide](../../README.md) / [Operations](../README.md)

# Dataset workflow map

Use the operation that matches what should change. Routine reading and export do not require materialization or maintenance.

| Task | Command or API | Procedure |
| --- | --- | --- |
| Import sequences | `init`, `import` | [Local quickstart](../../getting-started/cli-quickstart.md) |
| Inspect records and annotations | `head`, `schema`, `Dataset.scan()` | [Python API](../../reference/python-api.md) |
| Add tool outputs | `namespace register`, `attach` | [Overlays and registry](../../reference/overlay-and-registry.md) |
| Export the combined table | `export` | [Quickstart export](../../getting-started/cli-quickstart.md#5-validate-and-export) |
| Compare or transfer complete datasets | `diff`, then `pull` or `push` | [Sync quickstart](../sync/quickstart.md) |
| Combine selected datasets | `maintenance merge` | [Shared dataset assembly](../assembly/multi-source-shared-dataset.md) |
| Consolidate annotation parts | `maintenance overlay-compact` | [Maintenance](../../reference/maintenance.md) |
| Rewrite the base table with annotations | `materialize` | [Maintenance](../../reference/maintenance.md) |
| Inspect data changes | `events tail` | [Event log](../../reference/event-log.md) |

## Choose a transfer direction

Run SeqLedger on the machine you are working from. `pull` copies from the named remote to that machine; `push` copies from that machine to the remote. Preview the difference first, then choose one direction. They are not an automatic two-way merge.

Start with [remote setup](../sync/setup.md) if the remote is not configured. Use [sync target modes](../sync/modes.md) when mapping an explicit dataset directory or a single file. For failures, use [sync troubleshooting](../sync/troubleshooting.md).

## Connect processing steps

[Construct-to-Infer assembly](../assembly/construct-infer-shared-dataset-runbook.md) connects resolved sequence contexts to model outputs. [Multi-source assembly](../assembly/multi-source-shared-dataset.md) explains which records and annotations survive a merge. The [operations index](../README.md) lists more examples.

Contributors can run the [sync fidelity drills](../sync/fidelity-drills.md) when changing transfer or storage behavior.

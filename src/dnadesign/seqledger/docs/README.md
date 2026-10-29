<!--
**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-28
-->

[SeqLedger](../README.md)

# SeqLedger guide

SeqLedger keeps sequence records, derived annotations and a history of changes in a dataset directory. It gives command-line tools and Python programs a shared way to read and update those records.

**Start with [Create your first dataset](getting-started/cli-quickstart.md).** Import three example sequences, attach a score to each, inspect the result and export it. The walkthrough runs locally and needs no remote service.

## Three concepts to know

- A **dataset** holds sequence records with stable IDs derived from sequence content and biological type.
- A **namespace** groups related annotation columns, such as `demo__score`. A registry declares their names and types.
- An **overlay** stores annotations separately from the base sequence table. SeqLedger joins them when reading or exporting a dataset.

## Choose a task

| I want to… | Guide |
| --- | --- |
| Read and annotate data from Python | [Python API](reference/python-api.md) |
| Browse a dataset interactively | [Notebook and inspection commands](getting-started/notebook.md) |
| Transfer a dataset to another machine | [Sync over SSH](operations/sync/README.md) |
| Combine datasets from several tools | [Shared dataset assembly](operations/assembly/multi-source-shared-dataset.md) |
| Choose an operation for an existing dataset | [Workflow map](operations/routes/workflow-map.md) |

## Understand stored data

The [schema reference](reference/schema-contract.md) defines sequence identity and required columns. The [overlay guide](reference/overlay-and-registry.md) explains joins, updates and namespace registration. The [storage layout](reference/dataset-layout-and-code-map.md) shows the files these operations create.

[Reference](reference/README.md) also covers event logs, sequence views and maintenance. Exporting a table is different from copying a complete dataset, which includes its history and sidecars.

## Connect other tools

[Construct](../../construct/) can assemble sequences into a shared dataset, and [Infer](../../infer/) can add model outputs. Follow the [Construct-to-Infer procedure](operations/assembly/construct-infer-shared-dataset-runbook.md) when connecting those steps. [Notify](../../notify/) can consume dataset mutation events from `.events.log`.

The [operations index](operations/README.md) includes additional assembly examples and transfer procedures. Contributors can use the [architecture guide](architecture-introspection.md) for implementation boundaries and checks.

## construct docs

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-15

Use this page when you know what you want to do. Use [index.md](index.md) when you know what kind of document you need.

### Choose a task

- [Get to a first dry-run](getting-started.md): shortest path for the packaged demo and the blank custom-workspace flow.
- [Run the packaged promoter-swap demo](../workspaces/demo_promoter_swap_pdual10/README.md): inspect the packaged workspace, then use the [demo runbook](../workspaces/demo_promoter_swap_pdual10/runbook.md) to seed, validate, dry-run, and materialize it.
- [Run the packaged source-of-truth demo](../workspaces/demo_promoter_swap_pdual10_source_of_truth/README.md): inspect the packaged shared-dataset workspace, then use its [runbook](../workspaces/demo_promoter_swap_pdual10_source_of_truth/runbook.md) to materialize one consolidated USR dataset before switching to the shared infer and downstream handoff runbooks.
- [Create a custom construct workspace](../workspaces/README.md): scaffold a blank workspace, then use the [seed/import manifest reference](reference/seed-manifest.md), [workspace registry reference](reference/workspace-registry.md), and [config reference](reference/config.md) to define the study.
- [Run a workspace project by registry id](reference/workspace-registry.md): use the registry-backed workflow in the [CLI reference](reference/cli.md) and verify outputs with the [outputs reference](reference/outputs.md).
- [Assemble multiple upstream USR datasets before construct](../../usr/docs/operations/multi-source-source-of-truth-assembly.md): use the shared USR-owned runbook when control promoters, DenseGen outputs, or other USR-backed sources must be consolidated before construct reads them.
- [Hand off a construct-backed source-of-truth dataset to infer](../../usr/docs/operations/construct-infer-source-of-truth-runbook.md): use the shared cross-tool runbook when one USR dataset should stay canonical across construct, infer, and downstream event consumers.
- [Build a downstream promoter feature matrix for cluster or OPAL](../../usr/docs/operations/promoter-characterization-feature-matrix.md): use the shared USR-owned runbook when construct-expanded contexts should be compared against anchor-only promoters through infer-derived feature columns.
- [Debug a construct run](reference/cli.md): start with runtime validation, then check the [config reference](reference/config.md), [workspace registry reference](reference/workspace-registry.md), and [outputs reference](reference/outputs.md).

### Read by document type

- [Docs index](index.md): compact by-type map.
- [Getting started](getting-started.md): shortest runnable path.
- [CLI reference](reference/cli.md): command surface and failure posture.
- [Config reference](reference/config.md): job shape, placement rules, and output policy.
- [Outputs reference](reference/outputs.md): write behavior and lineage surfaces.
- [Seed/import manifest reference](reference/seed-manifest.md): import schema for your own input and template records.
- [Workspace registry reference](reference/workspace-registry.md): project inventory and registry-backed execution.
- [Developer notes](dev/README.md): maintainer notes and journal entries.

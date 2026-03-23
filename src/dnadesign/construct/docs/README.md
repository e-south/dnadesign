## Construct docs

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-20

Construct places anchor sequences or other focal DNA parts into explicit larger sequence contexts, then hands those realized sequences to downstream tools through stable USR datasets and `construct__*` lineage. Use this page to choose the next Construct task.

### Start here

- [Get to a first dry-run](getting-started.md): shortest path for the packaged demo and the blank custom-workspace flow.
- [Run the packaged local demo](../workspaces/demo_anchor_template_local/README.md): inspect the packaged anchor/template workspace, then use the [demo runbook](../workspaces/demo_anchor_template_local/runbook.md) to seed, validate, dry-run, and materialize it.
- [Run the packaged shared-dataset demo](../workspaces/demo_anchor_template_shared_dataset/README.md): inspect the packaged accumulation workspace, then use its [runbook](../workspaces/demo_anchor_template_shared_dataset/runbook.md) to materialize one consolidated USR dataset before switching to the shared infer and downstream handoff runbooks.
- [Create a custom Construct workspace](../workspaces/README.md): scaffold a blank workspace, then use the [seed/import manifest reference](reference/seed-manifest.md), [workspace registry reference](reference/workspace-registry.md), and [config reference](reference/config.md) to define the study.
- [Run a workspace project by registry id](reference/workspace-registry.md): use the registry-backed workflow in the [CLI reference](reference/cli.md) and verify outputs with the [outputs reference](reference/outputs.md).
- [Assemble multiple upstream USR datasets before Construct](../../usr/docs/operations/multi-source-shared-dataset-assembly.md): use the shared USR-owned runbook when DenseGen outputs, manual imports, or other USR-backed sources must be consolidated before Construct reads them.
- [Hand off a Construct-backed shared dataset to infer](../../usr/docs/operations/construct-infer-shared-dataset-runbook.md): use the shared cross-tool runbook when one USR dataset should stay shared across Construct, infer, and downstream event consumers.
- [Continue into a promoter-study feature dataset for Cluster or OPAL](../../usr/docs/operations/promoter-characterization-feature-matrix.md): use the shared USR-owned runbook when construct-expanded contexts should be compared against anchor-only promoter records through infer-derived feature columns.
- [Debug a Construct run](reference/cli.md): start with runtime validation, then check the [config reference](reference/config.md), [workspace registry reference](reference/workspace-registry.md), and [outputs reference](reference/outputs.md).

### Key docs

- [Getting started](getting-started.md): shortest runnable path for demos and blank workspaces.
- [Workspaces guide](../workspaces/README.md): scaffold, inspect, and organize workspace-local studies.
- [CLI reference](reference/cli.md): command surface and failure posture.
- [Config reference](reference/config.md): job shape, placement rules, and output policy.
- [Template/context contract](reference/template-contexts.md): the `construct__*` fields downstream infer uses for anchor-aware pooling.
- [Outputs reference](reference/outputs.md): write behavior and lineage surfaces.
- [Seed/import manifest reference](reference/seed-manifest.md): import schema for your own input and template records.
- [Workspace registry reference](reference/workspace-registry.md): project inventory and registry-backed execution.
- [Developer notes](dev/README.md): maintainer notes and journal entries.

### Boundary reminders

- `construct` owns sequence realization, placement semantics, and `construct__*` lineage.
- USR owns dataset persistence, dataset ids, and downstream reuse.
- One Construct job uses one template plus one or more placed parts.
- Larger studies stay explicit as multiple workspace projects, not one oversized config.
- Packaged workspaces default to workspace-local `outputs/usr_datasets`; shared USR roots are always explicit.

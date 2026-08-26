---
doc_id: construct-docs
title: Construct documentation
owner: dnadesign-maintainers
status: active
last_verified: 2026-08-21
---

# Construct documentation

Construct places declared DNA parts into explicit sequence contexts and records
the realized spans and lineage. Most users start with a workspace. Use the
workspace-less `compose` route only when the parts are already chosen and the
intended product is linear ssDNA.

### Start here

- [Get to a first dry-run](getting-started.md): shortest path for the packaged demo and the blank custom-workspace flow.
- [Run the packaged local demo](../workspaces/demo_anchor_template_local/README.md): inspect the packaged anchor/template workspace, then use the [demo runbook](../workspaces/demo_anchor_template_local/runbook.md) to seed, validate, dry-run, and materialize it.
- [Run the packaged shared-dataset demo](../workspaces/demo_anchor_template_shared_dataset/README.md): inspect the packaged accumulation workspace, then use its [runbook](../workspaces/demo_anchor_template_shared_dataset/runbook.md) to materialize one consolidated USR dataset before switching to the shared infer and downstream handoff runbooks.
- [Create a custom Construct workspace](../workspaces/README.md): scaffold a blank workspace, then use the [seed/import manifest reference](reference/seed-manifest.md), [workspace registry reference](reference/workspace-registry.md), and [config reference](reference/config.md) to define the project.
- [Run a workspace project by registry id](reference/workspace-registry.md): use the registry-backed workflow in the [CLI reference](reference/cli.md) and verify outputs with the [outputs reference](reference/outputs.md).
- [Assemble multiple upstream USR datasets before Construct](../../usr/docs/operations/assembly/multi-source-shared-dataset.md): use the shared USR-owned runbook when DenseGen outputs, manual imports, or other USR-backed sources must be consolidated before Construct reads them.
- [Hand off a Construct-backed shared dataset to infer](../../usr/docs/operations/assembly/construct-infer-shared-dataset-runbook.md): use the shared cross-tool runbook when one USR dataset should stay shared across Construct, infer, and downstream event consumers.
- [Debug a Construct run](reference/cli.md): start with runtime validation, then check the [config reference](reference/config.md), [workspace registry reference](reference/workspace-registry.md), and [outputs reference](reference/outputs.md).
- [Compose a declared linear ssDNA product](reference/linear-ssdna-composition.md): use the specialized workspace-less route when sequence parts are already selected.

### Key docs

- [Getting started](getting-started.md): shortest runnable path for demos and blank workspaces.
- [Workspaces guide](../workspaces/README.md): scaffold, inspect, and organize workspace-local projects.
- [CLI reference](reference/cli.md): command surface and failure posture.
- [Config reference](reference/config.md): job shape, placement rules, and output policy.
- [Linear ssDNA composition](reference/linear-ssdna-composition.md): `construct compose` command route, bundle layout, and Folding/BaseRender handoffs.
- [Annotated sequence-part placement](reference/annotated-sequence-parts.md): place one producer-authored sequence with nested digest-linked features without re-deriving it.
- [Template/context contract](reference/template-contexts.md): the `construct__*` fields downstream infer uses for anchor-aware pooling.
- [Outputs reference](reference/outputs.md): write behavior and lineage surfaces.
- [Seed/import manifest reference](reference/seed-manifest.md): import schema for your own input and template records.
- [Workspace registry reference](reference/workspace-registry.md): project inventory and registry-backed execution.
- [Developer notes](dev/README.md): maintainer notes and journal entries.

### Boundary reminders

- `construct` owns sequence realization, placement semantics, and `construct__*` lineage.
- USR owns dataset persistence, dataset ids, and downstream reuse.
- One Construct job uses one template plus one or more placed parts.
- Larger workflows stay explicit as multiple workspace projects, not one oversized config.
- Packaged workspaces default to workspace-local `outputs/usr_datasets`; shared USR roots are always explicit.
- `construct compose` writes caller-chosen local artifact bundles and does not create or register Construct workspaces.
- Annotated-part placement preserves producer-owned sequence and feature authority; Construct owns only placement and coordinate transformation.
- Provider and study names belong in caller-owned provenance values, not in the shared composition schema.

### Cross-tool examples

Use these only after the generic Construct flow is clear:

- [Construct → USR → Infer shared dataset](../../usr/docs/operations/assembly/construct-infer-shared-dataset-runbook.md)
- [Promoter characterization feature matrix](../../usr/docs/operations/promoter/characterization-feature-matrix.md)

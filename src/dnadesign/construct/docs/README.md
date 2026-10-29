<!--
**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-28
-->

[Construct](../README.md)

# Construct documentation

Construct combines input DNA parts with a template and records where each part
lands in the result. A workspace keeps the input references, placement settings
and output locations together. Start with the demo, then adapt a workspace for
your own sequences.

## Start here

1. [Run your first construct](getting-started.md): install the tool, preview a
   small example and inspect its saved sequences.
2. [Choose a workspace](../workspaces/README.md): use a packaged example or
   prepare your own inputs and placement configuration.

The [local demo](../workspaces/demo_anchor_template_local/README.md) writes
separate datasets for each placement. The
[shared-dataset demo](../workspaces/demo_anchor_template_shared_dataset/README.md)
combines two placements in one output dataset. Both keep their data inside the
copied workspace.

If you already have the segments for a linear ssDNA product, use
[sequence composition](reference/linear-ssdna-composition.md) to write a local
bundle without setting up a workspace or SeqLedger dataset.

## Find a setting or output

- [Configuration](reference/config.md): input records, templates, placement and window rules.
- [CLI commands](reference/cli.md): validation, execution, retries and workspace inspection.
- [Import inputs](reference/seed-manifest.md): create SeqLedger records from a sequence manifest.
- [Workspace registry](reference/workspace-registry.md): name projects and check that their configuration agrees with the registry.
- [Outputs and lineage](reference/outputs.md): inspect saved sequences, part coordinates and source identities.
- [Template contexts](reference/template-contexts.md): coordinates and metadata used by downstream consumers.
- [Annotated sequence parts](reference/annotated-sequence-parts.md): place a part while preserving its nested features.

## Connect other tools

Construct handles sequence placement and coordinate mapping. SeqLedger stores
its input and output datasets. Other tools can then read the assembled sequence
and named part spans without reconstructing the placement.

- [Combine upstream datasets](../../seqledger/docs/operations/assembly/multi-source-shared-dataset.md) before construction.
- [Pass Construct outputs to Infer](../../seqledger/docs/operations/assembly/construct-infer-shared-dataset-runbook.md) for sequence-model calculations.
- [Promoter characterization example](../../seqledger/docs/operations/promoter/characterization-feature-matrix.md) for a worked cross-tool application.

## Maintain or extend Construct

[Developer guidance](dev/README.md) links the public Python API, implementation
map and tests. Study-specific sequences and placement choices belong in input
files and configuration.

<!--
**Type:** route
**Plane:** data-plane
**Owner-boundary:** seqledger
**Entry artifact:** SeqLedger dataset, sync, or cross-tool shared-dataset work that still needs a route
**Exit artifact:** SeqLedger runbook or downstream handoff route
**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-29
-->

[SeqLedger guide](../README.md)

# SeqLedger operations runbooks

Choose the procedure that matches the dataset change you need. For a first local import and annotation, use the [quickstart](../getting-started/cli-quickstart.md).

## Transfer data

- [Sync over SSH](sync/README.md): set up a remote, compare copies and transfer in one direction.
- [Sync audit output](sync/audit-loop.md): inspect machine-readable transfer decisions.
- [Iterative HPC workflow](sync/hpc-agent-flow.md): coordinate remote production with local analysis.

## Combine or annotate datasets

- [Multi-source assembly](assembly/multi-source-shared-dataset.md): merge selected records and explicitly carry compatible annotations.
- [Construct to Infer](assembly/construct-infer-shared-dataset-runbook.md): assemble sequence contexts and use the resulting dataset for inference.
- [Permuter to Construct and Infer](assembly/permuter-construct-infer-shared-dataset.md): connect variant records to context realization and feature extraction.
- [Promoter feature-matrix example](promoter/characterization-feature-matrix.md): a worked application of these steps before clustering or selection.

## Maintain and verify

[Maintenance](../reference/maintenance.md) covers snapshots, compaction, merging and materialization. The [workflow map](routes/workflow-map.md) distinguishes those operations from routine reads and exports.

For contributors changing transfer behavior, [sync fidelity drills](sync/fidelity-drills.md) exercise failure paths and data checks. These are development checks, not prerequisites for using a dataset.

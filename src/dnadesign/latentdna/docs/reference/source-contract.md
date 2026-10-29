<!--
**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-28
-->

[LatentDNA guide](../README.md) / [Reference](README.md)

# Input sources

A source declares where data live and which columns identify rows and subjects. A view then selects the numerical representation to analyze. LatentDNA supports five source kinds:

| `kind` | Input |
| --- | --- |
| `parquet` | A Parquet table with vector or metadata columns |
| `matrix_bundle` | A numeric matrix and a matching row ledger |
| `usr` | A SeqLedger dataset, selected by `root` and `dataset` |
| `infer_feature_sidecar` | Infer vectors and their sequence-view aliases |
| `infer_feature_scalar_sidecar` | Infer scalar results and their aliases |

## Row and subject identity

`record_key` identifies an input row. `subject_key` identifies the entity to match across views, such as a sequence represented in two contexts. They may be the same column when each subject has one row. If several rows share a subject, declare the alignment and aggregation policy explicitly instead of relying on row order.

The [local walkthrough](../getting-started/README.md) shows a Parquet source. Relative source paths resolve from the workspace directory.

## Matrix bundles

A `matrix_bundle` directory contains `rows.parquet` and either `matrix.npy` or `matrix.npz`. Both key columns must exist in the row ledger, and its row count must match the matrix. Use `vector.kind: bundle_matrix`; a column selector is invalid for this source kind.

Materialization writes a workspace-owned matrix rather than using the external bundle as a mutable output. Keep the input ledger with the matrix so sequence identity is preserved.

## Infer sidecars

Infer vector and scalar sources join feature aliases, committed payload parts, sequence views and SeqLedger rows. They expose the numerical output as `value` and retain alias-level metadata; several aliases can reuse one payload without collapsing their distinct rows.

Use `read_mode: live` for data that may be written by Infer. Use `read_mode: snapshot` only for an immutable transferred copy, including a read-only input. Snapshot reads create no lock file and must not overlap writes or compaction. Committed parts are checked against manifest hashes, schemas and row counts; single-file payloads need external transfer checksums.

Missing sidecar files may describe a planned, empty source. Existing aliases that refer to absent payload keys are errors. See the [workspace schema](workspace-schema.md#sources-and-sequence-semantics) for fields and [views](view-contract.md) for materialization rules.

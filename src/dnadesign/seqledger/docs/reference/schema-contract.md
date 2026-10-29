<!--
**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-28
-->

[SeqLedger guide](../README.md) / [Reference](README.md)

# Sequence identity and schema

Every record has a stable sequence ID and seven required columns. Annotations add namespaced columns without changing that identity.

## Core schema

All columns below are non-null.

| Column | Type | Meaning |
| --- | --- | --- |
| `id` | string | SHA-1 of UTF-8 `bio_type\|sequence_norm` |
| `bio_type` | string | `dna`, `rna` or `protein` |
| `sequence` | string | Sequence text, with case preserved |
| `alphabet` | string | `dna_4`, `dna_5`, `rna_4`, `rna_5`, `protein_20` or `protein_21` |
| `length` | int32 | Length of `sequence_norm` |
| `source` | string | Import source label |
| `created_at` | timestamp(us, UTC) | Import time |

For hashing, `sequence_norm` is `sequence.strip()`. Leading and trailing whitespace is removed; case is preserved. Thus `ACGT` and `acgt` have different IDs, as do identical strings with different biological types. `bio_type` cannot contain `|`. Import duplicate handling and maintenance deduplication are separate from this identity rule.

## Annotation columns

Annotations use `<namespace>__<field>`, such as `demo__score`. A namespace must match `^[a-z][a-z0-9_]*$`, and its columns must be declared in the root registry. Annotation columns cannot replace core columns. `usr` and `usr_state` are reserved for system operations; use the dedicated deletion and state commands for those values.

CLI attachment rejects conflicting values unless `--allow-overwrite` is explicit. Python writers have their own overwrite arguments; choose the policy explicitly when updating annotations. See [overlays and the registry](overlay-and-registry.md).

Shared annotation conventions:

| Namespace | Purpose | Example columns |
| --- | --- | --- |
| `usr_label` | Record names and aliases | `usr_label__primary`, `usr_label__aliases` |
| `seq_annot` | Imported source annotations | `seq_annot__source_sha256`, `seq_annot__record_id`, `seq_annot__features` |
| `derived` | Parent-child lineage and retained annotations | `derived__parent_id`, `derived__product_kind`, `derived__source_interval_*`, `derived__features_*` |

`seq_annot__features` preserves source feature order, raw location text and repeated qualifiers. Where possible, it also records zero-based, half-open intervals, with confidence information for fuzzy or unsupported locations. The `derived__features_retained`, `derived__features_clipped` and `derived__features_lost` columns describe how those annotations carry into a derived sequence.

## Dataset names

Choose a dataset ID that identifies the collection, such as `reference_sequences`. Flat IDs and namespace-qualified IDs are supported. Keep record labels in `usr_label` and producing-tool provenance in annotation columns rather than relying on the dataset name to encode them.

## Parquet metadata

Configuration kind `usr`, metadata keys beginning with `usr:`, and columns beginning with `usr_label__` or `usr_state__` are stable format identifiers used by SeqLedger.

Base tables carry `usr:schema_version`, `usr:dataset_created_at`, `usr:id_hash` and `usr:registry_hash`. Overlays carry `usr:overlay_namespace`, `usr:overlay_key`, `usr:overlay_created_at`, `usr:registry_hash` and `usr:namespace_contract_hash`. These identify the format and registry contracts used to write each file.

## Sidecars and semantic aliases

A sequence can have several roles while retaining the same literal DNA, RNA or protein string. Sequence views record those roles separately from the base table:

- `_views/sequence_views.parquet` stores view identity, orientation, parent lineage, source intervals, anchor bounds and recommended pooling metadata.
- `view_id` identifies a sequence view. Several views may refer to one base record's `id`.
- `_views/view_semantics.parquet` optionally stores mutable metadata keyed by `view_id`, including `source_family`, `selection_basis`, `view_collections` and `role_tags`.
- Human labels, aliases and collection membership do not determine stable view identity.

| Product kind | Meaning |
| --- | --- |
| `source_record` | Imported source record |
| `selected_region` | Region selected from a source record |
| `construct_insert` | Sequence used as an insert in a construct |
| `analysis_window` | Sequence window derived for analysis |
| `realized_context` | Sequence emitted after constructing a context around an insert |

Product kind describes lineage and use, not a fixed length, orientation or experimental group. An `analysis_window` is not necessarily 60 bp, and a `realized_context` is not necessarily 1 kb. Record those dimensions and relationships in their dedicated fields. Use the [public readers](python-api.md#read-sequence-views) instead of reading sidecars directly.

## Continue

- [Overlays and the registry](overlay-and-registry.md) explain annotation writes and merge order.
- [Event log](event-log.md) documents mutation records.
- [Dataset layout](dataset-layout-and-code-map.md) shows where each artifact is stored.

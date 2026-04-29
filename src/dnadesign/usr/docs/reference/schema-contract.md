# USR schema contract

**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-27


## Core schema

| column | type | notes |
| --- | --- | --- |
| `id` | string | `sha1(UTF-8 bio_type\|sequence_norm)` |
| `bio_type` | string | `dna` \| `rna` \| `protein` |
| `sequence` | string | case-preserving, normalized by trim for hashing |
| `alphabet` | string | `dna_4`, `dna_5`, `rna_4`, `rna_5`, `protein_20`, `protein_21` |
| `length` | int32 | `len(sequence_norm)` |
| `source` | string | ingest provenance |
| `created_at` | timestamp(us, coordinated universal time) | ingest time |

`sequence_norm` is `sequence.strip()` and is the value used for id hashing. `bio_type` must not contain `|`.

## Required columns (non-null)

| column | type | description |
| --- | --- | --- |
| `id` | `string` | `sha1(UTF-8 bio_type\|sequence_norm)` |
| `bio_type` | `string` | one of `dna`, `rna`, `protein` |
| `sequence` | `string` | raw sequence (case preserved) |
| `alphabet` | `string` | `dna_4`, `dna_5`, `rna_4`, `rna_5`, `protein_20`, `protein_21` |
| `length` | `int32` | `len(sequence_norm)` |
| `source` | `string` | source label or file |
| `created_at` | `timestamp[us, coordinated universal time]` | ingest time |

## Derived columns

- Must be namespaced `<namespace>__<field>`.
- Must not overlap essential columns.
- Namespace regex: `^[a-z][a-z0-9_]*$`
- Reserved namespaces: `usr`
- Collision policy: hard error unless `--allow-overwrite` is explicit.

Conventional cross-tool label namespace:

- Use `usr_label__primary` for the preferred human-readable record name.
- Use `usr_label__aliases` for stable alternate names or legacy labels.
- Use tool-specific namespaces for tool provenance, not for the canonical human label itself.

Additive shared namespaces used by the current sequence-product contract:

| namespace | purpose | notable columns |
| --- | --- | --- |
| `usr_label` | canonical human-readable names and aliases | `usr_label__primary`, `usr_label__aliases` |
| `seq_annot` | source-faithful imported annotation overlays | `seq_annot__source_sha256`, `seq_annot__record_id`, `seq_annot__features` |
| `derived` | parent/child lineage and derived-product semantics | `derived__parent_id`, `derived__product_kind`, `derived__source_interval_*`, `derived__features_*` |

`seq_annot__features` preserves imported feature order, raw location text, normalized 0-based half-open intervals where available, qualifier multiplicity, and confidence for fuzzy or unsupported source locations.

`derived__features_retained`, `derived__features_clipped`, and `derived__features_lost` summarize how parent annotations survive derived sequence products such as `analysis_window` windows or template-expanded contexts.

Dataset id naming convention:

- Prefer the least-coupled semantic id that still makes the dataset obvious to operators.
- Flat ids are first-class: examples include `usr_promoter_references`, `usr_pdual10_plasmid_template`, and `anchor_template_slot_a_window_1kb_demo`.
- Namespace-qualified ids remain valid when they genuinely improve disambiguation.
- Avoid encoding the producing tool name in the dataset id when the dataset is intended to be consumed across sibling packages.

## Base table metadata

Parquet key/value metadata:

- `usr:schema_version`
- `usr:dataset_created_at`
- `usr:id_hash`
- `usr:registry_hash`

## Overlay metadata

Parquet key/value metadata for derived overlays:

- `usr:overlay_namespace`
- `usr:overlay_key`
- `usr:overlay_created_at`
- `usr:registry_hash`
- `usr:namespace_contract_hash`

## Sidecars and semantic aliases

USR keeps the base `records.parquet` schema stable and stores richer semantic view identity in additive sidecars.

- Dataset-local sequence views live at `_views/sequence_views.parquet`.
- Optional mutable view semantics live at `_views/view_semantics.parquet`.
- `view_id` identifies a semantic sequence product/view, not a base sequence row.
- Multiple `view_id` values may point to the same `id` in `records.parquet`.
- Sequence views carry product kind, orientation, parent lineage, optional source intervals, optional emitted anchor bounds, and recommended pooling metadata.
- Human labels and aliases are mutable metadata on the view row; they are not part of the stable semantic hash.
- The view-semantics addendum is keyed by `view_id` and can carry `source_family`,
  `selection_basis`, `view_collections`, and `role_tags` without changing stable
  view identity. It is the right place for study cohort membership; do not
  encode cohort membership by inventing misleading `product_kind` values.

This separation allows provenance-faithful native records, analysis-only cores, and forward or reverse-complement construct contexts to alias the same base sequence when the literal sequence string collides.

Current product-kind vocabulary:

- `source_record`: source-faithful imported record.
- `selected_region`: explicitly selected biological insert from a source record.
- `construct_insert`: construct-ready promoter/anchor row in a merged handoff dataset.
- `analysis_window`: Construct-derived 60 bp analysis-only comparability view.
- `realized_context`: Construct-realized emitted context; `orientation` and
  length distinguish forward 1 kb pDual contexts from reverse-complement
  contexts.

Product kind describes the sequence product's lineage and semantics, not every
downstream analysis cohort. A natively 60 bp promoter insert remains a
`construct_insert`; it must not be duplicated or relabeled as `analysis_window`
solely because it is 60 bp. `analysis_window` is reserved for rows Construct
derived by trimming/windowing or template-expanding around an explicit focal
rule. Exact 60 bp `seq_mean` and `core60_mean` equivalence belongs to Infer's
feature-alias layer, not to USR base rows.

## Next steps

- Overlay merge and registry behavior: [overlay-and-registry.md](overlay-and-registry.md)
- Event contract: [event-log.md](event-log.md)

# USR schema contract

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-14


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

Dataset id naming convention:

- Prefer the least-coupled semantic id that still makes the dataset obvious to operators.
- Flat ids are first-class: examples include `mg1655_promoters`, `plasmids`, and `pdual10_slot_a_window_1kb_demo`.
- Namespace-qualified ids remain valid when they genuinely improve disambiguation.
- Avoid encoding the producing tool name in the dataset id when the dataset is intended to be consumed across sibling packages.

## Base table metadata

Parquet key/value metadata:

- `usr:schema_version`
- `usr:dataset_created_at`
- `usr:id_hash`
- `usr:registry_hash`

## Next steps

- Overlay merge and registry behavior: [overlay-and-registry.md](overlay-and-registry.md)
- Event contract: [event-log.md](event-log.md)

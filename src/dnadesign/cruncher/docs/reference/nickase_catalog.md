## Nickase catalog reference

**Owner:** dnadesign-maintainers
**Doc kind:** reference
**Audience:** cassette workflow users and maintainers
**Updated by:** cruncher-maintainers on 2026-03-24
**Applies to:** workspace-local cassette nickase catalogs
**Last verified:** 2026-03-24
**Primary artifacts:** validated catalog entries used by `cassette validate|design`

### Contents
- [File shape](#file-shape)
- [Entry semantics](#entry-semantics)
- [Validation rules](#validation-rules)
- [Example](#example)

### File shape

Nickase catalogs are local YAML files loaded from `cassette.catalog.path`:

```yaml
nickases:
  schema_version: 1
  entries:
    - id: nb_left
      recognition_sequence: AACGA
      nicked_site_strand: forward
      cut_offset: 2
```

There is no remote fetch path in v1. The workflow reads the exact catalog file referenced by the spec and snapshots it into the run directory.

### Entry semantics

- `schema_version`: must be `1`.
- `entries`: non-empty list of nickase entries with unique `id` values.
- `id`: stable identifier referenced by `cassette.nicking.left.nickase` and `cassette.nicking.right.nickase`.
- `recognition_sequence`: canonical forward-strand recognition site, `A/C/G/T` only.
- `nicked_site_strand`: which strand of the canonical recognition site is nicked: `forward` or `reverse`.
- `cut_offset`: zero-based offset on the nicked site strand, measured from that strand's 5' start across the recognition site.
- `source`, `vendor`, `notes`, `tags`: optional metadata preserved in the catalog but not required for planning.

Cruncher scans both the canonical recognition sequence and its reverse complement in the duplex context. It projects
`nicked_site_strand` and `cut_offset` into the requested duplex strand coordinates automatically.

### Validation rules

- `recognition_sequence` must be asymmetric in v1. Palindromic sequences are rejected because strand-specific nick semantics would be ambiguous.
- `cut_offset` must be between `0` and the recognition sequence length, inclusive.
- duplicate `id` values are rejected.
- malformed YAML or non-mapping top-level payloads fail fast.

### Example

```yaml
nickases:
  schema_version: 1
  entries:
    - id: nb_left
      recognition_sequence: AACGA
      nicked_site_strand: forward
      cut_offset: 2
      source: local_demo
      vendor: internal
      notes: Left-flank nickase for demo cassette
    - id: nb_right
      recognition_sequence: AACGA
      nicked_site_strand: reverse
      cut_offset: 3
      source: local_demo
      vendor: internal
      tags:
        flank: right
```

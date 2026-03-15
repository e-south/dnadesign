## construct seed/import manifest reference

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-14

### Purpose

Use `construct seed import-manifest` to materialize canonical anchor or template records into one or more USR datasets before you write construct configs.

### Shape

```yaml
manifest_id: example_construct_inputs
datasets:
  - id: example_anchors
    notes: Example anchor inputs for a custom construct study.
    records:
      - label: example_anchor
        role: anchor
        topology: linear
        aliases: [example_anchor_alias]
        source_ref: replace-with-canonical-source
        sequence: ACGTACGT
  - id: example_templates
    notes: Example template records for a custom construct study.
    records:
      - label: example_template
        role: template
        topology: circular
        aliases: [example_template_alias]
        source_ref: replace-with-canonical-source
        sequence: AAAATTTTCCCCGGGG
```

### Contract

- `manifest_id`: required stable id for the import batch
- `datasets`: required non-empty list
- `datasets[].id`: semantic USR dataset id, preferably flat and biological
- `datasets[].records`: required non-empty list
- `records[].label`: preferred human-readable record name
- `records[].role`: intended construct role such as `anchor` or `template`
- `records[].topology`: free text today, but use stable values like `linear` or `circular`
- `records[].sequence`: required DNA sequence
- `records[].aliases`: optional alternate labels
- `records[].source_ref`: optional canonical source note or local provenance hint

### What gets written

- base USR records in the requested dataset ids
- `usr_label__primary` / `usr_label__aliases` for human-readable names
- `construct_seed__label`
- `construct_seed__manifest_id`
- `construct_seed__role`
- `construct_seed__source_ref`
- `construct_seed__topology`
- `construct_seed__sha256`

### Failure posture

- malformed YAML: fail before any dataset mutation
- empty dataset/record lists: fail before any dataset mutation
- blank ids, labels, roles, or topology: fail before any dataset mutation
- duplicate dataset ids inside one manifest: fail before any dataset mutation
- invalid DNA sequences: fail before any dataset mutation

### CLI

```bash
uv run construct seed import-manifest \
  --manifest inputs/import_manifest.yaml \
  --root outputs/usr_datasets
```

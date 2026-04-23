## Release-enzyme catalogs

**Owner:** dnadesign-maintainers
**Doc kind:** reference
**Audience:** snapback workflow users and maintainers
**Last updated by:** cruncher-maintainers on 2026-04-21
**Applies to:** workspace-local and built-in release-enzyme catalogs used by released-product snapback workflows
**Last verified:** 2026-04-21
**Primary artifacts:** validated release-enzyme catalog entries used by `snapback released-design|released-target-search`

### Contents
- [File shape](#file-shape)
- [Entry semantics](#entry-semantics)
- [Built-in preset](#built-in-preset)
- [Validation rules](#validation-rules)

### File shape

Release-enzyme catalogs use a separate top-level root from nickase catalogs:

```yaml
release_enzymes:
  schema_version: 1
  entries:
    - variant_id: BsaI-HFv2
      display_name: BsaI-HFv2
      recognition_sequence: GGTCTC
      top_cut_offset: 7
      bottom_cut_offset: 11
      class_label: type_iis
      commercial_confidence: primary_vendor_current
      warning_codes: [FLANKING_BASE_RECOMMENDED]
      recommended_5prime_flanking_bases: 6
      source_catalog_id: type_iis_release_v1
```

The normalized cut offsets are zero-based cut boundaries relative to the oriented recognition-site start. Cruncher scans both the recorded motif and its reverse complement, then derives the strand-pair cut coordinates from that normalized representation.

### Entry semantics

- `schema_version`: must be `1`
- `entries`: non-empty list of release-enzyme entries with unique `variant_id` values
- `variant_id`: stable release-enzyme identifier referenced by the released-product spec
- `display_name`: operator-facing enzyme name
- `recognition_sequence`: top-strand recognition motif; IUPAC symbols are allowed
- `top_cut_offset`: required top-strand cut boundary relative to the oriented site start
- `bottom_cut_offset`: required bottom-strand cut boundary relative to the oriented site start
- `class_label`: typed family label such as `type_iis`, `type_iia`, or `other_ds_re`
- `outside_site`: derived property, not an authored field
- `commercial_confidence`: typed provenance rank used in deterministic release-enzyme ordering
- `warning_codes`: typed warning inventory such as `FLANKING_BASE_RECOMMENDED`, `CUT_POSITION_VARIABILITY_RISK`, `METHYLATION_SENSITIVITY`, or `STAR_ACTIVITY_RISK`
- `recommended_5prime_flanking_bases`: optional operator hint preserved in normalized metadata
- `source_catalog_id`: source catalog identifier for the entry
- `source_url`: optional vendor or source provenance URL

Release-enzyme catalogs are intentionally separate from nickase catalogs because they model ds cuts, retained-side semantics, and paired cut coordinates rather than single-strand nick events.

### Built-in preset

The v1 built-in preset is `type_iis_release_v1`.

Current shipped entries:

- `BsaI-HFv2`
- `BsmBI-v2`
- `BbsI`
- `SapI`
- `BspQI`

The preset is intentionally small and conservative. It is a release-enzyme seed set for the released-product lane, not a generic all-restriction-enzyme ecosystem.

### Validation rules

- `recognition_sequence` must contain valid IUPAC nucleotide symbols
- both `top_cut_offset` and `bottom_cut_offset` are required
- duplicate `variant_id` values are rejected
- duplicate preset ids are rejected
- preset and overlay merges append entries; duplicate release-enzyme IDs fail fast
- malformed YAML or non-mapping payloads fail fast
- at least one preset or additional path must be provided when resolving release sources

For the released-product workflow contract, use [`../guides/snapback_released_workflow.md`](../guides/snapback_released_workflow.md).

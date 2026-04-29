## Nickase catalog reference

**Owner:** dnadesign-maintainers
**Doc kind:** reference
**Audience:** cassette and snapback workflow users and maintainers
**Last updated by:** cruncher-maintainers on 2026-04-21
**Applies to:** workspace-local and built-in Cruncher nickase catalogs
**Last verified:** 2026-04-21
**Primary artifacts:** validated catalog entries used by `cassette validate|design` and `snapback validate|design|solve`

### Contents
- [File shape](#file-shape)
- [Entry semantics](#entry-semantics)
- [Legacy compatibility](#legacy-compatibility)
- [Validation rules](#validation-rules)

### File shape

Nickase catalogs are local YAML files loaded from `cassette.catalog.path`:

```yaml
nickases:
  schema_version: 1
  entries:
    - id: Nt.BbvCI
      specificity_id: BbvCI
      raw_cut_notation: CCTCAGC(-5/none)
      source: neb
    - id: Nb.BbvCI
      specificity_id: BbvCI
      raw_cut_notation: CCTCAGC(none/-2)
      source: neb
```

There is no remote fetch path in the tracer bullet. The workflow reads the exact catalog file referenced by the spec and snapshots it into the run directory.

Solve mode also supports built-in preset catalogs plus local overlays:

```yaml
catalog_version: 2
preset_id: neb_nicking_v1
preset_ids:
  - neb_nicking_v1
generated_from: NEB public nicking enzyme resources
generated_on: 2026-04-21
normalization_policy: raw_cut_notation_is_authoritative

variants:
  - id: Nt.BspQI
    specificity_id: BspQI
    motif_top_5to3: GCTCTTC
    raw_cut_notation: GCTCTTC(1/none)
    vendor: NEB
    selection:
      outside_site: true
      snapback_tier: tier1
      commercial_confidence: primary_vendor_current
    operational:
      incubation_temp_c: 50

product_aliases:
  - alias_id: WarmStart Nt.BstNBI
    alias_kind: formulation
```

Snapback specs can compose multiple built-in presets without exporting them first:

```yaml
catalog:
  preset: neb_nicking_v1
  additional_presets:
    - thermo_nicking_v1
  additional_paths: []
```

### Entry semantics

- `schema_version`: must be `1`.
- `entries`: non-empty list of nickase variant entries with unique `id` values.
- `id`: stable variant identifier referenced by `cassette.nicking.left.nickase` and `cassette.nicking.right.nickase`.
- `specificity_id`: recognition-specificity family identifier shared across related variants.
- `motif_top_5to3`: top-strand recognition motif. IUPAC nucleotide symbols are allowed.
- `raw_cut_notation`: optional vendor/source notation parsed into normalized offsets.
- `raw_cut_offset_reference`: optional coordinate basis for raw notation values.
  Use `motif_start` when the raw values are already normalized as bond-boundary
  offsets from the left edge of `motif_top_5to3`; use `motif_end` when vendor
  notation reports offsets relative to the 3' end of the listed recognition
  motif, such as `GCAGTG(none/0)` for a bottom nick immediately after the
  six-base `GCAGTG` site.
- `top_cut_offset`: signed bond-boundary offset from the motif start on the top strand.
- `bottom_cut_offset`: signed bond-boundary offset from the motif start on the bottom strand.
- Exactly one of `top_cut_offset` or `bottom_cut_offset` must be defined for a nickase variant.
- `source`: optional provenance string preserved in the report metadata.
- `vendor`, `vendor_catalog_number`, `origin_class`, `source_family`: typed provenance fields for the catalytic entry.
- `notes`: optional entry-scoped notes preserved alongside the normalized rule.
- `selection`: optional typed machine-readable selection metadata.
  Fields currently include `outside_site`, `snapback_tier`, `commercial_confidence`, and `warning_codes`.
- `operational`: optional typed bench-facing metadata.
  Fields currently include `incubation_temp_c`, `buffer_family`, `heat_inactivation`, `methylation_sensitivity`, `star_activity_warning`, and `double_strand_cleavage_warning`.
- `metadata`: optional free-form dictionary preserved as normalized catalog metadata.
- `product_aliases`: optional non-catalytic product/formulation aliases that point at the underlying catalytic variant.

Cruncher scans both the recorded motif and its reverse complement in the evaluated duplex context. It then derives the actual nicked strand and bond boundary from the normalized offset representation.

`neb_nicking_v1` is the built-in seed preset used by `cassette solve`. It ships the NEB nicking variants listed in the phase 2/3 cassette solve spec, including a `WarmStart Nt.BstNBI` formulation alias that resolves to the `Nt.BstNBI` catalytic rule.

`thermo_nicking_v1` is a separate built-in commercial overlay for the currently verified Thermo additions:

- `Nb.Mva1269I`
- `Nb.Bpu10I`
- `Nt.Bpu10I`

The separation is intentional: catalytic rule additions live in their own preset, while alternate product vendors stay modeled as aliases of the primary catalytic entry.

If you want a local copy of the packaged preset for inspection or overlay authoring, export it with:

```bash
uv run cruncher cassette catalog init-neb --output configs/catalogs/neb_nicking_v1.yaml
```

### Legacy compatibility

Legacy v1-style entries are still accepted and normalized:

```yaml
nickases:
  schema_version: 1
  entries:
    - id: nb_left
      recognition_sequence: AACGA
      nicked_site_strand: forward
      cut_offset: 2
```

The loader converts legacy fields into the normalized form:

- `recognition_sequence` -> `motif_top_5to3`
- `nicked_site_strand: forward` -> `top_cut_offset = cut_offset`
- `nicked_site_strand: reverse` -> `bottom_cut_offset = len(motif) - cut_offset`

Do not mix legacy geometry fields with normalized `top_cut_offset` or `bottom_cut_offset` in the same entry.

### Validation rules

- `motif_top_5to3` must contain valid IUPAC nucleotide symbols.
- `raw_cut_notation` must parse cleanly, for example `CCTCAGC(-5/none)` or `CCTCAGC(none/-2)`.
- `raw_cut_notation` must agree with any explicit `motif_top_5to3` or `recognition_sequence` present in the same entry.
- `raw_cut_offset_reference`, when present, must be `motif_start` or `motif_end`.
- exactly one of `top_cut_offset` or `bottom_cut_offset` must be populated after normalization.
- duplicate `id` values are rejected.
- duplicate `product_aliases.alias_id` values are rejected.
- `preset_ids` must be unique when present.
- preset + overlay merges append entries; duplicate variant IDs fail fast and do not silently override.
- malformed YAML or non-mapping top-level payloads fail fast.

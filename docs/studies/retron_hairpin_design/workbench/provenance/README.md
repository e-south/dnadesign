---
doc_id: study-retron-hairpin-design-workbench-provenance
surface: study-workbench-provenance
study_id: retron_hairpin_design
owner: dnadesign-maintainers
last_verified: 2026-07-08
plane: evidence-plane
surface_role: run-provenance-and-source-records
---

## Retron Workbench Provenance

**Owner:** dnadesign-maintainers
**Last verified:** 2026-07-08

Stores what was run against a workbench design set. It stores compact
manifests, command contracts, hashes, and output posture, not bulky generated
artifacts.

### Route By Record Type

| Need | Open |
| --- | --- |
| Catalog compile invocation and digest | [compiler_runs/](compiler_runs/README.md) |
| GenBank, plot, or sequence-bundle materialization posture | [materializations/](materializations/README.md) |
| Reader SPOP MSD-region source records, per-variant GenBank inputs, and pairing facts | `msd_region_records/reader_spop_msd_structure_panel_v1/` |

### Boundary

Run records cite design sets and compiler inputs. They do not replace
`../design_sets/` as the source of experimental meaning, and they do not replace
generated output bundles.

### MSD Region Source Contract

`msd_region_records/reader_spop_msd_structure_panel_v1/source_inputs/variants/`
is the active GenBank source layer for the Reader SPOP structure panel. Each file
contains one retron variant, and `source_inputs/variant_sources.yaml` records the
orientation rule, hashes, replacement inputs, and retired bulk source metadata.

The retired `msd-regions - all DNA RNA.gb` file is migration provenance only.
New ingest, plotting, and materialization code must read the per-variant
GenBank sources or decomposed records.

Variant YAML records separate:

- `annotation_warnings`: review-level source problems.
- `annotation_notes`: benign boundary normalization, such as narrow foldback
  annotations or payload spans that do not include adjacent stem context.
- `pairing_segments`: derived payload, stem-extension, and foldback pairing
  facts with Watson-Crick, wobble, mismatch, and intent fields.

### MSD Region Implementation Surface

The source-ingest code is split by record lifecycle:

- `source_ingest/msd_region_genbank.py`: public import boundary for callers.
- `source_ingest/genbank_bundle.py`: GenBank file and source-directory parsing.
- `source_ingest/variant_sources.py`: per-variant source manifests.
- `source_ingest/record_normalization.py`: normalized feature and record
  assembly.
- `source_ingest/annotation_review.py`: annotation notes and warning posture.
- `source_ingest/pairing_segments.py`: payload, stem-extension, and foldback
  pairing facts.
- `source_ingest/payload_catalog.py`, `payload_motifs.py`, and
  `payload_sites.py`: payload family lookup, motif scoring, and binding-site
  classification.
- `source_ingest/comparison.py`: drift checks against older materialized
  outputs.
- `source_ingest/bundle_writer.py`: generated YAML manifest and compiler-spec
  bundle writing.

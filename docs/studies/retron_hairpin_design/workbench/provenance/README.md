---
doc_id: study-retron-hairpin-design-workbench-provenance
surface: study-workbench-provenance
study_id: retron_hairpin_design
owner: dnadesign-maintainers
last_verified: 2026-08-01
plane: evidence-plane
surface_role: run-provenance-and-source-records
---

## Retron Workbench Provenance

**Owner:** dnadesign-maintainers
**Last verified:** 2026-08-01

Stores what was run against a workbench design set. It stores compact
manifests, command contracts, hashes, and output posture, not bulky generated
artifacts.

### Route By Record Type

| Need | Open |
| --- | --- |
| Catalog compile invocation and digest | [compiler_runs/](compiler_runs/README.md) |
| GenBank, plot, or sequence-bundle materialization posture | [materializations/](materializations/README.md) |
| Hairpin-owned MSD-region source records, per-variant GenBank inputs, and pairing facts | [`msd_region_records/retron_msd_structure_panel_v1/manifest.yaml`](msd_region_records/retron_msd_structure_panel_v1/manifest.yaml) |
| Design, compiler, primitive, and handoff lineage for selected variants 195–204 | [`pes_retron_195_204.yaml`](materialized_variant_lineage/pes_retron_195_204.yaml) |

### Boundary

Run records cite design sets and compiler inputs. They do not replace
`../design_sets/` as the source of experimental meaning, and they do not replace
generated output bundles.

### MSD Region Source Contract

`msd_region_records/retron_msd_structure_panel_v1/source_inputs/variants/`
is the authoritative GenBank source snapshot for the immutable, hairpin-owned
MSD structure-evidence bundle. Each file
contains one retron variant, and `source_inputs/variant_sources.yaml` records the
orientation rule, hashes, replacement inputs, and retired bulk source metadata.

Do not rewrite that directory. The ingest CLI uses the same neutral bundle and
compiler-spec identity. Later publications must mint another neutral revision
rather than overwrite this provenance bundle.

The retired `msd-regions - all DNA RNA.gb` file is migration provenance only.
New ingest, plotting, and materialization code must read the per-variant
GenBank sources or decomposed records.

Variant YAML records separate:

- `annotation_warnings`: review-level source problems.
- `annotation_notes`: benign boundary normalization, such as narrow foldback
  annotations or payload spans that do not include adjacent stem context.
- `pairing_segments`: derived payload, stem-extension, and foldback pairing
  facts with Watson-Crick, wobble, mismatch, and intent fields.

For the selected `pES-retron-195` through `pES-retron-204` cohort, start from
`materialized_variant_lineage/pes_retron_195_204.yaml`. Each row links the
variant to one design-set row, compiler row, deliverable assignment, source
construct, payload and structural primitive IDs, source GenBank, and normalized
MSD record. The projection points one way to the immutable historical
source-bundle manifest; the source bundle does not discover or authorize later
selections. The loader requires the declared selected IDs to match the entry IDs
exactly, permits unrelated source records, and verifies every referenced path
and sequence/file digest. Cruncher workspaces remain source references inside
the owning design sets; this projection does not copy their state or assay
interpretation.

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
- `source_ingest/selected_lineage/`: typed selected-cohort lineage and
  fail-fast selection, path, identity, primitive, and digest validation.
- `source_ingest/bundle_manifest.py`: portable manifest projection from
  normalized records.
- `source_ingest/bundle_writer.py`: validate-before-write bundle publication
  and compiler-spec emission.

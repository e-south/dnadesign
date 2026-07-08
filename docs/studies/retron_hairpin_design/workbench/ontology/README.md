---
doc_id: study-retron-hairpin-design-workbench-ontology
surface: study-workbench-ontology
study_id: retron_hairpin_design
owner: dnadesign-maintainers
last_verified: 2026-07-08
plane: knowledge-plane
surface_role: controlled-vocabulary
---

## Retron Workbench Ontology

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-18

Stores controlled vocabulary for workbench records. Hypotheses, directions, and
effect tags live here so design sets can cite stable identifiers instead of
repeating local prose.

### Records

- `directions.yaml`: experimental directions, effect-tag definitions, and the
  preferred display order for Retron MSD workbench records.
- `payload_binding_sites.yaml`: reusable payload families, motif models,
  retained-span members, and reference payloads used to derive binding-site
  semantics from MSD records.

### Boundary

Use this directory for reusable vocabulary only. Cohort membership belongs in
`../design_sets/`, and run-specific compiler or materialization evidence
belongs in `../provenance/`.

Payload families describe literal payload sources and motif-register semantics.
They do not declare whether a construct worked in Reader; assay evidence stays
in the RT-lnRNA/Reader SPOP records.

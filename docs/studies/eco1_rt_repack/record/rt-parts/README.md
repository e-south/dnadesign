---
doc_id: eco1-rt-repack-rt-parts
surface: study-record-provider-publication
study_id: eco1_rt_repack
owner: dnadesign-maintainers
last_verified: 2026-07-29
---

## Eco1 RT repack part publications

These files publish opaque references and canonical sequence identity metadata
for RT parts produced by this study. They publish no CDS bytes or internal
generated candidate ids. They do not make `eco1_rt_repack` the owner of WT,
literature-derived, point-mutant, fusion, or future RT parts from other
providers.

`eco1-g3-distal-pair-v1.yaml` is the minimal checked-in publication needed for
the current D01/D02 assay handoff. Its opaque source reference and source digest
pin the provider-owned Twist handoff without exposing its private path or
payload. The file conforms to the provider-neutral
`dnadesign.contracts.sequence.RtPartPublicationV1` handoff; Eco1 ownership,
canonical CDS/protein digests, and lengths remain provider-owned data rather
than consumer code. A consumer that needs sequence bytes must resolve the
declared `provider_ref` through an authorized provider-owned authority.

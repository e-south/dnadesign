## Generic Linear ssDNA Composition

**Status:** accepted implementation reference
**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-14
**Primary study:** `retron_hairpin_design`

This page is the entry point for the linear ssDNA composition work. Use it to
find the current contract, the implementation record, and the deeper design
history without starting from the full source spec.

### Use This Page When

- adding or reviewing Construct linear ssDNA composition behavior;
- routing Folding or BaseRender integration through producer-owned bundles;
- checking whether a Retron-specific choice belongs in study records or in a
  generic service contract;
- linking to the design without starting from a long historical document.

### Current Authority

- [Construct linear ssDNA composition reference](../../../src/dnadesign/construct/docs/reference/linear-ssdna-composition.md):
  current generic assembly contract and bundle layout.
- [ADR 0002](../../architecture/decisions/adr-0002-generic-linear-ssdna-composition.md):
  accepted architecture decision and ownership boundary.
- [Retron linear ssDNA composition handoff](../../studies/retron_hairpin_design/contexts/linear-ssdna-composition.md):
  Retron-specific study choices and caveats.
- [Folding docs](../../../src/dnadesign/folding/docs/README.md):
  secondary-structure prediction commands, bundle mode, and backend policy.

### Contract Boundary

The generic boundary is:

```text
Construct assembles linear sequence artifacts.
Folding predicts secondary structure from assembled artifacts.
BaseRender renders sequence evidence maps.
Study records own biological rationale and study-specific labels.
```

Construct validates sequence mechanics, span coverage, transforms, repeats,
provenance, and topology. It must not hard-code Retron, TetO, TetR, snapback,
or other study vocabulary. Publication labels, colors, and study-facing names
belong in declared display profiles or study records.

### Artifact Ontology

| Artifact | Owner | Purpose |
| --- | --- | --- |
| `linear_ssdna_composition_v1` | Construct | Ordered segment assembly, annotations, repeats, transforms, and provenance. |
| `secondary_structure_prediction_v1` | Folding | Backend-neutral structure prediction output. |
| `sequence_evidence_map_v1` | BaseRender/Folding consumers | Component evidence spans used for linear and structure visualization. |
| Study handoff records | Study docs or study package | Study-specific selection, rationale, and labels. |

### Reading Order

1. Start with this page for the boundary and artifact map.
2. Use the [Construct reference](../../../src/dnadesign/construct/docs/reference/linear-ssdna-composition.md)
   for current behavior.
3. Use the [implementation record](../../exec-plans/completed/2026-05-13-generic-linear-ssdna-composition.md)
   for what shipped.
4. Use the [hardening follow-up plan](../../exec-plans/active/2026-05-14-linear-ssdna-composition-hardening-followups.md)
   for remaining work.
5. Use the [detailed source spec](linear-ssdna-composition/detailed-spec.md)
   only when you need the full design history.

### Claim Boundary

This is a service-contract design for generic linear ssDNA assembly and
secondary-structure QA. It is not a Retron biology summary, a candidate
selection method, or a replacement for study-owned evidence records.

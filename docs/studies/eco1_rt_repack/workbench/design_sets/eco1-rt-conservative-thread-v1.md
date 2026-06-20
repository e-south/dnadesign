## Eco1 RT Conservative Thread v1

**Owner:** dnadesign-maintainers
**Last verified:** 2026-06-19
**Status:** scaffold

This design set will contain the first conservative fixed-backbone Eco1 RT
candidate batch after the `thread` tracer bullet exists.

### Inclusion Intent

- Eco1 RT profile `eco1_rt_v1`.
- Conservative residue mask.
- Zero accepted mask violations.
- Fold-check coverage for selected full sequences.
- Explicit downstream handoff posture for RT-lnRNA collaboration.

### Planned Candidate Funnel

| Step | Planned output | Acceptance posture |
| --- | --- | --- |
| Structure authority | `backbone_bundle.yaml`, `residue_map.parquet` | Non-pending source, chain, retained context, numbering origin, and sequence hash. |
| Evidence profiles | `conservation_profile.parquet`, `contact_profile.parquet` | Per-position mapping, thresholds, source hashes, and no missing evidence inferred as designable. |
| Mask set | `mask_set.yaml` | Conservative union of manual, contact, conservation, unresolved, and cysteine-control masks. |
| Sampling request | `thread_plan.yaml` | Backend request manifest, seeds, temperatures, fixed-position source, and explicit no-fallback policy. |
| Raw samples | `sample_table.parquet` | Deterministic backend provenance, seeds, temperatures, and no fixed-position edits. |
| Deduplicated candidates | `candidate_table.parquet` | Stable ids, mutation list, mutation windows, and mask audit fields. |
| Fold QA | `foldcheck_report.parquet` | Full-sequence fold-check rows for every selected candidate. |
| Synthesis feasibility | `feasibility_report.parquet` | Full-gene first; bounded-window only when parent haplotypes remain traceable. |
| Downstream handoff | `candidate_handoff.yaml` | RT-only candidate promotion target, not an RT-lnRNA construct subject. |

### Candidate Id Shape

Use deterministic ids scoped to the design set:

```text
thread__eco1_rt_conservative_v1__cand_000001
```

The id encodes study/design-set scope and row order after deterministic
deduplication. Backend, score, and fold outcome remain fields, not id tokens.

### Current State

No candidates are selected. This page is the durable design-set placeholder.

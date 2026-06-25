---
doc_id: study-eco1-rt-conservative-thread-v1
surface: study-design-set
study_id: eco1_rt_repack
owner: dnadesign-maintainers
last_verified: 2026-06-25
---

## Eco1 RT Conservative Thread v1

**Status:** backend candidates materialized; fold QA smoke-only

This design set contains the first fixed-backbone Eco1 RT ProteinMPNN candidate
batch generated under `eco1_rt_clade9_plurality25_direct_contact5a_v1`.

### Inclusion Intent

- Eco1 RT profile `eco1_rt_v1`.
- Conservative residue mask.
- Zero accepted mask violations.
- Fold-check coverage before candidate handoff.
- Explicit downstream handoff posture for RT-lnRNA collaboration.

### Candidate Funnel

| Step | Output | Acceptance posture |
| --- | --- | --- |
| Structure authority | `backbone_bundle.yaml`, `residue_map.parquet` | Non-pending source, chain, retained context, numbering origin, and sequence hash. |
| Evidence profiles | `conservation_profile.parquet`, `contact_profile.parquet` | Per-position mapping, thresholds, source hashes, and no missing evidence inferred as designable. |
| Mask set | `mask_set.yaml` | Selected simple policy: protect NAxxH/YADD/VTG, Wang/Ec86 direct contacts, Ec86 clade 9 >=25% WT-plurality conservation calls, and mapped residues within 5 A retained DNA/RNA; classify terminal missing-backbone residues separately. |
| Sampling request | `thread_plan.yaml` | Backend request manifest, seeds, temperatures, fixed-position source, selected non-empty mask policy, mask hash, and explicit no-fallback policy. |
| Raw samples | `sample_table.parquet` | Deterministic backend provenance, seeds, temperatures, and no fixed-position edits. |
| Deduplicated candidates | `candidate_table.parquet` | Stable ids, mutation list, mutation windows, and mask audit fields. |
| Fold QA | `foldcheck_report.parquet` | Current artifact is a six-sequence smoke report. Full WT plus 96-candidate coverage is required before selection. |
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

The active backend batch is `eco1_rt_p25_5a_n96_20260624`. It produced 96
accepted ProteinMPNN sample rows and 96 accepted candidate rows with no
protected-position edits. No candidate has been selected for downstream handoff;
fold QA has only smoke-scale coverage, and synthesis feasibility remains
pending.

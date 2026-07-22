## Eco1 RT Repack Contract

**Owner:** dnadesign-maintainers
**Last verified:** 2026-07-06

This directory stores study-owned contract surfaces for Eco1 RT repack. Some
early readiness files are scaffold records; current phase validation and
selection-readiness checks also have executable providers under
`src/dnadesign/studies/units/eco1_rt_repack/operations/`.

### Contents

- `lifecycle/`: planning mode and phase sequence.
- `surfaces/`: artifact classes and generated-output policy.
- `status/`: checked-in snapshot expectations.
- `readiness/`: planned preflight groups and checks.
- `fixtures/thread/`: Eco1 profile and conservative mask cases for the planned
  `thread` tracer bullet.
- `schemas/`: study-owned schema stubs for Eco1 profile, artifact chain,
  candidate handoff, and RT-only downstream acceptance.

### Readiness Groups

| Group | Purpose |
| --- | --- |
| `thread_profile` | Confirms the study profile, profile schema, mask cases, and policy docs exist. |
| `structure_authority` | Forces structure source, chain, retained context, and numbering decisions before sampling. |
| `mask_contract` | Holds residue-map, conservation, contact, and mask-set contract expectations. |
| `sampling_plan` | Requires explicit backend, seed, temperature, fixed-position, and no-fallback policy. |
| `foldcheck_runtime` | Requires fold-validation semantics and nonfixture coverage for real handoffs. |
| `candidate_handoff` | Requires selected candidates, upstream hash closure, fold QA, and an RT-only sequence export. |
| `downstream_rt_lnrna_handoff` | Routes RT-only candidates to the downstream study without claiming construct ownership. |

Current readiness files intentionally use supported study preflight kinds. Some
Phase 1 and Phase 2 checks remain scaffold-level `path_exists` checks plus
explicit validator intent. Candidate-level acceptance evidence now comes from
code-backed validators and materializers that check artifact state, required
fields/columns, upstream hashes, fixture-vs-materialized separation, exact
six-class panel coverage, sequence-export scope, local-structure metric
availability, and negative cases in `fixtures/thread/conservative_mask_cases.yaml`.

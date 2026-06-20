## Eco1 RT Repack Contract

**Owner:** dnadesign-maintainers
**Last verified:** 2026-06-19

This directory stores the planned machine-readable study contract. It is a
record-plane scaffold, not an executable provider.

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
| `assembly_feasibility` | Requires full-gene/window feasibility evidence before synthesis-oriented handoff. |
| `candidate_handoff` | Requires selected candidates, upstream hash closure, fold QA, and feasibility. |
| `downstream_rt_lnrna_handoff` | Routes RT-only candidates to the downstream study without claiming construct ownership. |

Do not add an executable provider until at least one readiness group has a
machine-checkable implementation target.

Current readiness files intentionally use supported study preflight kinds. Most
Phase 1 and Phase 2 checks are scaffold-level `path_exists` checks plus explicit
validator intent. They are not acceptance evidence for materialized candidates
until code-backed validators check artifact state, required fields/columns,
upstream hashes, fixture-vs-materialized separation, and the negative cases in
`fixtures/thread/conservative_mask_cases.yaml`.

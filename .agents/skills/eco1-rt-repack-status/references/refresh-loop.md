# Refresh Loop

Use this loop when updating or reporting the Eco1 RT repack scaffold.

1. Confirm the requested study id is `eco1_rt_repack`.
2. Read `operations/ops.study.yaml` and confirm it points to existing contract
   parts.
3. Read `record/status.md`, `record/datasets.yaml`, and `record/campaign.yaml`
   before using context pages.
4. Read `routes/README.md` to choose the next owner surface.
5. Read only the context page or readiness check selected by the question.
6. Report whether each named artifact is planned, fixture-backed, or
   materialized.
7. For blockers, report the smallest failing gate: structure authority, mask
   contract, sampling plan, fold-check runtime, assembly feasibility,
   candidate handoff, or downstream handoff.
8. If changing source or tests, confirm the semantic package layout before
   status reporting: `operations/contracts/`, `operations/materialization/`,
   `tests/contracts/`, and `tests/materialization/`.
9. If changing the skill, run the skill audit and the system skill validator.

Fail fast when:

- the study id in a file does not equal `eco1_rt_repack`
- `thread` is described as broadly executable before code exists
- a sampling plan exists without explicit no-fallback backend policy
- a candidate handoff lacks upstream artifact provenance
- a materialized handoff uses fixture fold-check rows
- an RT-lnRNA handoff lacks an explicit downstream acceptance route
- a generated output path is being hand-edited instead of regenerated

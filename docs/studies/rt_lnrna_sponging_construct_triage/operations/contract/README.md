## RT-lnRNA Ops Contract Parts

These YAML fragments are loaded by `../ops.study.yaml`.

- `lifecycle/`: lifecycle mode and phase order.
- `surfaces/`: artifact refs and planned execution surfaces.
- `status/`: record-only snapshot scope.
- `readiness/`: providerless readiness scope, group bindings, next-scope rules,
  GenBank source-authority checks, and phase-named checks.
- `schemas/`: study-owned contract fixtures, including the Construct
  projection manifest schema.
- `fixtures/`: minimal planned candidate, overlay, and Construct projection
  examples. Candidate fixtures use `candidate_role` for working/failed study
  roles; GenBank/source-authority files may still retain historical
  `working_anchor` / `failed_anchor` source labels.

The checked-in Python helper under
`src/dnadesign/studies/studies/rt_lnrna_sponging_construct_triage/` validates
the GenBank source-authority registry, validates the multi-slot Construct
projection manifest, and can run a temporary Construct materialization proof
for the two control candidates. Do not register OPS status/preflight providers
until a concrete study provider exists.

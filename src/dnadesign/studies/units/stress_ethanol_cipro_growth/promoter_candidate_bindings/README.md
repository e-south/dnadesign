---
id: stress-ethanol-cipro-growth-promoter-candidate-bindings
title: Promoter candidate bindings
owner: stress_ethanol_cipro_growth
status: active
last_verified: 2026-07-14
---

# Promoter Candidate Bindings

This study package is the identity-routing authority for promoter candidates in
`stress_ethanol_cipro_growth`. It resolves a namespace-qualified alias to one
canonical candidate and sequence. Multiple exact aliases may name the same
candidate.

The published contract is `dnadesign.study.promoter_candidate_bindings.v1`:

- `manifest.json`
- `bindings.parquet`

Each binding records its alias namespace, canonical candidate and sequence,
source authority, candidate-table digest, and the public BaseRender adapter
projection. Resolution rejects duplicate typed aliases, missing candidates,
sequence disagreement, fuzzy joins, path escapes, and metadata that the public
BaseRender adapter cannot consume.

Sequence figures consume that projection through the
[BaseRender public API](../../../../baserender/docs/reference.md#public-api-boundary);
the binding package does not call renderer internals or define plot styling.

The artifact does not contain Reader measurements, SFXI values,
response-window values, RMF requirements or scores, OPAL state, or LatentDNA X
vectors. Those systems reference `candidate_id`; they do not redefine candidate
identity.

Reader selects `reader.design_id` aliases when it needs sequence context for an
assay deliverable. Synthesis and source records use their own namespaces. The
same binding artifact can therefore route study evidence that does not involve
Reader or OPAL.

## Stable SECG aliases

`docs/studies/stress_ethanol_cipro_growth/record/promoter_aliases.yaml` is the
append-only registry for concise study aliases. Each `SECG-NNN` value is bound
once to one candidate ID and one sequence digest. Selection view, rank,
objective, model round, and assay batch remain provenance and never enter the
alias. A candidate selected again in a later round reuses its existing alias;
a new candidate receives the next ordinal.

`first_assignment.nomination_batch_index` records when the alias first entered
a study candidate set. It is not a physical batch receipt. Ordered, received,
and assayed membership is recorded separately by the synthesis and measurement
lifecycle, at exact-alias granularity when those decisions are made.

The registry projects one identity into the public namespaces used at each
handoff:

- `study.promoter_alias`: `SECG-019`
- `synthesis.name`: `SECG-019`
- `reader.design_id`: `pDual-10-SECG-019`

Earlier `SECG-B0-*` names remain exact source aliases for their existing Reader
and synthesis records. They are not reassigned or discarded. The current
registry contains `SECG-001` through `SECG-036`; the next unassigned alias is
`SECG-037`.

The checked-in `latest` binding bundle remains the digest-pinned identity input
used by the current labels and model run. A later coordinated materialization
will add registry projections before new Reader measurements are published;
that publication must update downstream digest pins atomically rather than
silently rewriting frozen evidence.

Python consumers use the public `load_promoter_candidate_bindings` function.
It verifies the complete bundle before returning rows; consumers do not read
the Parquet record through package-private helpers.

The synthesis adapter reads digest-pinned manifests from either a multi-artifact
pre-assay record or one deduplicated selection-batch record. It does not branch
on an SFXI or RMF source-authority label.

From the repository root:

```bash
uv run python -m dnadesign.studies.units.stress_ethanol_cipro_growth.promoter_candidate_bindings \
  preview --repo-root .

uv run python -m dnadesign.studies.units.stress_ethanol_cipro_growth.promoter_candidate_bindings \
  materialize --repo-root .

uv run python -m dnadesign.studies.units.stress_ethanol_cipro_growth.promoter_candidate_bindings \
  verify \
  --bundle-dir \
  src/dnadesign/studies/units/stress_ethanol_cipro_growth/workbench/outputs/promoter_candidate_bindings/latest
```

Materialization verifies a complete staged bundle before directory-level
publication. Overwrite failure restores the prior bundle; if publication and
rollback both fail, the prior bundle remains at a durable sibling backup path.

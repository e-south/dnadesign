---
id: stress-ethanol-cipro-growth-promoter-candidate-bindings
title: Promoter candidate bindings
owner: stress_ethanol_cipro_growth
status: active
last_verified: 2026-07-13
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

The artifact does not contain Reader measurements, SFXI values,
response-window values, RMF requirements or scores, OPAL state, or LatentDNA X
vectors. Those systems reference `candidate_id`; they do not redefine candidate
identity.

Reader selects `reader.design_id` aliases when it needs sequence context for an
assay deliverable. Synthesis and source records use their own namespaces. The
same binding artifact can therefore route study evidence that does not involve
Reader or OPAL.

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

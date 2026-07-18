---
id: stress-ethanol-cipro-growth-reader-promoter-evidence
title: Reader promoter evidence
owner: stress_ethanol_cipro_growth
status: active
last_verified: 2026-07-18
---

# Reader promoter evidence

The study-owned adapter validates published Reader
`reader.response_window.promoter_evidence_bundle.v4` directories against an
explicit `dnadesign.study.promoter_candidate_bindings.v1` bundle. It stages
their static PNG/PDF artifacts for OPAL display. It writes the
`stress_ethanol_cipro_growth.reader_promoter_evidence.v1` schema with semantic
kind `promoter_response_evidence`; the default filename is
`reader_evidence_promoter_response.json`.

The manifest also declares the public
`opal.reader_evidence_manifest.v1` adapter. The study schema remains the
authority for candidate and experiment provenance; the adapter identifies only
the generic fields OPAL may render. OPAL therefore does not import or recognize
the stress-study schema name.

The handoff is display-only. Reader owns trajectories, reductions, figures,
and source manifests. The stress study validates candidate, sequence-authority,
exact-binding, and adapter-specific provenance. OPAL verifies and displays the
static media. Model-feature readiness uses a separate candidate-keyed contract.
The adapter does not calculate MSRB or any other OPAL objective, create
observed labels, or mutate campaign/model state.

The adapter resolves each exact `reader.design_id` alias through the study
binding artifact and checks candidate, sequence, source, and BaseRender adapter
provenance before publication. Published media live under the round-local
`reader_evidence_media/<Reader-manifest-digest>/` directory. Manifest paths are
relative and digest-verified, so OPAL does not need access to Reader's output
directory after publication.

Run from the `dnadesign` repository root:

```bash
uv run python -m dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.reader_promoter_evidence \
  preview --bindings-bundle <promoter-candidate-bindings-bundle> \
  <promoter-evidence-bundle> [<promoter-evidence-bundle> ...]

uv run python -m dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.reader_promoter_evidence \
  materialize --bindings-bundle <promoter-candidate-bindings-bundle> \
  --out-dir src/dnadesign/opal/campaigns/secg_msrb_greedy/inputs/r0 \
  <promoter-evidence-bundle> [<promoter-evidence-bundle> ...]

uv run python -m dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.reader_promoter_evidence \
  verify src/dnadesign/opal/campaigns/secg_msrb_greedy/inputs/r0/reader_evidence_promoter_response.json
```

`preview` and `verify` do not write. `materialize` atomically installs each
content-addressed media directory, then publishes the manifest as the final
commit point. It refuses to replace an existing manifest unless `--overwrite`
is explicit. None of the commands ingest or apply labels.

The adapter always emits `campaign_slug: secg_msrb_greedy`, the only
executable campaign destination for this evidence. Other campaign slugs occur
only as digest-pinned synthesis-source provenance.

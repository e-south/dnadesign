# Reader promoter evidence

This study-owned package validates published Reader
`reader.response_window.promoter_evidence_bundle.v1` directories and stages
their static PNG/PDF artifacts for OPAL display. It writes the existing
`stress_ethanol_cipro_growth.reader_evidence.v1` schema with semantic kind
`promoter_response_evidence`; the default filename is
`reader_evidence_promoter_response.json`.

The handoff is display-only. Reader owns trajectories, reductions, figures,
and source manifests. The stress study validates candidate, sequence-authority,
exact-binding, and adapter-specific provenance. OPAL verifies and displays the
static media. Model-feature readiness uses a separate candidate-keyed contract.
This package does not calculate RMF, create observed labels, or mutate
campaign/model state.

Run from the `dnadesign` repository root:

```bash
uv run python -m dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.reader_promoter_evidence \
  preview <reader-bundle> [<reader-bundle> ...]

uv run python -m dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.reader_promoter_evidence \
  materialize --out-dir src/dnadesign/opal/campaigns/secg_rmf_greedy/inputs/r0 \
  <reader-bundle> [<reader-bundle> ...]

uv run python -m dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.reader_promoter_evidence \
  verify src/dnadesign/opal/campaigns/secg_rmf_greedy/inputs/r0/reader_evidence_promoter_response.json
```

`preview` and `verify` do not write. `materialize` atomically writes one
manifest and refuses to replace an existing target unless `--overwrite` is
explicit. None of the commands ingest or apply labels.

This study adapter always emits `campaign_slug: secg_rmf_greedy`, the only
executable campaign destination for this evidence. Other campaign slugs occur
only as digest-pinned synthesis-source provenance.

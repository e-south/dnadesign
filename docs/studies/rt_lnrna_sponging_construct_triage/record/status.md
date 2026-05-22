## rt_lnrna_sponging_construct_triage

- Last verified: 2026-05-22
- Owner: dnadesign-maintainers
- Affiliated dataset registry: `datasets.yaml`
- Route map: `../routes/README.md`
- Study execution map: `../operations/runtime/command-groups/pipeline.yaml`
- Lifecycle posture: Phase 0/1 contract bootstrap
- OPS provider: none registered

### Current Phase

The study has a checked-in contract skeleton, minimal fixtures, GenBank source
authority for the two anchor rows, and a multi-slot Construct projection
manifest. It is not ready for Infer runs, LatentDNA materialization, or OPAL
training because Construct context views have not been materialized.

### Current Evidence

- Khan source inventory is pinned from
  `../../../../dnadesign-data/sources/literature/Khan_et_al_2024_retron_census/processed/`.
- Crawford source inventory is pinned from
  `../../../../dnadesign-data/sources/literature/Crawford_et_al_2025_retron_ncRNA_ML/processed/`.
- GenBank source authority is registered in
  `../workbench/provenance/genbank-source-authority.yaml`.
- Parsed GenBank offsets are recorded in
  `../workbench/provenance/genbank-feature-offset-audit.md`.
- Multi-slot Construct projection is declared in
  `../operations/contract/fixtures/construct/construct-projection-manifest.yaml`.
- `retron-eco1-rt.gb` matches the RT CDS and CDS translation in both
  `pes-retron-26.gb` and `pes-retron-43.gb`.
- `pes-retron-26-a1-a2.gb` is contained in `pes-retron-26.gb`.
- `retron-179-a1-a2.gb` is orientation evidence for explicit left/right base,
  snapback cap, foldback, and payload/complement geometry.

### Phase 0 Blockers

- Construct runtime execution from the multi-slot projection manifest.
- Construct context view materialization from audited offsets and slot spans.
- Fixed-size representation table contract for downstream Infer/LatentDNA/OPAL.

### Phase 1 Posture

The two anchor fixtures are source-authority resolved and
`construct_projection_status: representable` under the multi-slot projection
manifest. They can validate as study candidate rows, but they must not be
treated as materialized construct rows until Construct emits the declared
context views.

### Next Actions

1. Materialize construct context views with Construct's public multi-slot
   assembler.
2. Validate emitted slot spans against the projection manifest.
3. Run the schema/check fixtures before materializing candidate tables.
4. Keep abundance priors and future sponging labels separate through OPAL
   handoff.

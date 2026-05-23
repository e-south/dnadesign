## rt_lnrna_sponging_construct_triage

- Last verified: 2026-05-23
- Owner: dnadesign-maintainers
- Affiliated dataset registry: `datasets.yaml`
- Route map: `../routes/README.md`
- Study execution map: `../operations/runtime/command-groups/pipeline.yaml`
- Lifecycle posture: Phase 0/1 contract bootstrap
- OPS provider: none registered

### Current Phase

The study has a checked-in contract skeleton, minimal candidate fixtures,
GenBank source authority for two lab-anchor-derived candidate rows, a multi-slot
Construct projection manifest, and a test-backed temporary Construct
materialization path for the two controls. It is not ready for Infer runs,
LatentDNA materialization, or OPAL training because the realized context outputs
are not yet persisted as a study workspace/export bundle.

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
- `src/dnadesign/studies/studies/rt_lnrna_sponging_construct_triage/construct_materialization.py`
  converts the manifest and GenBank authority into temporary Construct configs
  for the two controls.
- Planned persistent USR dataset ids are registered in `../record/datasets.yaml`:
  `rt_lnrna_sponging_construct_triage_construct_slot_inputs_v1`,
  `rt_lnrna_sponging_construct_triage_construct_contexts_1600bp_v1`, and
  `rt_lnrna_sponging_construct_triage_opal_training_examples_v1`.
- Targeted tests assert the two realized 1,600 bp contexts, slot spans, real
  prefix/interstitial/suffix sequence, forward/reverse-complement rows, and the
  lnRNA anchor-mean diagnostic view.
- `retron-eco1-rt.gb` matches the RT CDS and CDS translation in both
  `pes-retron-26.gb` and `pes-retron-43.gb`.
- `pes-retron-26-a1-a2.gb` is contained in `pes-retron-26.gb`.
- `retron-179-a1-a2.gb` is orientation evidence for explicit left/right base,
  snapback cap, foldback, and payload/complement geometry.

### Remaining Blockers

- Persistent study workspace/export materialization for realized context rows.
- Fixed-size representation table contract for downstream Infer/LatentDNA/OPAL.
- Overlay resolver rules for promoting Crawford/Khan rows only when explicit RT
  plus lnRNA sequence authority exists.

### Phase 1 Posture

The two lab-anchor-derived candidate fixtures are source-authority resolved and
`construct_projection_status: representable` under the multi-slot projection
manifest. They now have a temporary integration proof that Construct can emit
the declared context views, but the checked-in candidate fixtures still point to
no persisted construct rows.

### Next Actions

1. Add a persistent study workspace/export step for the two control
   materializations.
2. Add the overlay resolver slice: Crawford rows can become construct
   candidates only with explicit Eco1 WT RT plus lnRNA authority; Khan remains
   overlay/provenance unless both sequence authorities are named.
3. Run the schema/check fixtures before materializing larger candidate tables.
4. Keep abundance priors and future sponging labels separate through OPAL
   handoff.

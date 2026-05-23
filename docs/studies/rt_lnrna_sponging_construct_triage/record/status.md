## rt_lnrna_sponging_construct_triage

- Last verified: 2026-05-23
- Owner: dnadesign-maintainers
- Affiliated dataset registry: `datasets.yaml`
- Route map: `../routes/README.md`
- Study execution map: `../operations/runtime/command-groups/pipeline.yaml`
- Lifecycle posture: Phase 3 Infer handoff blocked on missing feature sidecars and real labels
- OPS provider: none registered

### Current Phase

The study has checked-in Phase 0/1 contracts, two source-authority-resolved
lab-anchor candidate rows, a multi-slot Construct projection manifest, a
test-backed Construct materialization path for the two controls, and a fixed-size
representation-table contract for the next Infer/LatentDNA handoff. It is not
ready for OPAL training because Evo2 feature sidecars and real
`SpongingAssayObservation` labels are still absent.

### Current Evidence

- Khan source inventory is pinned from
  `../../../../dnadesign-data/sources/literature/Khan_et_al_2024_retron_census/processed/`.
- Crawford source inventory is pinned from
  `../../../../dnadesign-data/sources/literature/Crawford_et_al_2025_retron_ncRNA_ML/processed/`.
- GenBank source authority is registered in
  `../workbench/provenance/genbank-source-authority.yaml`.
- Parsed GenBank offsets are recorded in
  `../workbench/provenance/genbank-feature-offset-audit.md`.
- The target 1,600 bp context is `../1600bp-region.gb`, contained in
  pES-retron-26 at zero-based half-open vector coordinates `[56,1656)`.
- Multi-slot Construct projection is declared in
  `../operations/contract/fixtures/construct/construct-projection-manifest.yaml`.
- `src/dnadesign/studies/studies/rt_lnrna_sponging_construct_triage/construct_materialization.py`
  converts the manifest and GenBank authority into Construct configs for the
  two controls and emits the six source sequence-view lanes.
- USR dataset ids are registered in `../record/datasets.yaml`:
  `rt_lnrna_sponging_construct_triage_construct_slot_inputs_v1`,
  `rt_lnrna_sponging_construct_triage_construct_contexts_1600bp_v1`, and
  `rt_lnrna_sponging_construct_triage_opal_training_examples_v1`.
- Targeted tests assert the two realized 1,600 bp contexts, slot spans, real
  prefix/interstitial/suffix sequence, forward/reverse-complement rows, and the
  forward/RC lnRNA plus RT CDS anchor-mean views with
  `context_kind=template_custom`.
  The retron26 control emits `lnrna: [130,303)` and `rt_cds: [468,1431)`;
  the longer retron43 lnRNA symmetrically trims the region flanks and emits
  `lnrna: [123,310)` and `rt_cds: [475,1438)`.
- `../operations/contract/schemas/representation-table.schema.yaml` declares the
  fixed-size representation-table contract, including expected Evo2 7B vector
  dimensions and Khan/Crawford overlay integration boundaries.
- `../../../../src/dnadesign/latentdna/workspaces/rt_lnrna_sponging_construct_triage/config.yaml`
  declares the planned LatentDNA representation-health, Khan/Crawford ordinal
  audit, and appendix UMAP surfaces across intermediate and output-layer
  gallery views.
- `retron-eco1-rt.gb` matches the RT CDS and CDS translation in both
  `pes-retron-26.gb` and `pes-retron-43.gb`.
- `pes-retron-26-a1-a2.gb` is contained in `pes-retron-26.gb`.
- `retron-179-a1-a2.gb` is orientation evidence for explicit left/right base,
  snapback cap, foldback, and payload/complement geometry.

### Remaining Blockers

- Evo2 7B Infer sidecars for the six explicit `view_name` lanes.
- LatentDNA materialization of the declared representation-health, ordinal
  overlay, and UMAP gallery views after sidecars exist.
- Overlay resolver rules for promoting Crawford/Khan rows only when explicit RT
  plus lnRNA sequence authority exists.
- Real `SpongingAssayObservation` labels before any supervised OPAL run.

### Phase 1 Posture

The two lab-anchor-derived candidate fixtures are source-authority resolved and
`construct_projection_status: representable` under the multi-slot projection
manifest. Construct can emit the declared 1,600 bp views with the correct
`template_custom` sequence-view metadata. Candidate fixtures remain source
fixtures and do not inline generated view ids.

### Next Actions

1. Run Infer sequence-view completion preflight using explicit `view_name`
   selectors before any Evo2 execution.
2. Materialize Evo2 7B sidecars for the six declared source views.
3. Validate the LatentDNA workspace config, validate the review recipe, and
   materialize the declared health, ordinal-audit, and UMAP gallery views from
   sidecars.
4. Add the overlay resolver slice: Crawford rows can become construct
   candidates only with explicit Eco1 WT RT plus lnRNA authority; Khan remains
   overlay/provenance unless both sequence authorities are named.
5. Run the schema/check fixtures before materializing larger candidate tables.
6. Keep abundance priors and future sponging labels separate through OPAL
   handoff.

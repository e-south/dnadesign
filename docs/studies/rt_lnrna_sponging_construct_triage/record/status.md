## rt_lnrna_sponging_construct_triage

- Last verified: 2026-07-21
- Owner: dnadesign-maintainers
- Affiliated dataset registry: `datasets.yaml`
- Route map: `../routes/README.md`
- Study execution map: `../operations/runtime/command-groups/pipeline.yaml`
- Lifecycle posture: record-only and paused; Reader SPOP labels and LatentDNA review artifacts are materialized, while OPAL remains blocked on candidate-X selection
- OPS provider: study execution surfaces include a six-view Infer batch runbook

### Current Phase

The study has checked-in Phase 0/1 contracts, a consolidated Construct
materialization path for GenBank-authorized retrons, Crawford source-sequence
promotions, Khan source RT-lnRNA rows that fit the current 2,000 bp construct
geometry, a bounded compiler-generated MSD lnRNA variant fixture pool, and
RT-CDS in silico DMS variants, plus a fixed-size representation-table contract
for the next LatentDNA and OPAL handoff. It is not ready for OPAL training
because a fixed candidate-X vector has not yet been selected; durable
`SpongingAssayObservation` labels are now materialized and schema-validated.

### Current Evidence

- Khan source inventory is pinned from
  `../../../../dnadesign-data/sources/literature/Khan_et_al_2024_retron_census/processed/`.
- Crawford source inventory is pinned from
  `../../../../dnadesign-data/sources/literature/Crawford_et_al_2025_retron_ncRNA_ML/processed/`.
- GenBank source authority is registered in
  `../workbench/provenance/genbank-source-authority.yaml`.
- Parsed GenBank offsets are recorded in
  `../workbench/provenance/genbank-feature-offset-audit.md`.
- Study-owned GenBank files live under
  `../workbench/provenance/genbank/`; the temporary transfer directory has been
  drained.
- The target 2,000 bp context is `../workbench/provenance/genbank/2000bp-region.gb`,
  contained in pES-retron-26 at zero-based half-open vector coordinates
  `[56,2056)`.
- `../workbench/provenance/retron-variant-genbank-metadata.yaml` preserves
  available Benchling links, antibiotic markers, and user comments for 35
  retron whole-plasmid variants, ten retron-hairpin MSD-only handoffs, and the
  BL21 wild-type lnRNA-only source;
  `../workbench/provenance/retron-variant-genbank-catalog.yaml` records parsed
  lnRNA and RT slot source authority for 46 Construct-representable rows.
- Multi-slot Construct projection is declared in
  `../operations/contract/fixtures/construct/construct-projection-manifest.yaml`.
- `src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/construct_materialization.py`
  orchestrates Construct materialization. Its support modules keep contracts,
  manifest loading, Construct view config, Construct-subject row building, and
  USR overlay writes in the `materialization/` package, so the study
  no longer depends on one all-purpose source file for the six source
  sequence-view lanes.
- USR dataset ids are registered in `../record/datasets.yaml`:
  `rt_lnrna_sponging_construct_triage_construct_slot_inputs_v1`,
  `rt_lnrna_sponging_construct_triage_construct_contexts_2000bp_v1`, and
  `rt_lnrna_sponging_construct_triage_opal_training_examples_v1`.
- Targeted tests assert realized 2,000 bp contexts, slot spans, real
  prefix/interstitial/suffix sequence, forward/reverse-complement rows, and the
  forward/RC lnRNA plus RT CDS fixed-window anchor-mean views with
  `context_kind=template_custom`. The GenBank catalog uses one projection path
  for all 46 representable rows. The retron26 fixture emits slot spans
  `lnrna: [130,303)` and `rt_cds: [468,1431)`; the retron43 fixture emits
  `lnrna: [123,310)` and `rt_cds: [475,1438)`.
- The catalog-to-Construct materializer now dogfoods all 46 catalog rows into
  the consolidated 2,000 bp output surface. It groups rows by per-candidate
  window offset so positive lnRNA/RT length deltas truncate only the outer
  prefix/suffix flanks, while preserving full lnRNA and RT slot spans. The
  Sso7d-fusion retron47/retron48 rows emit `lnrna: [27,200)` and
  `rt_cds: [365,1535)`.
- The source-promotion resolver currently promotes 4,148 abundance-affiliated
  Crawford source lnRNA sequences and records 18 design-reference-only sequences
  as missing affiliated abundance observations. Promoted sequences are paired
  with fixed WT Eco1 RT after DNA4 validation, Eco1 forward k-mer orientation
  QC, reverse-complement rejection, and the manifest-derived lnRNA-centered
  2,000 bp construct-window preflight. Exact declared MSD substring and short
  flank matches are retained as QC annotations because abundance-bearing
  Crawford variants can intentionally alter those regions.
- Khan source rows resolve through the terminal-keyed Khan sequence-authority
  table, the Khan abundance-prior overlay, and the Mestre Supplementary Table S3
  RT locus authority path. The local sequence-authority refresh resolves 169 of
  171 RT CDS sequences with exact translation validation. The abundance-prior
  overlay has 99 numeric rows; 71 pass source ncRNA, translation-exact RT CDS,
  affiliated abundance, and current 2,000 bp lane-fit gates. Two RT loci remain
  unresolved, 58 sequence-authorized rows that fit the lane lack an affiliated
  abundance prior, and 40 sequence-authorized rows exceed the current lane.
- The current 2,000 bp context promotes 4,148 abundance-affiliated Crawford
  rows and 71 abundance-affiliated Khan rows. It also records 18 Crawford
  design-reference-only sequences and 58 Khan sequence-authority rows as
  missing affiliated abundance observations.
- The compiler-generated MSD lnRNA fixture pool lives at
  `../operations/contract/fixtures/source-promotions/msd-compiler-pool.yaml`.
  It compiles the YIU-compatible full combinatoric pool from five DE033
  Snapback cap primitive ranks and sixteen scar-nick TetO stem-base primitive
  ranks through the pure compiler API, requires `lnrna_insert_sequence_5to3` to
  equal the reverse complement of the MSD product, exact-matches the retron26
  template MSD plus 5-prime/3-prime flanks, and writes ordinary Construct
  subject rows with fixed Eco1 WT RT. It does not formalize or materialize a
  pre-Infer concat.
- The live consolidated Construct input dogfood contains 10,425 construct
  subjects: 46 GenBank-authorized subjects, 4,148 abundance-affiliated Crawford
  source-sequence subjects paired with fixed WT Eco1 RT, 71 abundance-affiliated
  Khan RT-lnRNA subjects, 80 compiler-generated MSD lnRNA variant subjects, and
  6,080 RT-CDS DMS subjects generated through the public `dnadesign.permuter`
  API.
- The live Construct output dogfood validates strictly with 20,850 realized
  context rows and 62,550 explicit sequence-view declarations. Each construct
  subject has all six required view names.
- The executable Infer-readiness gate at
  `../../../../src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/infer_readiness.py`
  now runs as a Construct materialization postcondition. It requires one
  forward context row, one reverse-complement context row, and exactly the six
  declared source sequence-view names per construct subject before the study can
  hand the dataset to Infer. A full temp dogfood through public dnadesign-data
  source IDs passed with 10,425 subjects, 20,850 Construct output rows, and
  62,550 sequence-view rows; the source-promotion issues are explicit: 76
  missing affiliated abundance observations, 2 missing RT CDS rows, and 40
  over-window Khan rows.
- `../operations/contract/schemas/representation-table.schema.yaml` declares the
  fixed-size representation-table contract, including expected Evo2 7B vector
  dimensions and Khan/Crawford overlay integration boundaries.
- `../../../../src/dnadesign/latentdna/workspaces/rt_lnrna_sponging_construct_triage/config.yaml`
  declares the current sidecar-backed LatentDNA review surface: dataset/source
  overview, representation-health gate, source/design-structure summary,
  Khan/Crawford ordinal audit, abundance margin galleries, slot/context
  robustness, candidate-X frontier/scorecard, appendix scree, and appendix
  UMAP views across intermediate and output-layer gallery views.
- `../contexts/reader-spop-label-contract.md` and
  `../operations/contract/readiness/checks/reader_spop_label_materialization.yaml`
  declare the Reader-to-Construct SPOP bridge. Reader owns the SPOP
  metric source-of-truth in `reader/docs/lib/spop_endpoint_in_reader.md`; this
  study bridge resolves Reader ratio artifacts through `records.json`, keeps
  assay subject identity separate from Construct subject identity, delegates
  scoring to Reader's public `score_spop_endpoint` API, materializes LatentDNA
  overlay tables, and routes OPAL SPOP campaigns as `spop_v1/spop` only after a
  selected `X` exists.
- Current Reader SPOP materialization evidence is label-planner clean with one explicit
  no-call warning:
  `uv run python -m dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reader_spop_plan`
  reports 56 observations across 40 candidate summaries, including
  `20260705_retron_Eco1_26_195_196_180_199_200_197_198_benchmark` and
  `20260720_retron_Eco1_26_180_201_202_203_204_benchmark`. The
  `20251105_retron_Eco1_RT_variants` Reader artifact is treated as a
  single-point mid-log read at approximately 10 h after seeding, even though the
  artifact stores row time as 0 h and the historical config reported 12 h.
  Retron176 in
  `20260507_retron_Eco1_26_43_172_173_174_175_176_benchmark` is intentionally
  omitted because the plate map carried retron176 but no actual strain was
  present in those wells.
- `retron-eco1-rt.gb` matches the RT CDS and CDS translation in both
  `pes-retron-26.gb` and `pes-retron-43.gb`.
- `pes-retron-26-a1-a2.gb` is contained in `pes-retron-26.gb`.
- `retron-179-a1-a2.gb` is orientation evidence for explicit left/right base,
  snapback cap, foldback, and payload/complement geometry.
- The variant catalog resolves sequence authority for retron18, 24-27, 43,
  45-56, 170-186, 195-204, and `msrmsdwt_bl21`.
- The study-owned Infer runbook
  `../../../../src/dnadesign/ops/runbooks/presets/infer_rt_lnrna_sponging_construct_triage_six_view_7b_batch_with_notify.yaml`
  records the batch entrypoint for completing the six-view Evo2 7B workload with one
  Notify watcher for the lane. The study-level fill command is
  `uv run ops runbook fill-infer --study-dir docs/studies/rt_lnrna_sponging_construct_triage`.

### Remaining Blockers

- Reader SPOP labels and OPAL handoff. Durable Reader label tables, LatentDNA
  sidecars, and review surfaces are materialized, but OPAL still needs a
  selected fixed-size `X` and the final training-table join.
- Khan source rows exceeding the current fixed 2,000 bp Construct window. The
  translation-validated Khan RT CDS path is present, but 40 cross-retron RTs are
  too long for the current lnRNA-centered 2,000 bp geometry when paired with
  their source ncRNA. Complete coverage needs either a larger context or an
  alternate windowing contract before those rows can enter the six-view Infer
  lane.
  The current over-window pressure test gives this coverage curve under the same
  placement policy: 2.1 kb still blocks 33 rows, 2.2 kb blocks 15, 2.3 kb
  blocks 5, 2.5 kb blocks 1, and 2.626 kb blocks 0. The active lane remains
  fixed at 2,000 bp until a separate larger-context contract is chosen.
- Crawford source-context equivalence. Promoted Crawford source lnRNA sequences
  are projected into the dnadesign dual-cassette context and explicitly
  annotated as not native/exact Crawford expression-context recreations; A1/A2
  extension geometry is not assumed to match the dnadesign A1/A2=20 convention.
### Phase 1 Posture

The checked-in construct-subject fixtures are representative GenBank
source-authority rows resolved as `construct_projection_status: representable`
under the multi-slot projection manifest. Construct can emit the declared
2,000 bp views with the correct `template_custom` sequence-view metadata.
Construct-subject fixtures remain source fixtures and do not inline generated
view ids.

### Next Actions

1. Review the LatentDNA candidate-X frontier and scorecard before selecting an
   OPAL `X`; do not default to the largest concat.
2. Decide whether to add a larger or alternate Construct context for
   over-capacity Khan source rows, or keep the current 2,000 bp lane as an
   Eco1-sized normalized context. The 2,000 bp lane is useful but not complete.
3. Rerun the Reader SPOP label planner as new sibling Reader manifests arrive,
   audit any endpoint drift, malformed treatments, or weak aTc positive
   controls, and refresh the durable label sidecar only after that audit passes.
4. Run the schema/check fixtures before materializing larger construct subject tables.
5. Keep abundance priors and Reader sponging labels separate through OPAL
   handoff.

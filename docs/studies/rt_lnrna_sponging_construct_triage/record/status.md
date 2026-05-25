## rt_lnrna_sponging_construct_triage

- Last verified: 2026-05-25
- Owner: dnadesign-maintainers
- Affiliated dataset registry: `datasets.yaml`
- Route map: `../routes/README.md`
- Study execution map: `../operations/runtime/command-groups/pipeline.yaml`
- Lifecycle posture: Phase 3 Infer handoff blocked on missing feature sidecars and real labels
- OPS provider: none registered

### Current Phase

The study has checked-in Phase 0/1 contracts, a consolidated Construct
materialization path for GenBank-authorized retrons, Crawford source-sequence
promotions, Khan source RT-lnRNA rows that fit the current 2,000 bp construct
geometry, and RT-CDS in silico DMS variants, plus a fixed-size
representation-table contract for the next Infer/LatentDNA handoff. It is not
ready for OPAL training because Evo2 feature sidecars and materialized
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
- Study-owned GenBank files live under
  `../workbench/provenance/genbank/`; the temporary transfer directory has been
  drained.
- The target 2,000 bp context is `../workbench/provenance/genbank/2000bp-region.gb`,
  contained in pES-retron-26 at zero-based half-open vector coordinates
  `[56,2056)`.
- `../workbench/provenance/retron-variant-genbank-metadata.yaml` preserves
  available Benchling links, antibiotic markers, and user comments for 35
  retron whole-plasmid variants plus the BL21 wild-type lnRNA-only source;
  `../workbench/provenance/retron-variant-genbank-catalog.yaml` records parsed
  lnRNA and RT slot source authority for 36 Construct-representable rows.
- Multi-slot Construct projection is declared in
  `../operations/contract/fixtures/construct/construct-projection-manifest.yaml`.
- `src/dnadesign/studies/studies/rt_lnrna_sponging_construct_triage/construct_materialization.py`
  converts the manifest and GenBank authority into Construct configs for the
  two controls and emits the six source sequence-view lanes.
- USR dataset ids are registered in `../record/datasets.yaml`:
  `rt_lnrna_sponging_construct_triage_construct_slot_inputs_v1`,
  `rt_lnrna_sponging_construct_triage_construct_contexts_2000bp_v1`, and
  `rt_lnrna_sponging_construct_triage_opal_training_examples_v1`.
- Targeted tests assert the two realized 2,000 bp contexts, slot spans, real
  prefix/interstitial/suffix sequence, forward/reverse-complement rows, and the
  forward/RC lnRNA plus RT CDS anchor-mean views with
  `context_kind=template_custom`.
  The retron26 control emits `lnrna: [130,303)` and `rt_cds: [468,1431)`;
  the longer retron43 lnRNA symmetrically trims the region flanks and emits
  `lnrna: [123,310)` and `rt_cds: [475,1438)`.
- The catalog-to-Construct materializer now dogfoods all 36 catalog rows into
  the consolidated 2,000 bp output surface. It groups rows by per-candidate
  window offset so positive lnRNA/RT length deltas truncate only the outer
  prefix/suffix flanks, while preserving full lnRNA and RT slot spans. The
  Sso7d-fusion retron47/retron48 rows emit `lnrna: [27,200)` and
  `rt_cds: [365,1535)`.
- The source-promotion resolver currently promotes 4,166 Crawford source lnRNA
  sequences from the union of design-reference and abundance-observation rows.
  They are paired with fixed WT Eco1 RT after DNA4 validation, Eco1 forward
  k-mer orientation QC, reverse-complement rejection, and the manifest-derived
  lnRNA-centered 2,000 bp construct-window preflight. Exact declared MSD
  substring and short flank matches are retained as QC annotations because
  abundance-bearing Crawford variants can intentionally alter those regions.
- Khan source rows now resolve through the terminal-keyed Khan sequence-authority
  table and the Mestre Supplementary Table S3 RT locus authority path. The local
  sequence-authority refresh resolves 169 of 171 RT CDS sequences with exact
  translation validation; all 99 abundance-assayed rows have RT plus ncRNA
  sequence authority. Under the current fixed 2,000 bp construct context, 129 Khan
  rows fit and promote. Two non-assayed RT loci remain unresolved, and 40
  otherwise sequence-authorized Khan rows are blocked before Construct because
  their source lnRNA plus RT CDS geometry falls outside the current
  lnRNA-centered 2,000 bp context.
- The current 2,000 bp context promotes all 4,166 Crawford rows and 129 Khan
  rows, but still leaves 40 otherwise sequence-authorized Khan rows blocked. It
  is therefore a normalized Eco1-sized lane, not a complete cross-retron
  coverage lane.
- The live consolidated Construct input dogfood contains 10,411 construct
  subjects: 36 GenBank-authorized subjects, 4,166 Crawford source-sequence
  subjects paired with fixed WT Eco1 RT, 129 Khan source RT-lnRNA subjects, and
  6,080 RT-CDS DMS subjects generated through the public `dnadesign.permuter`
  API.
- The live Construct output dogfood validates strictly with 20,822 realized
  context rows and 62,466 explicit sequence-view declarations. Each construct
  subject has all six required view names.
- `../operations/contract/schemas/representation-table.schema.yaml` declares the
  fixed-size representation-table contract, including expected Evo2 7B vector
  dimensions and Khan/Crawford overlay integration boundaries.
- `../../../../src/dnadesign/latentdna/workspaces/rt_lnrna_sponging_construct_triage/config.yaml`
  declares the planned LatentDNA PCA representation-health gate, appendix
  scree diagnostic, Khan/Crawford ordinal audit, and appendix UMAP surfaces
  across intermediate and output-layer gallery views.
- `../contexts/reader-spop-label-contract.md` and
  `../operations/contract/readiness/checks/reader_spop_label_materialization.yaml`
  declare the planned Reader-derived SPOP scalar materializer. It resolves
  Reader ratio artifacts through `records.json`, keeps assay subject identity
  separate from Construct subject identity, and routes
  `reader_spop_endpoint_auc_v1` to OPAL as `scalar_identity_v1/scalar`.
- Current Reader SPOP dry-run evidence is label-planner clean with one explicit
  no-call warning:
  `uv run python -m dnadesign.studies.studies.rt_lnrna_sponging_construct_triage.reader_spop_plan`
  reports 30 observations across 20 candidate summaries. The
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
  45-56, 170-186, and `msrmsdwt_bl21`.

### Remaining Blockers

- Evo2 7B Infer sidecars for the six explicit `view_name` lanes.
- LatentDNA materialization of the declared representation-health, ordinal
  overlay, scree, and UMAP gallery views after sidecars exist.
- Khan source rows exceeding the current fixed 2,000 bp Construct window. The
  translation-validated Khan RT CDS path is present, but 40 cross-retron RTs are
  too long for the current lnRNA-centered 2,000 bp geometry when paired with
  their source ncRNA. Complete coverage needs either a larger context or an
  alternate windowing contract before those rows can enter the six-view Infer
  lane.
- Crawford source-context equivalence. Promoted Crawford source lnRNA sequences
  are projected into the dnadesign dual-cassette context and explicitly
  annotated as not native/exact Crawford expression-context recreations; A1/A2
  extension geometry is not assumed to match the dnadesign A1/A2=20 convention.
- Durable Reader SPOP label sidecar. The planner now resolves live Reader rows
  through the variant GenBank catalog and carries the known 2025-11-05
  single-point endpoint caveat plus 2026-05-07 retron176 no-strain omission, but
  the durable sidecar still needs to be written and schema-validated.

### Phase 1 Posture

The two lab-anchor-derived construct-subject fixtures are source-authority
resolved and `construct_projection_status: representable` under the multi-slot
projection manifest. Construct can emit the declared 2,000 bp views with the
correct `template_custom` sequence-view metadata. Construct-subject fixtures
remain source fixtures and do not inline generated view ids.

### Next Actions

1. Run Infer sequence-view completion preflight using explicit `view_name`
   selectors before any Evo2 execution.
2. Materialize Evo2 7B sidecars for the six declared source views.
3. Validate the LatentDNA workspace config, validate the review recipe, and
   materialize the declared PCA health, scree, ordinal-audit, and UMAP gallery
   views from sidecars.
4. Decide whether to add a larger or alternate Construct context for
   over-capacity Khan source rows, or keep the current 2,000 bp lane as an
   Eco1-sized normalized context. The 2,000 bp lane is useful but not complete.
5. Run the Reader SPOP label planner against sibling Reader manifests and audit
   any endpoint drift, malformed treatments, or weak aTc positive controls
   before writing a durable label sidecar.
6. Run the schema/check fixtures before materializing larger construct subject tables.
7. Keep abundance priors and future sponging labels separate through OPAL
   handoff.

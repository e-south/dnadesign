---
doc_id: study-eco1-rt-repack-status
surface: study-record
study_id: eco1_rt_repack
owner: dnadesign-maintainers
last_verified: 2026-06-29
status_surface: record-only
---

## Eco1 RT Repack Status

### Current Phase

Phase 3 fold-check report validation passes locally for the full WT plus
96-candidate ColabFold report. The study has the required structure, source,
alignment, conservation evidence, manual mask authority, mask set, explicit
thread plan, ProteinMPNN request, backend run manifest, sample/candidate tables,
fold-check request, and compact fold-check report under
`src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/`.
This is full fold-check coverage, not downstream candidate-handoff readiness.
The selected mask rule is:

```text
eco1_rt_clade9_plurality25_direct_contact5a_v1
```

The rule is:

```text
protected =
  NAxxH / YADD / VTG
  OR Wang/Ec86 direct-contact mask prior
  OR Eco1 amino acid is evolutionarily conserved at >=25% WT plurality in the Ec86 clade 9 MSA
  OR mapped residue is within 5 A of retained DNA/RNA

non_fixed = NOT protected
```

Terminal residues `1`, `2`, and `312-320` are not protected by this policy.
They are classified as `non_fixed_missing_backbone`: unprotected, but not
directly mutable by fixed-backbone ProteinMPNN until coordinates are supplied or
handled separately.

### Materialized Evidence

- Structure authority is `ec86kit_7v9u_protomer1`: PDB `7v9u`,
  RT chain `A`, retained DNA chain `D`, retained RNA chains `E/F`, paired
  protomer excluded, and no paired-protomer dimerization retention objective.
- `backbone_bundle.yaml` and `residue_map.parquet` are materialized. The
  selected fixed-backbone structure has 309 mapped positions and 11 missing
  terminal positions: `1`, `2`, `312-320`.
- `structure_preprocessing_manifest.yaml`, `contact_profile.parquet`, and
  `contact_geometry_profile.parquet` are materialized from the selected 7V9U /
  ec86kit protomer context.
- Mestre-derived source authority covers
  `ec86_clade9_conservation_v1` and
  `ec86_iia3_cluster42_1_conservation_v1`; the full Mestre roster remains
  context/candidate-pool evidence, not the conservation denominator.
- Provider source acquisition, source-record QC, source FASTA sufficiency,
  Clustal Omega alignments, and `conservation_profile.parquet` are available
  for both selected profiles.
- `conservation_profile.parquet` has 640 rows: 320 positions per selected
  profile. The mask rule uses `ec86_clade9_conservation_v1` as the
  conservation veto: Eco1 residues are protected when the Eco1 amino acid is
  the clade 9 plurality residue at frequency `>=25%`.
- Manual motif authority records NAxxH `105-109`, YADD `195-198`, and VTG
  `243-245` as protected anchors. RT1-RT7 intervals remain annotation/review
  labels and do not blanket hard-fix residues under this rule.
- Wang/Ec86 direct-contact mask priors are protected when listed in
  `manual-mask-authority.yaml`.
- `mask_set.yaml` is materialized under
  `eco1_rt_clade9_plurality25_direct_contact5a_v1`; Phase 1 validates locally.
- `thread_plan.yaml` is materialized locally with explicit `proteinmpnn`
  backend selection, seeds `101`, `202`, `303`, temperatures `0.1` and `0.3`,
  a request hash, and `explicit_no_fallback` policy. The plan emits 123 mapped
  mutable positions and excludes terminal `non_fixed_missing_backbone`
  positions from fixed-backbone mutation.
- `proteinmpnn_request/request_manifest.yaml` is materialized locally. The Eco1
  wrapper resolves study paths and selected structure provenance, then calls
  `dnadesign.thread.adapters.proteinmpnn` for the protein-only chain export,
  chain-local position mapping, helper-compatible JSONL sidecars, request
  hashes, and generic request validation. The request declares `--omit_AAs C`,
  so cysteine is omitted during ProteinMPNN sampling.
- Official ProteinMPNN commit `8907e6671bfbfc92303b5f79c4b5e6ce47cdef57` was
  installed locally under `.var/tools/proteinmpnn` and used through an explicit
  `--proteinmpnn-root` path. The adapter invokes the public
  `protein_mpnn_run.py` CLI with helper-compatible parsed-PDB, chain-assignment,
  and fixed-position sidecars; fixed positions are ProteinMPNN chain-local
  sequence positions, not raw PDB residue numbers. The active backend batch is
  `eco1_rt_p25_5a_n96_20260624`: seeds `101`, `202`, `303`, temperatures
  `0.1` and `0.3`, and `num_seq_per_target: 16`.
- `sample_table.parquet` is materialized locally with 96 ProteinMPNN rows where
  `status=accepted`.
  rows. The named batch table is also retained at
  `sample_tables/eco1_rt_p25_5a_n96_20260624.parquet`.
- `candidate_table.parquet` is materialized locally with 96 rows where
  `status=accepted` and no protected-position or outside-mutable-position
  mutations. The
  named batch table is also retained at
  `candidate_tables/eco1_rt_p25_5a_n96_20260624.parquet`.
- `foldcheck_request/foldcheck_request_manifest.yaml` and
  `foldcheck_request/input_sequences.fasta` are materialized locally. The FASTA
  contains one WT baseline plus the 96 candidate-table rows where
  `status=accepted`, represented as full 320-aa canonical sequences. The
  request is intended for the ColabFold
  `colabfold_batch` CLI on BU SCC; LocalColabFold provides the pixi environment
  that exposes that command.
- `foldcheck_report.parquet` is materialized from BU SCC ColabFold full job
  `6228979`, run under `/project/dunlop/esouth/foldcheck/eco1_rt/full_96_a4948b42/`.
  The compact report was normalized on SCC and synced back locally. It covers
  the WT baseline plus all 96 ProteinMPNN candidates selected from
  candidate-table rows where `status=accepted`, with
  `accepted: 97` and `errored: 0`. The first full screen used
  `--num-models 1`; raw ColabFold output remains on SCC project storage.
  Summary metrics from the normalized report are pLDDT min/mean/max
  `88.634 / 91.167 / 93.200` and candidate C-alpha RMSD-to-WT min/mean/max
  `0.761 / 2.182 / 28.901` A.
- `foldcheck_review/` is materialized as a compact review bundle. It writes a
  96-row candidate ranking, a selected structure-panel manifest, a full
  local fold-structure manifest, a selected-panel ChimeraX command script, a
  full-fold-set ChimeraX command script, an Atlas subset manifest, and a visual
  review manifest. The visual manifest points to four SVG plots, a ChimeraX
  structure-overlay script, and a scoped marimo notebook with alt text and plain
  descriptions. The optional structure-overlay PNG is rendered only when an
  operator passes the explicit ChimeraX render flag. The structure overlay aligns WT
  and selected candidate model residues `3-311` to ec86kit/7V9U reference
  residues `1-309` over C-alpha atoms before rendering; this matches the
  mapped-residue review frame and avoids relying on an interactive ChimeraX
  alignment command. The notebook reads the manifest and presents the figures
  through a dropdown review surface with evidence rows and interpretation
  limits. These visuals summarize fold-review classes, ColabFold pLDDT/RMSD,
  cryoEM-reference RMSD, representative structure overlays, and Biohub ESMC SAE
  coverage; they are review aids, not candidate acceptance gates. The
  local full structure set contains one normalized PDB for the WT runtime model
  plus each of the 96 candidate models, copied from the SCC ColabFold run
  without pulling the full raw output tree. The ranking separates
  `wt_runtime_ca_rmsd` from `cryoem_mapped_ca_rmsd`: the former is candidate
  C-alpha RMSD to the WT ColabFold runtime model, while the latter is direct
  mapped-residue RMSD to the ec86kit/7V9U protein backbone. Because the model
  PDBs are now local, `cryoem_mapped_ca_rmsd_status` is `available` for all 96
  candidate rows. After Kabsch row-vector convention hardening, candidate
  cryoEM-reference mapped RMSD has min/mean/max `1.958 / 2.395 / 2.792 A`.
  Review classes are `strong_fold_preserved: 17`,
  `good_fold_preserved: 53`, `low_confidence: 9`, `review_band: 14`, and
  `structural_outlier: 3`.
- `review_deliverables/` is materialized as a study-owned visual bundle. It
  writes `review_deliverable_manifest.yaml`, a canonical-coordinate
  Mestre-derived clade 9 alignment panel, linear mask tracks, a ChimeraX
  mask-context script, ProteinMPNN diversity SVGs, a Tao-style
  ColabFold RMSD/pLDDT joint plot for the current single mask policy, WT ESMC
  model-constraint audit plots, an ESMC feature-review section, two
  interactive structure-browser manifests, and a manifest-backed marimo
  notebook. The notebook is organized as a progressive analysis surface:
  constraint evidence for the design mask, ProteinMPNN designs and fold triage,
  ESMC feature review, and a planned feasibility/handoff section. WT ESMC
  masked-marginal scoring is shown with the mask-constraint evidence as a
  review-only model-constraint audit, not as a current mask input. Structure
  views are selected through the same
  section/visual controls as static plots. In the reference section, the browser
  shows the off-white ec86kit/7V9U backbone and lets the reviewer switch among
  mask and motif highlight categories. Each mask category uses the same
  high-contrast highlight color because only one category is displayed at a
  time. Mask highlights are translated from canonical Ec86 residue positions to
  the exported backbone residue-number basis before rendering, so terminal
  missing-backbone residues do not shift the displayed selections. In the
  ProteinMPNN design and fold-triage section, the browser loads one selected
  PDB plus the ec86kit/7V9U reference into a browser-native 3D view for
  mouse-driven inspection. The notebook aligns each selected query model to the
  reference in memory over mapped C-alpha atoms before rendering, so local raw
  ColabFold PDB files remain unchanged. The selected structure view now shows a
  compact metric strip for mean pLDDT, WT-runtime C-alpha RMSD, direct cryoEM
  mapped C-alpha RMSD, sequence identity, and mutation count. These browser
  views use `dnadesign.thread.structure_views` and are review-only; ChimeraX
  remains the explicit opt-in still-render and pose-capture path. The
  notebook fits visual previews to the column and keeps full artifact paths in
  the evidence details. The manifest also links the existing foldcheck_review
  SVG/PNG visuals and Permuter-style WT ESMC masked-marginal SVGs instead of
  duplicating them. The
  model-constraint audit compares clade 9 WT plurality
  with ESMC masked-marginal entropy, best alternate LLR, and residue-wise
  constraint tracks. In the current WT table, plurality and entropy are
  inversely correlated (`Pearson r = -0.7838`, `R2 = 0.6144`). This is a review
  aid for future mask-policy discussion, not a current-mask update. The SAE
  feature-review section uses the existing all-97 Biohub ESMC sparse tables to
  show WT-active residue activation patterns and candidate/WT activation ratios
  for those same exact-dictionary features. It does not attach Atlas labels or claim
  processivity. The same section includes a joint review panel that places WT
  first, orders ProteinMPNN variants by SAE-feature similarity to WT, and shows
  ColabFold pLDDT plus summed WT masked-marginal single-substitution LLR as
  side markers. The LLR sum is a review covariate, not a joint protein
  likelihood, and the panel is not an activity or acceptance score. Mapping
  selected SAE feature activations onto the same interactive structure-view
  contract is a planned follow-on that requires an explicit feature/residue
  selection policy; it is not part of the current mask browser. Manifest
  paths are relative to the manifest location, so the visual bundle can move
  with the study workspace. The generated ChimeraX scripts use relative paths
  for staged local structures and keep raw remote paths out of the local review
  command path. The ChimeraX mask-context PNG is an opt-in render; ordinary
  materialization records a skipped optional-render status unless an existing
  PNG is being retained.
  Every deliverable row carries a display title, input hashes, alt text, a
  plain description, and an interpretation
  limit. WT model-constraint rows also carry the masked-marginal method summary
  and row-count evidence used by the marimo notebook. SVG outputs keep editable
  text nodes plus embedded title/description metadata.
- An all-97 ESM Atlas semantic-profile probe has been materialized through
  `dnadesign.thread.adapters.esm_atlas` and the thin Eco1 wrapper at
  `operations/materialization/atlas_semantic_profile/`. It uses
  the hash-lookup endpoint and writes compact study-local artifacts:
  `atlas_semantic_profile.parquet`, `atlas_protein_activations.parquet`,
  `atlas_residue_activations.parquet`, `atlas_feature_catalog.parquet`, and
  `structure_predictions/structure_prediction_registry.parquet`. A bounded
  `--allow-fold-on-miss --prediction-set-id atlas_esmfold_on_miss_all97_20260626`
  run selected WT plus all 96 candidates and allowed 5 new requests. WT was
  accepted with rich sparse Atlas data and one Atlas/ESMFold-derived structure
  registry row. The first four synthetic ProteinMPNN candidates still returned
  explicit Atlas HTTP 404 rows, and the remaining 92 candidates are marked
  `atlas_request_not_attempted_due_to_max_new_requests`. The sparse tables
  therefore still contain only the WT-derived 2,095 protein-level nonzero
  activations, 20,480 per-residue activations, and 100 feature-catalog rows.
  This is model-derived semantic context only. The no-auth Atlas hash-lookup
  path has not produced rich query-level SAE rows for synthetic candidates; use
  the sequence-similarity endpoint as a separate semantic-neighborhood artifact
  if synthetic-candidate Atlas context is needed without the authenticated
  Biohub ESMC/logits API.
- A Biohub ESMC/logits SAE profile is materialized separately through
  `dnadesign.thread.adapters.biohub_esmc` and the thin Eco1 wrapper at
  `operations/materialization/biohub_esmc_sae_profile/`. This uses the
  authenticated public API path `POST /api/v1/encode` followed by
  `POST /api/v1/logits`; it does not use Atlas hash lookup and does not run a
  fold model. The current conservative run selected WT plus all 96
  fold-report rows accepted by the validator with model `esmc-6b-2024-12` and SAE
  model `esmc-6b-2024-12-sae-layer60-k64-codebook16384`,
  `normalize_features=true`. The materialization accepted all 97 query rows.
  Each 320-residue
  sequence has 20,480 sparse per-residue activations, matching 64 active SAE
  features per residue. The generated tables contain 97 profile rows, 204,935
  protein-feature summary rows, 1,986,560 residue-feature rows, and 2,328
  observed feature-catalog rows. The request manifest stores the key label and
  redacted authorization only; it does not store the Biohub token. It now also
  records non-secret method references for the Biohub ESMC SAE feature
  interpretation notebook and the Biohub logits API. This is rich
  query-time SAE context for synthetic sequences, not fold validation,
  processivity evidence, or an acceptance gate.
  The sparse residue-feature tables are residue-only after BOS/EOS trimming:
  `token_count` is 322 for the 320-aa sequences, stored residue positions run
  from 1-320, and every residue has exactly 64 nonzero SAE features for the
  `k64` dictionary. Feature descriptions are fetched only for the exact
  Biohub 6B layer-60 16k dictionary used in this run. They are model-derived
  descriptions from the Biohub interpretation workflow, not curated functional
  annotations and not assay evidence.
- A WT-only Biohub ESMC masked-marginal mutation-scoring materializer is
  implemented at
  `operations/materialization/biohub_esmc_wt_mutation_scoring/`. It uses the
  same authenticated `POST /api/v1/encode` -> `POST /api/v1/logits` public API
  path but requests sequence logits rather than SAE activations. The full WT
  run is materialized for the 320-aa Ec86 sequence: 320 accepted position
  entropy rows, 6,080 accepted non-WT single-substitution LLR rows, a
  mask-context join, compact plots, and a redacted request manifest. The
  manifest records non-secret method references for the Biohub ESMC
  mutation-scoring notebook, the Biohub logits API, and the ESMC primary
  literature context. The method follows the Biohub notebook pattern: mask one
  residue, read sequence logits at the masked position, compute Shannon entropy
  in bits for the full returned vocabulary plus `canonical_entropy_bits` for
  canonical amino acids, and compute each alternate-residue LLR as
  `log P(alternate) - log P(WT)`. The `fraction_negative_alternate_llr` field is
  over the 19 non-WT canonical alternates, not over the WT residue. This is a
  model-constraint audit for future mask-policy review, not experimental deep
  mutational scanning and not a current-mask update.

### Mask Counts

Applying the current rule gives this row classification:

| Class | Count |
| --- | ---: |
| `non_fixed` mapped residues | 123 |
| `non_fixed_missing_backbone` terminal residues | 11 |
| evolutionarily conserved at >=25% WT plurality in the Ec86 clade 9 MSA | 106 |
| within 5 A retained DNA/RNA | 120 |
| NAxxH / YADD / VTG | 12 |
| Wang/Ec86 direct substrate-contact priors | 8 |

Total unprotected positions: `134`. Directly fixed-backbone ProteinMPNN mutable
positions from the current 7V9U backbone: `123`.

### Prior Mask Checks

`contact_risk_profile.yaml` remains an evidence review. It does not protect or
release residues under the current mask.

The previous 20 A all-fixed mask is diagnostic history: it showed that broad
retained-nucleic-acid proximity fixes the whole RT and is therefore too blunt
for Eco1. The current `mask_set.yaml` uses the direct 5 A rule.

### Validator Commands

Phase 0 scaffold validation:

```bash
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.contract_validation --repo-root . --phase phase0_scaffold
```

Phase 1 contract validation:

```bash
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.contract_validation --repo-root . --phase phase1_thread_contract
```

Phase 2 backend-ingest validation:

```bash
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.contract_validation --repo-root . --phase phase2_real_backend_ingest
```

Phase 3 fold-check report validation:

```bash
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.contract_validation --repo-root . --phase phase3_foldcheck_report
```

### Current Next Actions

1. Inspect the fold-check structure panel and review plots, especially the two
   high-RMSD outliers, low-pLDDT rows, and candidates where WT-runtime RMSD and
   cryoEM-reference RMSD tell different stories, before candidate selection.
2. Materialize a Biohub ESMC SAE feature-window summary before using SAE rows
   for stratification. The current Biohub SAE run uses the source-described
   `esmc-6b-2024-12-sae-layer60-k64-codebook16384` dictionary, so feature ids
   and descriptions now refer to the same model, layer, sparsity, and codebook.
   The Biohub SAE feature interpretation notebook should be treated as the
   method pattern for ranking features, locating residue-level activation peaks,
   and reviewing feature prevalence; it does not make model-derived feature
   descriptions into curated functional annotations.
3. Inspect the `review_deliverables/` marimo surface. It currently covers
   constraint evidence for the design mask, ProteinMPNN designs and fold triage
   with static and interactive structure views, ESMC feature review, and a
   planned feasibility/handoff gate. WT SAE structure frames and the Biohub ESMC
   feature-window heatmap remain downstream of `sae_feature_window_summary`.
4. Add a separate Atlas sequence-similarity materializer if synthetic-candidate
   Atlas neighborhood context is needed through the no-auth Atlas API. Do not
   keep retrying the hash-lookup/on-demand endpoint for the 96 synthetics unless
   the API behavior changes.
5. Build the assembly/synthesis feasibility report.
6. Build a candidate selection panel from fold-report rows accepted by the
   validator, structure review, feasibility review, and SAE annotation strata.
7. Select candidates only from rows with accepted fold-check coverage and
   feasibility review, then
   define the downstream RT-lnRNA candidate handoff accepted by
   `rt_lnrna_sponging_construct_triage`.

### Blockers

- `dnadesign.thread` now exposes generic ProteinMPNN request, sample-ingest,
  candidate-table, fold-check request/report contracts, and ColabFold output
  normalization. SCC LocalColabFold is installed under
  `/projectnb/dunlop/esouth/tools/localcolabfold`, and
  `colabfold_batch --help` succeeds when the pixi environment `lib/` directory
  is on `LD_LIBRARY_PATH`. This is local CLI execution, not a hosted ColabFold
  API and not the native DeepMind AlphaFold2 distribution.
- Atlas hash-lookup/on-demand coverage has been probed for the all-97 request.
  WT is accepted; the first synthetic candidate requests still return explicit
  404 rows even with `fold_on_miss=true`, and the remaining synthetic rows are
  intentionally unattempted. Atlas remains an optional post-fold semantic audit
  and stratification layer, not a candidate acceptance gate.
- No assembly feasibility report exists.
- No RT-only candidate handoff or RT-lnRNA acceptance record exists.

### Non-Goals

- Wet-lab protocol execution.
- Prime-editing campaign ownership.
- Replacing the RT-lnRNA sponging construct study.
- Hiding Eco1-specific biology inside a reusable tool package.

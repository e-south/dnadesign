---
doc_id: study-eco1-rt-repack-status
surface: study-record
study_id: eco1_rt_repack
owner: dnadesign-maintainers
last_verified: 2026-07-06
status_surface: record-only
---

## Eco1 RT Repack Status

### Current Phase

Selection readiness is materialized for the expanded design-class pool, and
RT-only `candidate_handoff.yaml` is not materialized. Phase 3 fold-check report
validation still passes locally for the baseline WT plus 96-candidate ColabFold
report and for the expanded design-class review bundle used for assay-panel
preparation. The study has the required structure, source, alignment,
conservation evidence, manual mask authority, mask set, explicit thread plan,
ProteinMPNN request, backend run manifests, sample/candidate tables, fold-check
requests, compact fold-check reports, and expanded review evidence under
`src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/`.
This is full fold-check and model-annotation coverage for candidate review, not
downstream candidate-handoff readiness.
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
  `status=accepted`. The named batch table is also retained at
  `sample_tables/eco1_rt_p25_5a_n96_20260624.parquet`.
- `candidate_table.parquet` is materialized locally with 96 rows where
  `status=accepted` and no protected-position or outside-mutable-position
  mutations. The
  named batch table is also retained at
  `candidate_tables/eco1_rt_p25_5a_n96_20260624.parquet`.
- `design_classes/` is materialized as an expansion request surface. It keeps
  the current 5 A class as the baseline and adds five class-specific
  ProteinMPNN request roots: clade 9 p25 contact 6/8/10 A, clade 9 p50 contact
  5 A, and II-A3/`42_1` p50 contact 5 A. Each generated class has its own
  `mask_set.yaml`, `thread_plan.yaml`, and `proteinmpnn_request/` sidecars.
  The aggregate `candidate_pool.parquet` now contains 576 nonredundant
  synthetic candidates: 96 from the baseline 5 A class plus 96 from each of the
  five added classes. Sequence hashes are unique across the pool.
- `foldcheck_request/foldcheck_request_manifest.yaml` and
  `foldcheck_request/input_sequences.fasta` are materialized locally. The FASTA
  contains one WT baseline plus the 96 candidate-table rows where
  `status=accepted`, represented as full 320-aa canonical sequences. The
  request is intended for the ColabFold
  `colabfold_batch` CLI on BU SCC; LocalColabFold provides the pixi environment
  that exposes that command.
- `design_classes/foldcheck_request/` and `design_classes/foldcheck_report.parquet`
  are materialized for WT plus all 576 expanded candidates. The normalized
  expanded report has one WT row and 576 accepted synthetic rows.
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
  alignment command. The notebook presents the figures through dropdowns with
  source rows and interpretation limits. These visuals summarize fold-review
  classes, ColabFold pLDDT/RMSD,
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
- `design_classes/foldcheck_review/foldcheck_candidate_ranking.parquet` is
  materialized for the 576-candidate expanded pool. Current review classes are
  `strong_fold_preserved: 280`, `good_fold_preserved: 188`,
  `low_confidence: 105`, and `review_band: 3`.
- `design_classes/selection/candidate_handoff_sequences.csv` is the flat
  reviewer-facing protein-sequence export for the selected six rows. It carries
  candidate ids, selection slots, protein sequences, sequence hashes, codon
  policy id, DNA-design status, and restriction-screen status. It is not an
  E. coli codon-optimized DNA design and has not passed restriction-site
  screening.
- `review_deliverables/` is materialized as the study visual bundle. It writes
  `review_deliverable_manifest.yaml`, the clade 9 and subtype MSA panels,
  design-class mask evidence, a ChimeraX mask-context script, baseline
  ProteinMPNN audit plots, an expanded design-class ProteinMPNN/ColabFold
  fold-validation plot, WT ESMC model-check plots, ESMC/SAE check plots, three
  interactive structure-browser manifests, expanded panel-selection plots, the
  selected-protein sequence CSV link, and a marimo notebook. The MSA panels show
  the 25% mask denominator, the 50% design-class threshold cue, and subtype
  membership in the clade 9 source set.
  The notebook is organized by
  scientific question: mask basis, sequence proposals and fold checks, panel
  selection, and ESMC/SAE checks. WT ESMC masked-marginal scoring appears with
  the mask evidence as a model check, not as a mask input. ESMC scores disagree
  across model sizes, and SAE windows remain WT-like across the expanded pool,
  so neither signal selects panel rows or implies activity.
  Structure views are selected through the same section/visual controls as
  static plots. The reference browser shows the off-white ec86kit/7V9U backbone
  and lets the reviewer switch among mask and motif highlight categories. The
  baseline ColabFold browser loads one selected PDB plus the ec86kit/7V9U
  reference into a browser-native 3D view. The panel-selection browser uses the
  expanded fold-check structure set and shows WT plus the six selected variants
  with fold metrics, mutation counts, MSA support, and near retained DNA/RNA or
  thumb-track chemistry fields. The structure side summary also exposes protein sequence fields
  for each selected candidate when the structure-browser manifest is regenerated
  from the current materializer. Query coordinates are aligned to the reference in memory
  over mapped C-alpha atoms, and local raw ColabFold PDB files are unchanged.
  Structure-control labels are section-specific: mask evidence category for the
  reference mask browser, fold-review group for baseline fold browsing, and
  design class for selected-panel browsing. Molecule-visibility toggles remain
  stable as a reviewer moves between sections.
  ChimeraX remains the explicit opt-in still-render and pose-capture path.
  Manifest paths are relative to the manifest location, so the visual bundle can
  move with the study workspace. The ChimeraX mask-context PNG is an opt-in
  render; ordinary materialization records a skipped optional-render status
  unless an existing PNG is being retained.
  Every deliverable row carries a display title, input hashes, alt text, a
  plain description, and an interpretation
  limit. WT model check rows also carry the masked-marginal method summary
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
  accepted with sparse Atlas data and one Atlas/ESMFold-derived structure
  registry row. The first four synthetic ProteinMPNN candidates still returned
  explicit Atlas HTTP 404 rows, and the remaining 92 candidates are marked
  `atlas_request_not_attempted_due_to_max_new_requests`. The sparse tables
  therefore still contain only the WT-derived 2,095 protein-level nonzero
  activations, 20,480 per-residue activations, and 100 feature-catalog rows.
  This is model-derived feature context only. The no-auth Atlas hash-lookup
  path has not produced candidate-level SAE rows for synthetic candidates; use
  the sequence-similarity endpoint as a separate model-neighborhood artifact
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
  interpretation notebook and the Biohub logits API. This is
  query-time SAE context for synthetic sequences, not fold validation,
  processivity evidence, or an acceptance gate.
  The sparse residue-feature tables are residue-only after BOS/EOS trimming:
  `token_count` is 322 for the 320-aa sequences, stored residue positions run
  from 1-320, and every residue has exactly 64 nonzero SAE features for the
  `k64` dictionary. Feature descriptions are fetched only for the exact
  Biohub 6B layer-60 16k dictionary used in this run. They are model-derived
  descriptions from the Biohub interpretation workflow, not curated functional
  annotations and not assay evidence.
- The expanded design-class Biohub ESMC/logits SAE profile is also materialized
  under `design_classes/`. It covers WT plus all 576 nonredundant synthetic
  candidates with 577 accepted profiles, 1,216,696 protein-feature rows, and
  11,816,960 residue-feature rows. This supports windowed model review and
  candidate triage; it is not a hard acceptance gate.
- The three-window SAE summary is materialized under
  `design_classes/biohub_esmc/sae_feature_window_summary.parquet`. It has 1,731
  rows: 577 sequences across the 23-position catalytic-palm control,
  120-position nucleic-acid contact surface, and 107-position mutable
  substrate-proximal annulus/basic-surface windows. It reports WT-delta
  activation summaries, signed feature deltas, and window-space redundancy;
  it is local model review evidence, not a selection gate.
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
  model check for future mask-policy review, not experimental deep
  mutational scanning and not a current-mask update. The 300M ESMC grid remains
  under `biohub_esmc/mutation_scoring/`; the materializer routes non-default
  models such as `esmc-6b-2024-12` to model-specific subdirectories so
  rescoring does not overwrite that grid. A separate 6B ESMC WT grid is now
  materialized under `biohub_esmc/mutation_scoring/esmc_6b_2024_12/` with 320
  accepted position rows, 6,080 accepted non-WT substitution rows, and request
  hash `sha256:2d2c5114e15734e51c5694a105aa2e43496218fae5e952c69b6dedb5601b3c2c`.
  The review-deliverables bundle derives separate 300M and 6B additive
  candidate LLR tables and plots from those grids. In the expanded
  576-candidate pool, the 300M additive total is positive for 505 candidates
  and negative for 71, with median `12.871`. The 6B additive total is negative
  for all 576 synthetic candidates, with median `-78.647`. These LLR lanes are
  model-review evidence only. They are not nonnegative eligibility thresholds,
  panel tie-breaks, or whole-protein likelihoods.

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

1. Review the expanded panel-selection bundle under
   `outputs/thread/design_classes/selection/`.
2. Treat `feasibility_report.parquet` as computational full-gene feasibility,
   not a synthesis quote or wet-lab assembly plan. All 576 expanded synthetic
   rows are currently feasible under this computational gate.
3. Use `candidate_triage_table.parquet` as the reviewer filter surface. It has
   468 eligible rows, 105 low-confidence rows marked ineligible, and 3
   review-band rows marked as manual reserve only. The regenerated table now
   adds MSA support, mutation geography near retained DNA/RNA or thumb-track,
   local chemistry, and source-artifact hashes for those review axes.
4. Use the six panel-selection SVGs under `selection/plots/` for notebook
   review: `selection_design_class_gate_counts`,
   `selection_class_local_percentiles`, `selection_six_sequence_distance`,
   `selection_selected_substitutions_across_rt`,
   `selection_regional_mutation_burden`, and
   `selection_na_facing_chemistry_balance`. They show gate counts by design
   class, within-class review percentiles, selected-row sequence distance,
   selected substitutions across RT regions, regional mutation burden, and
   chemistry changes near retained DNA/RNA or thumb-track positions.
5. Use `candidate_selection_panel.parquet` and
   `candidate_handoff_sequences.csv` as the current six-row RT-only protein
   review surface. The panel selects one feasible, fold-preserved
   representative from each design class by MSA support, mutation geography
   near retained DNA/RNA or thumb-track, local chemistry warnings, sequence
   nonredundancy, and fold metrics. The selected rows are all
   `strong_fold_preserved`:
   `thread_candidate_f8de74828ad8`,
   `thread_candidate_3b8ec09dffa4`, `thread_candidate_8145a7ffbfd6`,
   `thread_candidate_9545e08c9ab9`, `thread_candidate_b134f9a1f060`, and
   `thread_candidate_7d5861f87291`.
6. Keep SAE windowing as review evidence only. The current table records
   `wt_like_not_used_for_selection` for every synthetic row, so SAE does not
   nominate a mechanistic slot.
7. Emit `candidate_handoff.yaml` only after the selected rows, sequence hashes,
   and upstream artifact hashes are reviewed and accepted by the RT-only
   handoff contract.

### Blockers

- `dnadesign.thread` now exposes generic ProteinMPNN request, sample-ingest,
  candidate-table, fold-check request/report contracts, and ColabFold output
  normalization. SCC LocalColabFold is installed under
  `/projectnb/dunlop/esouth/tools/localcolabfold`, and
  `colabfold_batch --help` succeeds when the pixi environment `lib/` directory
  is on `LD_LIBRARY_PATH`. This is local CLI execution, not a hosted ColabFold
  API and not the native DeepMind AlphaFold2 distribution.
- Atlas hash-lookup/on-demand coverage has been probed for the baseline all-97 request.
  WT is accepted; the first synthetic candidate requests still return explicit
  404 rows even with `fold_on_miss=true`, and the remaining synthetic rows are
  intentionally unattempted. Atlas remains optional post-fold model-feature
  review context, not a candidate acceptance gate.
- RT-only `candidate_handoff.yaml` is not materialized.
- Downstream RT-lnRNA acceptance or rejection is not materialized.
- The review-deliverables marimo notebook includes the expanded panel-selection
  table and plots: gate counts, class-local percentiles, selected-row sequence distance,
  selected substitutions across RT regions, regional mutation burden, chemistry
  changes near retained DNA/RNA or thumb-track positions, the selected protein
  sequence CSV, the selected-panel structure browser, the candidate-handoff
  readiness checklist, and the compact selected-panel table.

### Non-Goals

- Wet-lab protocol execution.
- Prime-editing campaign ownership.
- Replacing the RT-lnRNA sponging construct study.
- Hiding Eco1-specific biology inside a reusable tool package.

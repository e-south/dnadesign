---
doc_id: study-eco1-rt-repack-status
surface: study-record
study_id: eco1_rt_repack
owner: dnadesign-maintainers
last_verified: 2026-07-09
status_surface: record-only
---

## Eco1 RT Repack Status

### Current Phase

Selection readiness is materialized for the generation-policy v2 pool, and v3
ProteinMPNN request manifests are staged for a rerun with upstream near-region
`omit_AA_jsonl` constraints. RT-only `candidate_handoff.yaml` is not
materialized. The compact Phase 3 fold-check contract validates locally, and
the generation-policy selection-readiness and review-deliverables surfaces are
current for protein-panel review of the last completed pool. The
review-deliverables bundle is materialized with a degraded optional-render
status: core notebook, selection, and structure-browser artifacts are current,
while optional ChimeraX PNG renders are reused or skipped unless explicitly
regenerated. The study has the required structure, source, alignment,
conservation evidence, manual mask authority, mask set, explicit thread plan,
ProteinMPNN request, backend run manifests, sample/candidate tables, fold-check
requests, compact fold-check reports, and review evidence under
`src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/`.
This is full v2 fold-check coverage plus existing model-annotation context for
candidate review, not downstream candidate-handoff readiness.
The active protected-position rule is:

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
  installed locally under `.var/tools/proteinmpnn` and used through explicit
  `--proteinmpnn-root` paths. The adapter invokes the public
  `protein_mpnn_run.py` CLI with helper-compatible parsed-PDB, chain-assignment,
  and fixed-position sidecars; fixed positions are ProteinMPNN chain-local
  sequence positions, not raw PDB residue numbers.
- `generation_policies_v3/` is staged for the next ProteinMPNN rerun. It
  contains `generation_policy_manifest.yaml`,
  `generation_policy_positions.parquet`, `generation_policy_alphabets.parquet`,
  and one ProteinMPNN request subtree per policy. The requested raw count is
  1008: 336 per policy. The near and combined policies use upstream
  `omit_AA_jsonl` sidecars for residue-specific near retained DNA/RNA
  alphabets; distal-only uses the global no-new-cysteine omission.
- `generation_policies_v2/` is the last completed generation-policy pool. It contains
  1007 nonredundant ProteinMPNN candidates sampled under complete policies:
  `distal_scaffold_repack_v1` with 335 rows,
  `near_dna_rna_acid_free_v1` with 336 rows, and
  `combined_near_acid_free_plus_distal_v1` with 336 rows. Each generated sample
  row carries one policy id/version/hash; downstream selection does not combine
  mutations across policies.
- `generation_policies_v2/foldcheck_request/`,
  `generation_policies_v2/foldcheck_report.parquet`, and
  `generation_policies_v2/foldcheck_review/` are materialized for the current
  v2 generation-policy pool. The compact v2 review bundle supplies the local
  structure set used by the active selected-panel structure browser.
- The v2 fold-review ranking separates `wt_runtime_ca_rmsd` from
  `cryoem_mapped_ca_rmsd`: the former is candidate C-alpha RMSD to the WT
  ColabFold runtime model, while the latter is direct mapped-residue RMSD to the
  ec86kit/7V9U protein backbone. Local PDBs are staged for browser review; raw
  ColabFold output trees remain outside the compact record.
- `generation_policies_v2/selection/candidate_handoff_sequences.csv` is the
  active flat reviewer-facing protein-sequence export for the selected
  primary-panel rows. It carries
  candidate ids, selection slots, protein sequences, sequence hashes, codon
  policy id, DNA-design status, and restriction-screen status. It is not an
  E. coli codon-optimized DNA design and has not passed restriction-site
  screening.
- `review_deliverables/` is materialized as the study visual bundle with
  degraded optional-render status. It writes `review_deliverable_manifest.yaml`,
  clade 9 and subtype MSA panels, a current mask-evidence structure browser,
  WT ESMC model-check plots, ESMC/SAE check plots, active v2 panel-selection
  plots, selected-panel structure browsing, the selected-protein sequence CSV
  link, and a marimo notebook. The notebook is organized by question: mask
  basis, sequence proposals and fold checks, panel selection, and ESMC/SAE
  checks. ESMC and SAE rows are model/method context; they do not select panel
  rows or imply activity. Structure views use stable section/visual controls,
  browser-native py3Dmol rendering, cached structure text reads, and source-row
  interpretation limits. ChimeraX remains the explicit opt-in still-render and
  pose-capture path. Manifest paths are relative to the manifest location, so
  the visual bundle can move with the study workspace. The ChimeraX mask-context
  PNG is an opt-in render; ordinary materialization records a skipped
  optional-render status
  unless an existing PNG is being retained.
  Every deliverable row carries a display title, input hashes, alt text, a
  plain description, and an interpretation
  limit. WT model check rows also carry the masked-marginal method summary
  and row-count evidence used by the marimo notebook. SVG outputs keep editable
  text nodes plus embedded title/description metadata.
- ESM Atlas and Biohub ESMC/SAE artifacts are model-check context only. Biohub
  ESMC materialization uses the authenticated public API path
  `POST /api/v1/encode` followed by `POST /api/v1/logits`; request manifests
  store redacted authorization metadata and do not store tokens. These artifacts
  do not run fold models, do not select panel rows, and do not support
  processivity or strand-displacement claims.
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
  candidate LLR tables and plots from those grids. These LLR lanes are
  model-review evidence only. They are not eligibility thresholds, panel
  tie-breaks, or whole-protein likelihoods.

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

### Generation-Policy Orthogonality Audit

The active v2 policies are complete ProteinMPNN generation policies, not
mutation bins. Every candidate is a full sequence produced under exactly one
policy. The three policy surfaces are:

| Policy | Candidate rows | Selected rows | Panel interpretation |
| --- | ---: | ---: | --- |
| `distal_scaffold_repack_v1` | 335 | 6 | Conservative distal scaffold repacking |
| `near_dna_rna_acid_free_v1` | 336 | 0 | Near-region rows did not pass current chemistry/support gates |
| `combined_near_acid_free_plus_distal_v1` | 336 | 0 | Combined rows did not pass current chemistry/support gates |

The selected panel preserves the declared Wang thumb-contact track, direct
retained DNA/RNA contacts, catalytic/retron motifs, and the C-terminal/thumb
primer-RNA recognition context. It does not support saying that the selected
panel directly tests thumb-track tuning, processivity, or strand displacement.

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

Current local result: passes with 0 issues. The compact fold-check report
matches request hash
`sha256:649937ef4e5cfe3dee57de8a99096645a265116616e4499ae6970f88702efc0d`
after refresh through the generic `dnadesign.thread.foldcheck` writer from the
validated normalized rows; do not hand-edit parquet metadata.

### Local-Structure Region Crosswalk

The selection-readiness local-structure gate uses explicit Eco1 residue sets and
derived contact shells. "Branch-recognition" is a shorthand for the protected
retron-initiation context represented here by the NAxxH and VTG motif contexts,
the YADD catalytic context, and retained DNA/RNA contact geometry; it is not a
separate branch-pocket model in this slice.

| Region id | Residue basis | Current local RMSD threshold |
| --- | --- | ---: |
| `retron_x_naxxh_context` | Eco1 residues 99-115 around NAxxH 105-109 | 1.25 A |
| `catalytic_initiation_context` | Eco1 residues 189-204 around YADD 195-198 | 1.50 A |
| `retron_y_vtg_context` | Eco1 residues 237-251 around VTG 243-245 | 1.60 A |
| `thumb_contact_track_context` | Wang/Ec86 positions 238, 239, 240, 249, 257, 261, 264, and 298; stricter preservation screen for documented thumb/RNA contact context | 2.50 A |
| `c_terminal_primer_rna_recognition_context` | Eco1 mapped residues 255-311 in the C-terminal primer-RNA recognition context; canonical residues 312-320 are missing backbone in the current 7V9U-backed fixed-backbone scope | 2.50 A |
| Near retained DNA/RNA region | Derived near retained DNA/RNA region: residues >5 A and <=10 A from retained DNA/RNA, excluding motif contexts, direct contacts, and thumb-track positions. Machine-readable id: `near_retained_dna_rna_annulus` | 3.00 A |
| `distal_scaffold_control` | Mapped residues outside motif contexts, direct contacts, the near retained DNA/RNA region, and thumb-track positions | 4.75 A |

All local RMSD values are computed after one global mapped C-alpha fit to the
ec86kit/7V9U-backed reference. Region-specific fitting is not used because it
would hide local shifts relative to the global fold.

The C-terminal primer-RNA recognition context is an overlapping review region,
not a mutually exclusive mutation bucket. It is included because Ec86
primer-RNA studies support treating this C-terminal/thumb region as a cognate
RNA-recognition context. The gate preserves local backbone geometry; it is not
an activity or specificity prediction.

### Current Next Actions

1. Review the active panel-selection bundle under
   `outputs/thread/generation_policies_v2/selection/`.
2. Treat `feasibility_report.parquet` as computational full-gene feasibility,
   not a synthesis quote or wet-lab assembly plan. All 1007 v2 synthetic
   rows are currently feasible under this computational gate.
3. Use `candidate_triage_table.parquet` as the reviewer filter surface. The
   current simplified funnel is 1007 accepted candidates, 855 preservation-pass
   rows, 335 chemistry/support-pass primary candidates, and 6 selected rows.
   The table records policy provenance, MSA support, mutation geography near
   retained DNA/RNA, thumb-track and C-terminal primer-RNA recognition contexts,
   local chemistry, local-structure gate fields, and source-artifact hashes for
   those review axes. Canonical mutation tokens are parsed strictly because
   mutation geography affects hard-gate status. Local-structure metrics must be
   available for each declared review region and pass the declared local
   C-alpha RMSD thresholds before a row can be panel-eligible. These thresholds
   are declared structural-preservation review cutoffs, not literature-calibrated
   activity boundaries. In the current v2 pool, 893 rows pass the local-structure
   gate and 114 rows exceed at least one local RMSD threshold; C-terminal/thumb
   local movement is the dominant threshold failure.
4. Use `local_structure_threshold_sensitivity.parquet` and
   `region_msa_support.parquet` as audit tables for the local RMSD gate and
   regional natural-sequence support.
5. Use the current panel-selection SVGs under `selection/plots/` for notebook
   review: `selection_primary_panel_sankey`,
   `selection_local_structure_stratification`,
   `selection_local_structure_threshold_sensitivity`,
   `selection_local_structure_by_region`,
   `selection_regional_mutation_burden`,
   `selection_na_facing_chemistry_balance`,
   `selection_regionwise_msa_support`,
   `selection_near_region_charge_sensitivity`,
   `selection_selected_substitutions_across_rt`,
   and `selection_six_sequence_distance`. They
   show the primary-panel funnel, local RMSD thresholds against the
   candidate population, local-threshold sensitivity, selected-row local RMSD
   by RT region, regional mutation burden, chemistry changes near retained
   DNA/RNA or thumb-track positions, region-wise MSA support including the
   C-terminal primer-RNA recognition context, near-region charge sensitivity,
   selected substitutions across RT regions, and selected-row mutation-set
   dissimilarity.
6. Treat `contexts/generation-policy-cleanup-dev-spec.md` as the v2 policy
   implementation record. The active run uses complete ProteinMPNN generation
   policies and does not assemble variants by merging mutations from separate
   policies.
7. Use `candidate_selection_panel.parquet` and
   `candidate_handoff_sequences.csv` as the current RT-only protein
   review surface. The panel is selected globally after preservation and
   chemistry/support gates. The preservation gate requires strong fold,
   protected/contact/thumb-track preservation, and one declared local RMSD gate
   table that includes the C-terminal primer-RNA recognition region. The
   chemistry/support gate requires no acidic gains near retained DNA/RNA and no
   unobserved proximal substitutions. Mutation-set dissimilarity, basic losses
   and Pro/Gly gains near retained DNA/RNA, regional MSA support, local RMSD
   values inside the gate, and fold metrics define the final rank order.
   Generation policy is context, not a quota. The final reduction is panel
   selection, not a global activity ranking.
   The selected rows are all `strong_fold_preserved`, all come from
   `distal_scaffold_repack_v1`, and have zero Wang thumb-contact-track edits,
   zero near retained DNA/RNA edits, zero near-region acidic gains, zero
   proximal unsupported substitutions, and zero C-terminal primer-RNA recognition
   context edits. The correct interpretation is: the current primary panel is a
   conservative distal-scaffold repack panel. It does not test direct
   thumb-track tuning or claim processivity or strand-displacement improvement.
8. Use `near_region_charge_sensitivity.parquet` and
   `charge_sensitivity_shortlist.parquet` as non-selector audit evidence. The
   current v2 pool contains near/combined policy rows with proximal basic gains,
   but they also carry acidic gains and the relaxed shortlist still includes
   basic-to-acidic charge reversals. The audit is useful for deciding whether a
   future chemistry rule or upstream alphabet should change; it does not make
   those rows primary-panel candidates.
9. Keep SAE windowing as review evidence only. ESMC/SAE annotation is not part
   of v2 selection and does not nominate a mechanistic slot.
10. Emit `candidate_handoff.yaml` only after the selected rows, sequence hashes,
   and upstream artifact hashes are reviewed and accepted by the RT-only
   handoff contract. The current `candidate_handoff_sequences.csv` marks
   protein rows as eligible for RT-only handoff review; it is not the completed
   downstream handoff file.

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
- The review-deliverables marimo notebook includes the active v2 panel-selection
  table and plots: gate counts, local-structure threshold audit views,
  primary-panel funnel, selected-row mutation-set dissimilarity, selected
  substitutions across RT regions, regional mutation burden, chemistry changes
  near retained DNA/RNA or thumb-track positions, region-wise MSA support, the
  selected protein sequence CSV, the selected-panel structure browser, the
  candidate-handoff readiness checklist, and the compact selected-panel table.

### Non-Goals

- Wet-lab protocol execution.
- Prime-editing campaign ownership.
- Replacing the RT-lnRNA sponging construct study.
- Hiding Eco1-specific biology inside a reusable tool package.

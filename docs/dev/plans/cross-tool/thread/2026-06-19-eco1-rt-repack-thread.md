---
doc_id: dev-thread-eco1-rt-repack-candidate-review
surface: cross-tool-dev-spec
study_id: eco1_rt_repack
owner: dnadesign-maintainers
last_verified: 2026-06-29
status: active_next_slice
primary_slice: assembly-feasibility-report-v1
---

## Eco1 RT Repack Candidate Review, Feasibility, And RT-Only Handoff

This spec is the current cross-tool development surface for the next Eco1 RT
repack slice. The study has already moved past the original tracer bullet:
ProteinMPNN sampling, candidate normalization, ColabFold fold checking,
fold-check review, local PDB staging, and all-97 Biohub ESMC query-time SAE
collection are materialized. The next engineering task is to make candidate
review deterministic enough for feasibility assessment and an RT-only handoff.

### Decision Summary

Primary implementation slice:

```text
assembly-feasibility-report-v1
```

The hard handoff eligibility rule is:

```text
accepted candidate row
+ accepted fold-check row
+ reviewed structure metrics
+ feasible synthesis row
+ upstream hash closure
-> RT-only candidate_handoff eligibility
```

Biohub ESMC/SAE data are semantic annotation and stratification evidence. They
must not become a hidden processivity score, a fold-validation substitute, or an
acceptance gate. A biochemical assay remains the only evidence for processivity,
strand displacement, or hairpin readthrough.

### Current Record Posture

Use the checked-in record as the source of truth:

- Status: `docs/studies/eco1_rt_repack/record/status.md`
- Dataset registry: `docs/studies/eco1_rt_repack/record/datasets.yaml`
- Route map: `docs/studies/eco1_rt_repack/routes/README.md`
- Fold policy: `docs/studies/eco1_rt_repack/contexts/fold-validation-policy.md`
- Feasibility policy:
  `docs/studies/eco1_rt_repack/contexts/synthesis-feasibility-policy.md`

The current runtime artifact root is:

```text
src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/
```

Materialized inputs for this slice:

- `conservation_alignments/`: accepted Clustal Omega alignments for the
  selected Ec86 clade 9 and II-A3/cluster42_1 profiles. The broad clade 9
  alignment has 303 records after QC; the focused II-A3/cluster42_1 alignment
  has 45 records. Use those counts exactly. Do not describe the clade 9
  denominator as "about 40" rows.
- `conservation_visualizations/`: existing MSA QC sidecars, position plots,
  selected-row overviews, and plurality/gap summaries from the generic
  `aligner.msa.visualization` API.
- `candidate_table.parquet`: 96 accepted ProteinMPNN candidates.
- `foldcheck_report.parquet`: WT plus all 96 candidates, all accepted.
- `foldcheck_review/foldcheck_candidate_ranking.parquet`: 96 candidate rows with
  ProteinMPNN metrics, ColabFold confidence, WT-runtime RMSD, direct
  ec86kit/7V9U mapped-residue RMSD, and review classes.
- `foldcheck_review/foldcheck_full_structure_set.yaml`: local normalized PDBs
  for WT plus all 96 candidates, with raw ColabFold trees still on BU SCC.
- `biohub_esmc_sae_profile.parquet`: Biohub ESMC/logits materialization
  accepted all 97 query rows.
- `biohub_esmc_protein_features.parquet`: 204,935 sparse per-sequence SAE
  feature summary rows.
- `biohub_esmc_residue_features.parquet`: 1,986,560 sparse per-residue SAE
  rows, equal to 97 sequences x 320 residues x 64 active features.
- `biohub_esmc_feature_catalog.parquet`: 2,328 observed feature indices from
  the exact current Biohub SAE dictionary. Labels and descriptions are fetched
  for this same dictionary.
- `biohub_esmc/mutation_scoring/`: implemented WT-only ESMC masked-marginal
  mutation-scoring lane. The full 320-position WT run is materialized with
  position entropy, 6,080 non-WT single-substitution LLR rows, mask-join,
  redacted manifest, and compact plot artifacts. This is a model-constraint
  audit, not an update to the current mask.

Missing by design:

- `feasibility_report.parquet`
- `candidate_selection_panel.parquet`
- `candidate_handoff.yaml`
- downstream RT-lnRNA accept/reject record

### Evidence Ladder

Use one question per model layer:

```text
cryoEM structure gives the scaffold
-> ProteinMPNN proposes fold-compatible sequence candidates on that scaffold
-> ColabFold asks whether those sequences still fold like the scaffold
-> Biohub ESMC/SAE asks how query-time model features change across candidates
-> ESM Atlas may add public-protein neighborhood context where available
-> biochemical assays decide processivity, strand displacement, and hairpin readthrough
```

Do not write that ProteinMPNN proves stability. Do not write that ColabFold
proves activity. Do not write that SAE features measure processivity.

The closest methods analogue is Tao-style fixed-backbone RT redesign: protect
functionally important or conserved residues, use ProteinMPNN to propose
sequences on the remaining design canvas, and use AlphaFold-family structure
prediction as a structural filter. Eco1 differs in the biological details:
Wang/Ec86/7V9U supplies the specific cryoEM scaffold and substrate-contact
context, Mestre clade 9 supplies the selected conservation denominator, and the
current mask uses a direct 5 A retained DNA/RNA contact rule rather than a broad
distance sweep.

### Owner Boundaries

| Surface | Owns | Does not own |
| --- | --- | --- |
| `eco1_rt_repack` | Eco1 structure authority, mask policy, feature-window interpretation, feasibility policy, selection policy, RT-only handoff intent. | Generic backend mechanics, RT-lnRNA construct subjects, wet-lab assay truth. |
| `dnadesign.thread` | Generic fixed-backbone mechanics: ProteinMPNN request/sample ingest, candidate table construction, fold-check contracts, ColabFold parsing, Atlas/Biohub adapters, structure-prediction registry. | Eco1 catalytic masks, Wang/Ec86 contact priors, assay claims, study-specific selection thresholds. |
| BU SCC | Heavy ColabFold runtime and raw prediction trees. | Candidate selection or local study policy. |
| Biohub/Atlas | External model APIs and public-protein neighborhood context. | Fold-check acceptance, processivity claims, or downstream handoff decisions. |
| downstream RT-lnRNA study | Accept/reject RT-only handoff into a paired construct context. | Eco1 repack mask policy or current candidate generation. |

### Method Wording

ProteinMPNN:

> ProteinMPNN was used as a fixed-backbone inverse-folding sampler. The selected
> Ec86 RT backbone was converted to helper-compatible ProteinMPNN inputs, fixed
> positions were supplied in chain-local one-indexed sequence coordinates,
> cysteine was omitted, and sampling used explicit seeds, temperatures, and
> `num_seq_per_target`. The output rows are sequence proposals, not stability
> measurements or functional evidence.

ColabFold:

> Fold checks were run on BU SCC with the ColabFold `colabfold_batch` command
> installed through LocalColabFold. The input FASTA contained WT Ec86 RT and the
> accepted ProteinMPNN candidates as full 320-aa canonical sequences. The first
> full screen used `--num-models 1`. `dnadesign` normalized ColabFold outputs
> into a compact fold-check report with runtime provenance, confidence metrics,
> and C-alpha RMSD fields. This asks whether designed sequences preserve the
> Ec86 RT fold; it does not measure RT activity.

Biohub ESMC/SAE:

> Biohub ESMC/logits was used to collect query-time SAE activations for WT plus
> all 96 fold-accepted candidate sequences. These rows are model-derived
> semantic annotation. They can support review and assay-panel stratification,
> but they cannot by themselves establish processivity, strand displacement, or
> structured-template readthrough.

Biohub ESMC masked-marginal mutation scoring:

> Biohub ESMC/logits can also be used on masked WT contexts to produce
> DMS-shaped in silico single-substitution scores. For Eco1 this lane is
> WT-only: 320 masked positions and 6,080 non-WT amino-acid substitutions. It
> is a model-constraint audit that can be compared with the current mask, not
> experimental deep mutational scanning and not a change to the current mask.

Method note:

> For each WT Ec86 position, the materializer replaces that residue with `_`,
> calls Biohub ESMC `/api/v1/encode` and `/api/v1/logits` with sequence logits
> enabled, and reads the logit vector at the masked residue. It records
> Shannon entropy in bits for the full returned vocabulary and
> `canonical_entropy_bits` for the canonical amino-acid subset. It computes
> each single-substitution LLR as `log P(alternate residue) - log P(WT
> residue)`. The stored `logit_residue_offset` records whether the logits
> include a beginning-of-sequence token. Negative LLR means the model assigns
> lower probability to the alternate residue than to the WT residue in that
> masked context. The stored `fraction_negative_alternate_llr` is computed over
> the 19 non-WT canonical alternates, not over the WT residue itself.

ESM Atlas:

> ESM Atlas lookup/on-demand probing remains separate from Biohub ESMC/logits.
> In the current all-97 probe, WT returned rich Atlas rows while synthetic
> ProteinMPNN candidates returned explicit hash-lookup failures or were left
> unattempted by the request cap. Do not treat the Atlas hash-lookup path as
> synthetic-candidate SAE coverage.

### Metric Ontology

Keep native fold metrics separate from Eco1-derived review metrics.

Native ColabFold metrics are model-confidence fields:

- global pLDDT summary;
- PAE summary when present in runtime output;
- runtime parameters and model count;
- model artifact and score artifact hashes.

Eco1-derived structure metrics are scaffold-preservation fields:

- `wt_runtime_ca_rmsd`: candidate C-alpha RMSD to the WT ColabFold model from
  the same runtime screen.
- `cryoem_mapped_ca_rmsd`: direct mapped-residue RMSD to the ec86kit/7V9U
  protein backbone.
- region-level RMSD fields for protected motifs or structural regions, only
  after those regions are declared in code and validated.

Interpretation rule:

```text
ColabFold confidence asks whether the model trusts its predicted fold.
Eco1-derived RMSD asks whether the candidate preserves the intended Ec86/7V9U-like scaffold.
```

A candidate should not pass because pLDDT is high. It can be eligible only when
the fold-check row is accepted, structure-review metrics are available, and no
protected-region or feasibility rule fails.

### SAE Feature Compatibility

Do not mix SAE dictionaries by feature id.

The earlier Ec86 hairpin/processivity feature notes refer to an Atlas-style
feature interpretation panel. Those feature ids must be treated as tied to a
specific model, layer, sparsity, and codebook. The current all-97 Biohub run
uses:

```text
model: esmc-6b-2024-12
sae_model: esmc-6b-2024-12-sae-layer60-k64-codebook16384
feature_dictionary_size: 16384
normalize_features: true
```

Therefore, feature ids and source-backed descriptions in the current Biohub
SAE review lane refer to the same 6B layer-60 16k dictionary. They remain
model-derived interpretation aids, not curated Eco1 functional annotations.

The next implementation should use two layers:

1. **Model-specific window summary.** Summarize current Biohub ESMC feature
   activations over declared Eco1 residue windows without turning feature
   descriptions into assay evidence.
2. **Interpreted feature panel.** Add labels or named biological roles only for
   feature ids whose interpretation is source-backed for the exact model and
   codebook.

### SAE Window Summary

Materialize:

```text
src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/biohub_esmc/sae_feature_window_summary.parquet
```

Use a subdirectory for new Biohub-derived summaries to avoid adding more flat
root-level files. Do not move the existing all-97 source tables in this slice.

One row per:

```text
candidate_id x feature_index x window_id
```

Required fields:

```yaml
candidate_id: string
sequence_hash: string
feature_panel_id: string
sae_model: string
feature_dictionary_size: int
feature_index: int
feature_label: string | null
window_id: string
window_start_0based: int
window_end_0based_exclusive: int
activation_sum: float
activation_mean: float
activation_max: float
activation_argmax_residue_0based: int | null
activation_argmax_residue_1based: int | null
coverage_fraction_above_threshold: float
fragment_count_above_threshold: int
peak_shift_vs_wt: int | null
wt_activation_sum: float | null
delta_vs_wt_sum: float | null
ratio_vs_wt_sum: float | null
zscore_across_candidates: float | null
window_status: string
interpretation_role: string
```

Declared Eco1 windows for the first pass:

| Window id | Residues, zero-based half-open | Purpose |
| --- | --- | --- |
| `nterm_fingers_0_53` | `0-53` | N-terminal/fingers-side model-feature shifts. |
| `fingers_palm_0_113` | `0-113` | Forward template-contact region review. |
| `primer_grip_71_121` | `71-121` | Template-primer positioning region review. |
| `motif_a_101_131` | `101-131` | Motif A catalytic-environment review. |
| `intermotif_136_148` | `136-148` | Inter-motif structural hub review. |
| `precatalytic_helix_171_195` | `171-195` | Active-site gating-region review. |
| `dxd_yadd_187_203` | `187-203` | YADD/DxD catalytic motif review. |
| `thumb_206_319` | `206-319` | C-terminal thumb/palm scaffold review. |
| `thumb_core_279_295` | `279-295` | Thumb-core peak-window review. |

Allowed status values:

```text
wt_like
enriched
depleted
shifted
fragmented
missing
not_interpreted
```

Forbidden column names:

```text
processivity_score
strand_displacement_score
hairpin_unwinding_score
activity_score
enzyme_efficiency_score
```

### Feasibility Report

Materialize the already-registered planned artifact:

```text
src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/feasibility_report.parquet
```

One row per ProteinMPNN candidate. WT may appear only as an optional baseline
row if the policy explicitly says so; handoff eligibility is candidate-only.

Required fields:

```yaml
candidate_id: string
sequence_hash: string
parent_sequence_id: string
parent_sequence_hash: string
mutation_count_total: int
mutation_count_mutable_region: int
mutation_count_protected_region: int
protected_mutation_violation_count: int
protected_mutation_violations_json: string
mutation_windows_json: string
max_mutation_window_length: int
max_mutation_window_mutation_count: int
mutation_window_density_max: float
nearest_parent_id: string | null
nearest_parent_distance_aa: int | null
nearest_parent_distance_fraction: float | null
parent_haplotype_id: string | null
parent_haplotype_distance_aa: int | null
synthesis_tier: string
synthesis_blockers_json: string
codon_policy_id: string | null
sequence_complexity_flags_json: string
feasibility_status: string
feasibility_reason: string
feasibility_policy_id: string
input_candidate_table_hash: string
input_mask_policy_hash: string
input_foldcheck_report_hash: string
created_at_utc: string
created_by: string
```

Allowed `synthesis_tier` values:

```text
easy
standard
difficult
blocked
unknown
```

Allowed `feasibility_status` values:

```text
feasible
review
blocked
missing_inputs
```

Fail-fast rules:

- exactly 96 candidate rows unless an explicit baseline policy adds WT;
- unique `candidate_id`;
- no missing sequence hashes;
- no `feasible` row with protected mutation violations;
- every candidate exists in `candidate_table.parquet`;
- every handoff-eligible candidate has an accepted fold-check row;
- synthesis economics remain study policy, not a generic `thread` decision.

### Candidate Selection Panel

Materialize:

```text
src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/selection/candidate_selection_panel.parquet
```

This table is the review surface between feasibility and handoff. It should not
replace `candidate_table.parquet`.

Required fields:

```yaml
candidate_id: string
eligible_for_handoff: bool
rank_within_eligible: int | null
fold_preservation_rank: int
cryoem_preservation_rank: int
feasibility_rank: int | null
semantic_stratum: string
selection_bucket: string
selection_reason: string
input_foldcheck_review_hash: string
input_feasibility_report_hash: string
input_sae_window_summary_hash: string | null
```

Allowed `selection_bucket` values:

```text
primary_candidate
structural_control
semantic_diversity_candidate
outlier_for_review
feasibility_blocked
excluded
```

Eligibility rule:

```text
candidate_table.status == accepted
AND foldcheck_report.status == accepted
AND foldcheck_review row is present
AND feasibility_report.feasibility_status == feasible
AND protected_mutation_violation_count == 0
AND required upstream hashes are present
```

SAE rows may diversify candidates inside the fold-accepted and feasible set.
They must not select a candidate on their own.

### RT-Only Candidate Handoff

Materialize the already-registered planned artifact:

```text
src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/candidate_handoff.yaml
```

The handoff is RT-only. It must not create an RT-lnRNA construct subject.

Required top-level fields:

```yaml
handoff_id: eco1_rt_repack_candidate_handoff_v1
handoff_kind: rt_only_candidate_handoff
study_id: eco1_rt_repack
subject_kind: reverse_transcriptase_protein_only
construct_subject_created: false
downstream_acceptance_required: true
source_artifacts: {}
selection_policy: {}
candidates: []
```

Validator rules:

- fail if `feasibility_report.parquet` is missing;
- fail if `candidate_selection_panel.parquet` is missing;
- fail if a selected candidate lacks an accepted fold-check row;
- fail if a selected candidate is feasibility-blocked;
- fail if any required upstream hash is missing;
- fail if any construct-subject id is emitted;
- fail if SAE is configured as an acceptance gate.

### Scientific Deliverable And Visualization Contract

The visual plan should tell the study in order, using one figure family per
question. Do not build a single overloaded plot. Write all new deliverables
under:

```text
src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/review_deliverables/
```

The deliverable root must contain `review_deliverable_manifest.yaml`. Every
persisted figure, frame, or notebook input must record:

- artifact path;
- input artifact hashes;
- source table or structure ids;
- alt text;
- plain description;
- interpretation limit;
- whether the panel is manuscript-facing, review-only, or optional/heavy.

Current foundation status: `review-deliverable-foundation-v1` is materialized.
It writes the visual manifest, canonical-coordinate MSA plurality/mask SVG,
linear mask-track SVG, ChimeraX mask-context script/render, ProteinMPNN
diversity SVGs, and a manifest-backed marimo notebook. It also links existing
foldcheck_review SVG/PNG visuals instead of duplicating them. The visual
manifests use manifest-relative paths so the review bundle can move with the
study workspace. Static marimo checks and HTML export are the dogfood path for
notebook resolution. Biohub ESMC feature-window heatmaps, WT SAE feature
structure frames, feasibility matrices, and handoff panels remain follow-on
deliverables.

#### Deliverable 1: MSA Plurality And Mask Context

Artifact group:

```text
msa_plurality_mask_panel/
```

Main outputs:

- `msa_plurality_mask_panel.svg`
- `msa_plurality_mask_panel.alt.md`
- optional `msa_plurality_mask_panel.html` for full-row inspection

Purpose:

```text
Show how the current conservation mask was derived and where it lands on the
Eco1/Ec86 sequence.
```

Inputs:

- `ec86_clade9_conservation_v1.aligned.fasta` with 303 aligned records after QC;
- `ec86_iia3_cluster42_1_conservation_v1.aligned.fasta` with 45 aligned records;
- `conservation_profile.parquet`;
- `rt-annotation-tracks.yaml`;
- `manual_mask_authority.yaml`;
- `mask_set.yaml`.

Rendering contract:

- The Eco1/Ec86 target row must be first.
- The main static figure should use a declared display subset, not silently plot
  all 303 clade 9 rows with unreadable labels.
- A full-row view belongs in HTML or marimo, not a dense static SVG.
- Column backgrounds should mark `>=25%` WT-plurality protected positions.
- Separate tracks should mark NAxxH, YADD, VTG, Wang/Ec86 direct-contact priors,
  retained DNA/RNA 5 A contacts, and the mutable design canvas.
- Y-axis labels must be source labels from the declared row metadata, not FASTA
  order guesses.

Interpretation limit:

```text
The MSA panel explains the conservation component of the mask. It does not
prove that conserved residues are functionally required in Eco1, and it does not
make the full Mestre roster the scoring denominator.
```

#### Deliverable 2: Linear Mask And 3D Scaffold Context

Artifact group:

```text
mask_structure_context/
```

Main outputs:

- `linear_mask_tracks.svg`
- `mask_structure_context.cxc`
- `mask_structure_context.png`
- optional `mask_structure_context.svg` if exported through a reproducible
  vector route

Purpose:

```text
Show which residues were fixed or protected, which residues remained designable,
and where those classes sit on the cryoEM-backed RT fold.
```

Rendering contract:

- Use an off-white RT cartoon as the neutral base.
- Color protected categories with a small categorical palette:
  - catalytic/motif anchors;
  - Wang/Ec86 direct-contact priors;
  - retained DNA/RNA 5 A contact residues;
  - clade 9 plurality-protected residues;
  - mutable design-canvas residues.
- Use a declared precedence or multi-track legend for residues with multiple
  protection reasons. Do not let overlapping colors hide mask reasons.
- The linear track and structure panel must use the same category names.
- RT1-RT7 labels do not blanket hard-fix residues. They may be shown as context
  spans, but not as blanket protected spans.

Interpretation limit:

```text
This panel shows design policy and structure context. It does not evaluate
candidate folding and does not claim activity.
```

#### Deliverable 3: ProteinMPNN Candidate Diversity

Artifact group:

```text
proteinmpnn_candidate_diversity/
```

Main outputs:

- `proteinmpnn_score_mutation_burden.svg`
- `sequence_identity_vs_score.svg`
- `mutation_density_by_position.svg`

Purpose:

```text
Show what ProteinMPNN changed relative to the Ec86/7V9U reference before fold
review.
```

Metrics:

- ProteinMPNN score and global score;
- sampling seed, temperature, and sample index;
- sequence identity to WT/Ec86 reference;
- mutation count;
- mutation density by canonical residue position;
- protected-position violation count, expected to remain zero.

Interpretation limit:

```text
Sequence identity is descriptive. Lower sequence recovery is not bad by itself
if fold-review metrics remain acceptable and protected residues are preserved.
```

#### Deliverable 4: ColabFold Structure Review Panels

Artifact group:

```text
colabfold_structure_review/
```

Main outputs:

- `structure_panel_best_worst.png`
- `structure_panel_best_worst_manifest.yaml`
- optional `structure_contact_sheet_all97.png`
- optional `structure_contact_sheet_all97_manifest.yaml`

Purpose:

```text
Show whether candidate structures preserve the Ec86 scaffold, and make outliers
visually inspectable without loading every model manually.
```

Rendering contract:

- Always render a small panel first: WT, top preserved candidates, high-RMSD
  outliers, low-pLDDT rows, and one or more deterministic controls.
- Render the all-97 contact sheet only as an optional/heavy artifact.
- Cache per-structure thumbnails by structure hash and view preset so reruns do
  not rerender unchanged PDBs.
- Keep full 3D PDB inspection available through the existing full-set ChimeraX
  script.
- Labels beneath each thumbnail should include only compact metrics:
  - `pLDDT`;
  - sequence identity to WT;
  - `cryoem_mapped_ca_rmsd`;
  - `wt_runtime_ca_rmsd` if space allows.
- Do not call this an ESMFold2 panel; the current full structure set is
  ColabFold output.

Performance note:

```text
Rendering 97 structures is feasible but should be treated as a cached contact
sheet job, not as an always-on notebook render. The marimo notebook should load
pre-rendered images and manifests, not rerender ChimeraX views interactively.
```

#### Deliverable 5: WT Biohub ESMC SAE Feature Structure Frames

Artifact group:

```text
biohub_esmc_wt_feature_frames/
```

Main outputs:

- `frames/<sae_model>__feature_<feature_index>.png`
- `biohub_esmc_wt_feature_frames_manifest.yaml`
- optional `biohub_esmc_wt_feature_frames.mp4`

Purpose:

```text
Use WT Ec86 as the control sequence and show where selected Biohub ESMC SAE
features activate on the protein structure.
```

Rendering contract:

- Use the WT ColabFold model or the ec86kit/7V9U reference consistently; record
  which one is used.
- Use an off-white protein base.
- Color residues by per-residue activation for one feature at a time.
- Frame labels must include feature index, SAE model id, and any source-backed
  feature label. If no label exists for the current model/codebook, write
  `unlabeled feature`.
- Select features by a declared rule: WT top features, highest fold-accepted
  variance, or source-backed polymerase labels for the exact SAE model. Do not
  mix labels across SAE dictionaries.

Interpretation limit:

```text
These frames show model-derived activation localization. They are not evidence
for processivity, strand displacement, or hairpin readthrough.
```

#### Deliverable 6: Biohub ESMC Variant Feature Heatmap

Artifact group:

```text
biohub_esmc_feature_heatmap/
```

Main outputs:

- `biohub_esmc_feature_window_heatmap.svg`
- `biohub_esmc_feature_window_heatmap.alt.md`

Purpose:

```text
Compare selected Biohub ESMC feature-window activations across WT and the 96
candidate variants.
```

Rendering contract:

- Rows are WT plus candidate variants.
- Default row order should be structural: sort candidates by
  `cryoem_mapped_ca_rmsd`, with WT first and outliers labelled.
- Columns should be a declared subset of features or feature windows:
  - WT top features;
  - high-variance features across fold-accepted candidates;
  - model-matched source-backed polymerase features if available.
- Do not show all feature indices.
- Values should be WT-normalized activation ratios or z-scores, with missing
  values explicit.
- Use this heatmap as semantic stratification after fold review, not as a hidden
  candidate selector.

Interpretation limit:

```text
The heatmap shows which model-derived feature activations are retained, shifted,
or depleted across candidates. It does not rank processivity.
```

#### Deliverable 7: Feasibility And Handoff Matrix

Artifact group:

```text
selection_and_feasibility/
```

Main outputs:

- `synthesis_feasibility_matrix.svg`
- `candidate_selection_panel.svg`

Purpose:

```text
Show which structurally reviewed candidates are feasible, selected, blocked, or
reserved as controls.
```

Inputs:

- `feasibility_report.parquet`;
- `candidate_selection_panel.parquet`;
- `candidate_handoff.yaml` when present.

This deliverable must not render until feasibility exists.

#### Notebook Surface

The marimo notebook should stay manifest-backed. It should expose dropdowns for:

- MSA and mask context;
- linear/3D mask context;
- ProteinMPNN diversity;
- fold-review structure panels;
- WT SAE feature frames;
- variant SAE heatmap;
- feasibility and handoff once present.

The notebook must read manifests and pre-rendered artifacts. It must not
hard-code plot paths, rerun Biohub requests, rerender all ChimeraX structures,
or implement selection logic inline.

### Implementation Tickets

1. **WT ESMC masked-marginal mutation scoring**
   - Generic DMS-grid utilities live under
     `dnadesign.permuter.src.scoring.esmc_masked_marginal/`.
   - The Eco1 materializer lives under
     `operations/materialization/biohub_esmc_wt_mutation_scoring/`.
   - Use WT only, not all 97 sequence backgrounds.
   - The full WT 320-position run writes `wt_position_entropy.parquet`,
     `wt_substitution_llr.parquet`, `wt_mutation_scoring_mask_join.parquet`, a
     redacted manifest with non-secret method references, and compact SVG plots
     with embedded title/description metadata.
   - Keep `--resume-existing`, `--max-new-requests`, and request spacing in
     future refreshes so the lane remains resumable and conservative with
     Biohub API usage.
   - Do not use this lane to update
     `eco1_rt_clade9_plurality25_direct_contact5a_v1`.

2. **SAE feature-window summary**
   - Add study-owned materializer:
     `operations/materialization/biohub_esmc_feature_windows/`.
   - Keep generic sparse-row utilities in `dnadesign.thread.adapters.biohub_esmc`
     only if they are not Eco1-specific.
   - Validate model id, SAE model id, dictionary size, row counts, and WT joins.

3. **Feasibility report**
   - Add materializer:
     `operations/materialization/feasibility_report/`.
   - Add contract package:
     `operations/contracts/feasibility/`.
   - Start with full-sequence candidate feasibility. Do not introduce pooled
     recombination until parent haplotypes and structural coupling checks are
     explicit.

4. **Selection panel**
   - Add materializer:
     `operations/materialization/candidate_selection_panel/`.
   - Reject SAE-only selection and missing feasibility/fold rows.

5. **RT-only handoff**
   - Add materializer:
     `operations/materialization/candidate_handoff/`.
   - Reuse generic handoff/hashing helpers only after the Eco1 shape is stable.
   - Keep downstream RT-lnRNA acceptance as a separate contract.

6. **Visual bundle extension**
   - Foundation materialized in `operations/materialization/review_deliverables/`
     with MSA plurality/mask context, linear mask tracks, a ChimeraX mask-context
     script/render, ProteinMPNN candidate diversity, linked fold-review SVG/PNG
     visuals, WT ESMC masked-marginal constraint visuals, exact-dictionary
     Biohub ESMC SAE interpretation plots, a joint SAE-similarity/fold/LLR
     review panel, and a manifest-backed marimo notebook.
   - Visual manifests must use manifest-relative paths, and notebook dogfood
     must include `marimo check` plus HTML export so missing linked media is
     caught before review.
   - ChimeraX command scripts should use paths relative to the script
     directory for staged local structures. Keep raw SCC paths as provenance in
     manifests, not as required paths inside the local review script.
   - SVG outputs should retain editable text nodes and include title/desc
     metadata plus manifest alt text. Display titles belong in the manifest so
     marimo remains a manifest-backed review surface rather than a second label
     registry.
   - Next visual extension starts after `sae_feature_window_summary.parquet`:
     WT SAE structure frames and the Biohub ESMC feature-window heatmap.
   - Treat the all-97 structure contact sheet and feature-frame video as
     optional/heavy deliverables with cached per-structure or per-feature
     intermediates.
   - Keep SVGs and PNGs alt-text-backed, manifest-recorded, and sequentially
     useful for a concise scientific methods/results narrative.

7. **Phase wording cleanup**
   - Use `phase3_foldcheck_report` for current fold-check validation.
   - Reserve `phase4_downstream_promotion` for RT-only handoff and downstream
     accept/reject readiness.

### Validation Plan

Run after implementation:

```bash
uv run pytest src/dnadesign/thread/tests src/dnadesign/studies/units/eco1_rt_repack/tests -q
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.contract_validation --repo-root . --phase phase0_scaffold
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.contract_validation --repo-root . --phase phase1_thread_contract
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.contract_validation --repo-root . --phase phase2_real_backend_ingest
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.contract_validation --repo-root . --phase phase3_foldcheck_report
uv run ruff check .
uv run ruff format --check .
uv run python -m dnadesign.devtools.architecture.boundaries --repo-root .
uv run python -m dnadesign.devtools.docs.checks --repo-root . --max-sor-age-days 92
bash .agents/skills/eco1-rt-repack-status/scripts/audit-eco1-rt-repack-status-skill.sh
uv run marimo check src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/review_deliverables/notebooks/eco1_review_deliverables.py
uv run marimo check src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/foldcheck_review/notebooks/eco1_foldcheck_review.py
uv run marimo export html src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/review_deliverables/notebooks/eco1_review_deliverables.py --no-include-code -o /tmp/eco1_review_deliverables.html -f
uv run marimo export html src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/foldcheck_review/notebooks/eco1_foldcheck_review.py --no-include-code -o /tmp/eco1_foldcheck_review.html -f
git diff --check
test ! -e outputs
```

Add focused tests for:

- SAE model/dictionary mismatch rejection;
- window summary peak/fragment calculations;
- feasibility rejection on protected mutations;
- handoff rejection when feasibility is missing;
- handoff rejection when construct subjects are emitted;
- selection rejection when SAE is the only positive evidence;
- deterministic reruns where output hashes should be stable.

### Definition Of Done

This slice is done when:

```text
sae_feature_window_summary.parquet exists and validates
feasibility_report.parquet exists and validates
candidate_selection_panel.parquet exists and validates
candidate_handoff.yaml exists and validates
review_deliverable_manifest.yaml exists and validates
MSA plurality/mask context visual renders from declared alignment inputs
linear-plus-3D mask context visual renders from declared mask/structure inputs
ProteinMPNN sequence-diversity visuals render from candidate_table.parquet
cached ColabFold structure-review panel renders from local staged PDBs
WT Biohub ESMC SAE feature frames render for a declared feature subset
Biohub ESMC feature-window heatmap renders from sae_feature_window_summary.parquet
selection visuals render from materialized feasibility/selection inputs
all visual manifests include alt text and interpretation limits
status.md, datasets.yaml, routes, and command groups name the new state
phase wording separates fold-check validation from downstream promotion
handoff validator proves no construct subject was created
```

The only final claim enabled by this slice is:

```text
These RT-only candidates are fold-checked, structure-reviewed, computationally feasible, and ready for downstream accept/reject review.
```

This slice still must not claim:

```text
These candidates are more processive.
These candidates have strand-displacement activity.
These candidates read through hairpins better.
```

Those claims require downstream experimental evidence.

### Source Roles

- Tao et al. supplies the fixed-backbone RT redesign method pattern:
  protect functional/conserved residues, generate RT sequence proposals, and
  structurally filter candidates. It does not define Eco1's biological objective.
- ProteinMPNN supplies the public fixed-backbone inverse-folding CLI and helper
  JSONL workflow.
- Wang et al. and 7V9U supply the Ec86 RT-msDNA/msrRNA cryoEM scaffold and
  substrate-contact context.
- ColabFold supplies the `colabfold_batch` structural-fidelity path used on
  BU SCC. LocalColabFold supplies the install/environment path for that CLI.
- Candido et al., Biohub ESMC, and ESM Atlas supply model-derived semantic
  representation context. They do not supply biochemical processivity evidence.
- The Biohub ESMC mutation-scoring notebook supplies the masked-marginal
  sequence-logit pattern: mask one residue, compute per-position entropy, and
  compute zero-shot single-substitution LLRs. It does not supply experimental
  DMS data or an Eco1 mask rule.

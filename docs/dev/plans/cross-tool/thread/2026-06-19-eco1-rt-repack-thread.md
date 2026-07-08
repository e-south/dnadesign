---
doc_id: dev-thread-eco1-rt-repack-candidate-review
surface: cross-tool-dev-spec
study_id: eco1_rt_repack
owner: dnadesign-maintainers
last_verified: 2026-07-08
status: active_handoff_slice
primary_slice: rt-only-candidate-handoff-v1
---

## Eco1 RT Repack Candidate Review, Feasibility, And RT-Only Handoff

This spec is the current cross-tool development surface for the Eco1 RT
repack review and RT-only handoff slice. The study has already moved past the original tracer bullet:
ProteinMPNN sampling, candidate normalization, ColabFold fold checking,
fold-check review, local PDB staging, Biohub ESMC query-time SAE collection, and
the expanded design-class candidate pool are materialized locally.

Selection readiness is now materialized for a six-row protein review panel from
the 576 synthetic candidates. The active task is to review those rows before
emitting the RT-only `candidate_handoff.yaml` through the handoff contract. The
panel path checks computational buildability, removes fold-risk candidates,
requires local-structure and near retained DNA/RNA chemistry/support checks,
and selects six globally without forcing one row per design class. These steps
prepare the protein review set; they do not predict improved strand
displacement.

### Scientific Flow

The panel path is:

```text
accepted candidate row
+ accepted fold-check row
+ reviewed structure metrics
+ feasible synthesis row
+ preservation and chemistry/support gates
+ nonredundant global panel selection
+ upstream hash closure
-> RT-only protein review eligibility
```

The panel-selection deliverables are materialized under
`outputs/thread/design_classes/selection/`:
`feasibility_report.parquet`, `candidate_triage_table.parquet`, and
`candidate_selection_panel.parquet`, plus the flat
`candidate_handoff_sequences.csv` protein sequence export. The current funnel
is 576 accepted candidates, 204 preservation-pass rows, 105
chemistry/support-pass rows, and 6 selected rows. Design classes are
mask-policy context, not panel quotas; the selected panel currently includes
five `contact10a` rows and one `contact8a` near-region basic-gain row. The
panel tie-breaks use mutation-set dissimilarity, local chemistry risk, regional
MSA support, local RMSD values inside the declared gate, and fold metrics
inside the eligible set. The review plots include the primary-panel funnel,
local-structure gate views, regional mutation/chemistry/MSA support views,
selected substitutions, and mutation-set dissimilarity. ESMC LLR and SAE
windows are retained as review evidence, but they are not used for selection.

Defer APBS, HADDOCK, AlphaFold3 complex modeling, MD, EVcouplings, Tranception,
Evo2, computational stability prediction, whole-protein ESMC
pseudo-likelihood, and global SAE clustering unless a later v1.1/v2 task opens
one of those lanes explicitly. For the first panel, stability is handled by
fold plausibility, build feasibility, and later empirical expression or
thermal-screen data when those data exist. Biohub ESMC/SAE data remain model
review evidence. They are not fold validation, processivity evidence, or
candidate acceptance gates. A biochemical assay remains the only evidence for
processivity, strand displacement, or hairpin readthrough.

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
- Baseline `candidate_table.parquet`: 96 accepted ProteinMPNN candidates from
  `eco1_rt_clade9_plurality25_contact5a_v1`.
- `design_classes/candidate_pool.parquet`: 576 nonredundant synthetic
  candidates, 96 per design class across the baseline 5 A policy plus clade 9
  p25 contact 6/8/10 A, clade 9 p50 contact 5 A, and
  II-A3/`42_1` p50 contact 5 A. Sequence hashes are unique.
- `design_classes/foldcheck_report.parquet`: WT plus all 576 expanded
  candidates, all accepted in the normalized fold-check report.
- `design_classes/foldcheck_review/foldcheck_candidate_ranking.parquet`: 576
  candidate rows with ProteinMPNN metrics, ColabFold confidence, WT-runtime
  RMSD, direct ec86kit/7V9U mapped-residue RMSD, and review classes. Current
  expanded fold-review counts are `strong_fold_preserved: 280`,
  `good_fold_preserved: 188`, `low_confidence: 105`, and `review_band: 3`.
- Baseline `foldcheck_review/foldcheck_full_structure_set.yaml`: local
  normalized PDBs for WT plus the first 96 baseline candidates, with raw
  ColabFold trees still on BU SCC. The expanded design-class fold-check root is
  the current source for the 576-row protein review panel.
- Baseline `biohub_esmc_sae_profile.parquet`: Biohub ESMC/logits
  materialization accepted all 97 query rows.
- `design_classes/biohub_esmc_sae_profile.parquet`: Biohub ESMC/logits
  materialization accepted WT plus the expanded 576-candidate pool.
- `design_classes/biohub_esmc_protein_features.parquet`: 1,216,696 sparse
  per-sequence SAE feature summary rows.
- `design_classes/biohub_esmc_residue_features.parquet`: 11,816,960 sparse
  per-residue SAE rows, equal to 577 sequences x 320 residues x 64 active
  features.
- `design_classes/biohub_esmc_feature_catalog.parquet`: observed exact-dictionary
  Biohub ESMC SAE feature rows for the expanded pool. Labels and descriptions
  must refer to the same 6B layer-60 16k dictionary.
- `design_classes/biohub_esmc/sae_feature_window_summary.parquet`: materialized
  three-window SAE summary with 1,731 rows, covering 577 sequences across the
  23-position catalytic-palm control, 120-position nucleic-acid contact surface,
  and 107-position mutable near retained DNA/RNA review windows.
- `biohub_esmc/mutation_scoring/`: implemented WT-only ESMC masked-marginal
  mutation-scoring lane. The full 320-position WT run is materialized with
  position entropy, 6,080 non-WT single-substitution LLR rows, mask-join,
  redacted manifest, and compact plot artifacts. This is a WT sequence-model
  check, not an update to the current mask.
- `biohub_esmc/mutation_scoring/esmc_6b_2024_12/`: implemented model-specific
  output root for the same WT masked-marginal grid under `esmc-6b-2024-12`.
  The materialized 6B grid has 320 accepted position rows and 6,080 non-WT
  single-substitution LLR rows. Non-default models route to model-specific
  roots so this rescore cannot overwrite the current 300M run.
- `review_deliverables/biohub_esmc_sequence_scoring/`: implemented standalone
  candidate-preference comparison from the complete WT ESMC
  single-substitution grid. It writes `biohub_esmc_variant_llr_scores.parquet`,
  `esmc_candidate_preference_vs_wt.svg`, and
  `biohub_esmc_sequence_scoring_manifest.yaml`. The score is an additive
  WT-context LLR sum, not a whole-protein pseudo-likelihood.
  When the 6B WT grid exists, the same bundle also writes a separate 6B
  candidate-preference table/plot under
  `review_deliverables/biohub_esmc_sequence_scoring/esmc_6b_2024_12/` plus a
  300M-versus-6B model-comparison plot. Treat both as model-derived review
  context. In the expanded 576-candidate run, the 300M additive total is
  positive for most candidates while the 6B additive total is negative for every
  synthetic candidate. Use these rows as model-review context only; do not
  require nonnegative 6B additive totals and do not use either model size to
  select panel rows.

Remaining handoff blockers:

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
sequences at the remaining mutable positions, and use AlphaFold-family structure
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
> the fold-accepted candidate sequences. Baseline coverage is WT plus 96
> candidates; the expanded design-class coverage is WT plus 576 candidates.
> These rows are model-derived feature context. They can support review and
> assay-panel planning, but they cannot by themselves establish
> processivity, strand displacement, or structured-template readthrough.

Biohub ESMC masked-marginal mutation scoring:

> Biohub ESMC/logits can also be used on masked WT contexts to produce
> DMS-shaped in silico single-substitution scores. For Eco1 this lane is
> WT-only: 320 masked positions and 6,080 non-WT amino-acid substitutions. It
> is a model check that can be compared with the current mask, not
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

Biohub ESMC candidate preference:

> The review-deliverables bundle now includes a standalone candidate-preference
> table and plot. For each ProteinMPNN candidate, it sums the already
> materialized WT-context single-substitution LLR values over that candidate's
> canonical mutations and also reports the per-mutation mean. This is useful for
> checking whether the proposed substitutions are more or less model-preferred
> than the WT residue at the same positions under the WT masked context. It is
> not a joint protein likelihood, not leave-one-out whole-protein
> pseudo-likelihood, and not an activity measurement.

Stability/developability boundary:

> Computational stability prediction is not part of the v1 selection funnel for
> the many-mutant Eco1 RT designs. The first panel uses fold plausibility,
> buildability, MSA support, mutation geography, nucleic-acid-facing chemistry,
> and sequence diversity.
> Expression, solubility, DSF/nanoDSF, basal RT activity, or a later explicit
> structure-energy review can be recorded when those data exist, but they are not
> required inputs for the current triage table.

ESM Atlas:

> ESM Atlas lookup/on-demand probing remains separate from Biohub ESMC/logits.
> In the current all-97 probe, WT returned Atlas rows while synthetic
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

The expanded design-class run now has a compact source table:

```text
src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/design_classes/biohub_esmc/sae_feature_window_summary.parquet
```

Use the existing sparse Biohub ESMC SAE tables as inputs. Do not rerun Biohub
requests for this summary. Do not cluster whole-protein SAE vectors for v1.

One row per:

```text
candidate_id x window_id
```

Required fields:

```yaml
candidate_id: string
sequence_hash: string
design_class_id: string | null
model: string
sae_model: string
feature_dictionary_size: int
window_id: string
window_label: string
residue_count: int
residue_positions_1based_json: string
window_purpose: string
window_vector_hash: string
wt_window_vector_hash: string
cosine_distance_to_wt: float
activation_sum: float
wt_activation_sum: float
activation_delta_sum_vs_wt: float
top5_signed_feature_deltas_json: string
nearest_candidate_id: string
nearest_candidate_window_cosine_distance: float | null
window_redundancy_rank: int | null
window_redundancy_group: string | null
window_status: string
used_for_selection: bool
method_id: string
interpretation_limit: string
```

Declared Eco1 windows for v1:

| Window id | Purpose |
| --- | --- |
| `catalytic_palm_control` | Negative-control window around the catalytic palm. It should remain WT-like; large shifts are concerning, not exciting. |
| `thumb_palm_na_binding_surface` | Mechanism-adjacent thumb/palm surface near nucleic-acid handling. Use only after feasibility and fold checks. |
| `mutable_near_retained_dna_rna_region` | Mutable near retained DNA/RNA review window and basic/polar surface context. This is a distance-defined protein-review window for hypothesis selection, not a strand-displacement proxy. |

Allowed status values:

```text
accepted
wt_control
```

Current selection posture:

```text
SAE window summaries remain WT-like in this pool and are not used for selection.
```

Future rule if a later pool separates in SAE-window space:

```text
SAE can nominate at most one review contrast only after feasibility and fold
gates pass. It cannot be an acceptance gate.
```

Forbidden column names:

```text
processivity_score
strand_displacement_score
hairpin_unwinding_score
activity_score
enzyme_efficiency_score
sae_master_score
```

### Candidate Triage Table

Materialized:

```text
src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/design_classes/selection/candidate_triage_table.parquet
```

This table is the reviewer-facing filter surface for the expanded candidate
pool. It combines existing fold and ESMC evidence with feasibility,
SAE-window, and sequence-diversity fields. It reports inclusion and exclusion
reasons without reducing them to one combined rank.

One row per nonredundant synthetic candidate. WT can appear in companion summary
views, but WT is not a selectable six-variant slot.

Required field groups:

```yaml
candidate_id: string
sequence_hash: string
design_class_id: string
mask_policy_id: string
mutation_count_total: int
sequence_distance_to_wt: int
nearest_selected_distance_aa: int | null
fold_review_class: string
mean_plddt: float
wt_runtime_ca_rmsd: float | null
cryoem_mapped_ca_rmsd: float | null
esmc_300m_additive_llr_total: float | null
esmc_6b_additive_llr_total: float | null
sae_window_status: string
sae_mechanistic_contrast_window_id: string | null
sae_mechanistic_contrast_rank: int | null
clade9_alt_observed_fraction: float | null
clade9_alt_frequency_mean: float | null
clade9_unobserved_mutation_count: int | null
clade9_rare_or_unobserved_mutation_count: int | null
subtype_alt_observed_fraction: float | null
subtype_alt_frequency_mean: float | null
subtype_unobserved_mutation_count: int | null
subtype_rare_or_unobserved_mutation_count: int | null
selection_support_profile_id: string
selection_support_alt_observed_fraction: float | null
selection_support_alt_frequency_mean: float | null
selection_support_unobserved_mutation_count: int | null
catalytic_or_direct_contact_mutation_count: int | null
nucleic_acid_facing_mutation_count: int | null
thumb_contact_track_mutation_count: int | null
distal_scaffold_mutation_count: int | null
nucleic_acid_facing_charge_delta: int | null
nucleic_acid_facing_basic_gain_count: int | null
nucleic_acid_facing_basic_loss_count: int | null
nucleic_acid_facing_acidic_gain_count: int | null
nucleic_acid_facing_proline_glycine_gain_count: int | null
nucleic_acid_facing_chemistry_warning_count: int | null
feasibility_status: string
hard_gate_status: string
hard_gate_failure_reasons_json: string
slot_candidate_status: string
input_candidate_pool_hash: string
input_foldcheck_review_hash: string
input_feasibility_report_hash: string
input_sae_window_summary_hash: string | null
input_conservation_profile_hash: string
input_clade9_alignment_hash: string
input_subtype_alignment_hash: string
input_contact_geometry_profile_hash: string
```

Allowed `hard_gate_status` values:

```text
eligible
ineligible
missing_inputs
```

Hard gates stay narrow:

- accepted candidate row;
- canonical mutation tokens parse without partial matching;
- zero protected-position mutations;
- zero catalytic or direct DNA/RNA-contact mutations;
- accepted fold-check row;
- available local-structure metrics for each declared RT review region;
- local C-alpha RMSD at or below the exploratory threshold for each declared
  RT review region;
- fold-review class allowed by the selection policy;
- feasible synthesis/buildability row;
- required upstream hashes present.

ESMC additive LLR and SAE windows remain review fields only. Panel tie-breaks
use natural-sequence support from the selected MSA denominator, mutation
geography, nucleic-acid-facing chemistry, sequence nonredundancy, and fold
metrics after feasibility and fold checks pass.

### Feasibility Report

Materialized for the expanded design-class pool:

```text
src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/design_classes/selection/feasibility_report.parquet
```

One row per nonredundant synthetic candidate in the selected candidate pool. WT
may appear only as an optional baseline row if the policy explicitly says so;
handoff eligibility is candidate-only.

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

- row count matches the selected nonredundant candidate pool unless an explicit
  baseline policy adds WT;
- unique `candidate_id`;
- no missing sequence hashes;
- no `feasible` row with protected mutation violations;
- every candidate exists in `candidate_table.parquet`;
- every handoff-eligible candidate has an accepted fold-check row;
- synthesis economics remain study policy, not a generic `thread` decision.

### Candidate Selection Panel

Materialized:

```text
src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/design_classes/selection/candidate_selection_panel.parquet
```

This six-row table records the proposed protein review panel between feasibility and handoff. It
does not replace `candidate_table.parquet` or `candidate_triage_table.parquet`.

Required fields:

```yaml
candidate_id: string
sequence_hash: string
design_class_id: string
eligible_for_handoff: bool
selection_slot: string
slot_rank: int | null
selected_for_panel: bool
selection_reason: string
tie_break_trace_json: string
nearest_selected_distance_aa: int | null
fold_review_class: string
feasibility_status: string
hard_gate_status: string
input_candidate_triage_table_hash: string
input_foldcheck_review_hash: string
input_feasibility_report_hash: string
input_sae_window_summary_hash: string | null
```

For v1, `selection_slot` is the design class represented by the selected row:

```text
eco1_rt_clade9_plurality25_contact5a_v1
eco1_rt_clade9_plurality25_contact6a_v1
eco1_rt_clade9_plurality25_contact8a_v1
eco1_rt_clade9_plurality25_contact10a_v1
eco1_rt_clade9_plurality50_contact5a_v1
eco1_rt_iia3_cluster42_1_plurality50_contact5a_v1
```

Eligibility rule:

```text
candidate_table.status == accepted
AND foldcheck_report.status == accepted
AND foldcheck_review row is present
AND feasibility_report.feasibility_status == feasible
AND protected_mutation_violation_count == 0
AND catalytic_or_direct_contact_mutation_count == 0
AND required upstream hashes are present
```

ESMC and SAE rows are recorded but do not select a candidate in the current
panel.

Deterministic tie-break order:

1. pass the preservation and chemistry/support gates;
2. avoid near retained DNA/RNA basic losses and Pro/Gly gains;
3. add mutation-set dissimilarity from already selected rows;
4. retain regional MSA support;
5. keep local RMSD values low inside the declared gate;
6. retain fold metrics inside the accepted fold class;
7. use deterministic hashes as the final tie-break.

The panel is a global conservative-diverse selection from the eligible pool,
not a top-six activity ranking and not one required row per design class.
Design classes remain useful context for where candidates came from:

| Design class | Current role |
| --- | --- |
| `eco1_rt_clade9_plurality25_contact5a_v1` | Input mask-policy context; not selected in the current primary panel. |
| `eco1_rt_clade9_plurality25_contact6a_v1` | Input mask-policy context; not selected in the current primary panel. |
| `eco1_rt_clade9_plurality25_contact8a_v1` | Input mask-policy context; contributes the current near-region basic-gain row. |
| `eco1_rt_clade9_plurality25_contact10a_v1` | Input mask-policy context; contributes five current selected rows. |
| `eco1_rt_clade9_plurality50_contact5a_v1` | Input mask-policy context; not selected in the current primary panel. |
| `eco1_rt_iia3_cluster42_1_plurality50_contact5a_v1` | Input mask-policy context; not selected in the current primary panel. |

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
- fail if a selected candidate mutates a catalytic or direct-contact position;
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

Current review-bundle status: `review-deliverable-foundation-v1` is materialized.
It writes the visual manifest, canonical-coordinate MSA plurality/mask SVG,
linear mask-track SVG, ChimeraX mask-context script/render, baseline
ProteinMPNN audit SVGs, and a
marimo notebook that reads the visual manifest. It links existing foldcheck_review SVG/PNG
visuals instead of duplicating them. The visual manifests use
manifest-relative paths so the review bundle can move with the study workspace.
Static marimo checks and HTML export are the dogfood path for notebook
resolution. Biohub ESMC feature-window heatmaps and WT SAE feature structure
frames are materialized figures. The panel-selection plots and table
show the expanded-pool panel decision, including a selected mutation
chemistry/geography map and a selected-panel py3Dmol browser from the expanded
fold-check structure set. RT-only handoff remains a downstream record, not a
notebook-derived decision.

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
- Static MSA figures should show the declared denominator rows. Full-row SVGs
  are acceptable when the notebook provides zoom/scroll and labels remain
  legible; do not apply an arbitrary row cutoff.
- Column backgrounds should mark `>=25%` WT-plurality protected positions.
- Separate tracks should mark NAxxH, YADD, VTG, Wang/Ec86 direct-contact priors,
  retained DNA/RNA 5 A contacts, and the mutable ProteinMPNN positions.
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

- `structure_panel_fold_review_examples.png`
- `structure_panel_fold_review_examples_manifest.yaml`
- optional `structure_contact_sheet_all97.png`
- optional `structure_contact_sheet_all97_manifest.yaml`

Purpose:

```text
Show whether candidate structures preserve the Ec86 scaffold, and make outliers
visually inspectable without loading every model manually.
```

Rendering contract:

- Always render a small panel first: WT, fold-preserved examples, high-RMSD
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

- `sae_feature_heatmap_manifest.yaml`
- marimo-rendered selected-feature heatmap

Purpose:

```text
Compare one selected WT-active Biohub ESMC SAE feature at a time across WT and
the 96 candidate variants.
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
- Use this heatmap as model review context after fold review, not as a hidden
  candidate selector.

Interpretation limit:

```text
The heatmap shows which model-derived feature activations are retained, shifted,
or depleted across candidates. It does not rank processivity.
```

#### Deliverable 7: Selection Readiness And RT-Only Sequence Export

Artifact group:

```text
design_classes/selection/
```

Main outputs:

- `feasibility_report.parquet`
- `candidate_triage_table.parquet`
- `candidate_selection_panel.parquet`
- `candidate_handoff_sequences.csv`
- `plots/selection_primary_panel_sankey.svg`
- `plots/selection_design_class_contrast.svg`
- `plots/selection_local_structure_stratification.svg`
- `plots/selection_local_structure_by_region.svg`
- `plots/selection_local_structure_threshold_sensitivity.svg`
- `plots/selection_premise_alignment.svg`
- `plots/selection_selected_substitutions_across_rt.svg`
- `plots/selection_regional_mutation_burden.svg`
- `plots/selection_na_facing_chemistry_balance.svg`
- `plots/selection_regionwise_msa_support.svg`
- `plots/selection_six_sequence_distance.svg`

Purpose:

```text
Show how accepted candidates pass preservation and chemistry/support gates,
then explain the global six-row primary panel with mutation-set dissimilarity,
regional substitutions, local RMSD, region-wise MSA support, chemistry fields,
and the flat protein-sequence export.
```

Inputs:

- `feasibility_report.parquet`;
- `candidate_triage_table.parquet`;
- `candidate_selection_panel.parquet`;
- `candidate_handoff_sequences.csv`;

These plots render only after feasibility, triage, and the selection panel are
materialized. They do not create handoff eligibility by themselves.

#### Notebook Surface

The marimo notebook should read the manifest and expose dropdowns for:

- MSA and mask context;
- linear/3D mask context;
- ProteinMPNN diversity;
- fold-review structure panels;
- WT SAE feature frames;
- variant SAE heatmap;
- panel-selection plots, selected protein sequences, py3Dmol structure views,
  and the candidate-handoff readiness checklist.

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

2. **Candidate ESMC additive LLR review**
   - The review-deliverables materializer writes:
     `review_deliverables/biohub_esmc_sequence_scoring/biohub_esmc_variant_llr_scores.parquet`
     `review_deliverables/biohub_esmc_sequence_scoring/biohub_esmc_sequence_scoring_manifest.yaml`
     and
     `review_deliverables/biohub_esmc_sequence_scoring/esmc_candidate_preference_vs_wt.svg`.
   - The current method is `esmc_additive_wt_single_substitution_llr_v1`.
   - A 6B WT-grid rescore uses the same WT-context masked-marginal additive
     calculation but writes a separate method id,
     `esmc_6b_2024_12_additive_wt_single_substitution_llr_v1`, and a separate
     review-deliverables subdirectory. Compare 300M and 6B ranks with the
     emitted model-comparison plot before using either table as explanatory
     review context. Do not require nonnegative 6B additive totals under the current
     rescore because every synthetic candidate is negative on that absolute
     scale.
   - Keep this separate from SAE feature activations and from future
     leave-one-out whole-protein pseudo-likelihood. Do not label it as joint
     protein likelihood.

3. **SAE feature-window summary**
   - Use the materialized study-owned materializer:
     `operations/materialization/sae_window_summary/`.
   - Keep generic sparse-row utilities in `dnadesign.thread.adapters.biohub_esmc`
     only if they are not Eco1-specific.
   - Validate model id, SAE model id, dictionary size, row counts, and WT joins.
   - Emit only the three v1 windows declared above.
   - Compute cosine distance to WT, summed activation delta to WT, top 5 signed
     feature deltas, and window-space redundancy. Do not emit a long feature
     interpretation narrative.

4. **Selection readiness**
   - Materialized in `operations/materialization/selection_readiness/`.
   - Emits `selection/feasibility_report.parquet`,
     `selection/candidate_triage_table.parquet`,
     `selection/candidate_selection_panel.parquet`, and a manifest under the
     expanded design-class output root.
   - Joins candidate pool, fold-review rows, MSA support, mutation geography,
     nucleic-acid-facing chemistry, sequence diversity, ESMC additive LLR rows,
     SAE windows, and feasibility.
   - Rejects missing required inputs. Does not compute or store a combined
     score. Does not use ESMC or SAE as selection gates or panel tie-breaks.

5. **RT-only handoff**
   - Add materializer:
     `operations/materialization/candidate_handoff/`.
   - Reuse generic handoff/hashing helpers only after the Eco1 shape is stable.
   - Keep downstream RT-lnRNA acceptance as a separate contract.

8. **Visual bundle extension**
   - Foundation materialized in `operations/materialization/review_deliverables/`
     with MSA plurality/mask panels, design-class mask evidence, a ChimeraX
     mask-context script/render, ProteinMPNN candidate diversity, linked
     fold-review SVG/PNG visuals, WT ESMC masked-marginal constraint visuals,
     exact-dictionary Biohub ESMC SAE interpretation plots, an interactive
     selected-feature SAE activation heatmap, and a manifest-driven marimo
     notebook.
   - Visual manifests must use manifest-relative paths, and notebook dogfood
     must include `marimo check` plus HTML export so missing linked media is
     caught before review.
   - ChimeraX command scripts should use paths relative to the script
     directory for staged local structures. Keep raw SCC paths as provenance in
     manifests, not as required paths inside the local review script.
   - SVG outputs should retain editable text nodes and include title/desc
     metadata plus manifest alt text. Display titles belong in the manifest so
     marimo reads manifest titles rather than maintaining a second label
     registry.
   - Next visual extension starts after `sae_feature_window_summary.parquet`:
     WT SAE structure frames and the Biohub ESMC feature-window heatmap.
   - Treat the all-97 structure contact sheet and feature-frame video as
     optional/heavy deliverables with cached per-structure or per-feature
     intermediates.
   - Keep SVGs and PNGs alt-text-backed, manifest-recorded, and sequentially
     useful for a concise scientific methods/results narrative.

9. **Deferred v1.1/v2 metrics**
   - Whole-protein ESMC pseudo-likelihood is implemented as a resumable method
     surface but is not required for the first protein review panel. A full expanded run
     is request-heavy: WT plus 576 candidates at 320 positions requires 184,640
     masked sequence-logit calls.
   - Defer APBS, HADDOCK, AlphaFold3 complex modeling, MD, EVcouplings,
     Tranception, Evo2, computational stability prediction, and global SAE
     clustering unless a later task opens one of those lanes with a specific
     question.

10. **Phase wording cleanup**
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

### Selection-Readiness Done State

The current selection-readiness slice is done when:

```text
sae_feature_window_summary.parquet exists and validates
feasibility_report.parquet exists and validates
candidate_triage_table.parquet exists and validates
candidate_selection_panel.parquet exists and validates
candidate_handoff_sequences.csv exists and validates
review_deliverable_manifest.yaml exists and validates
MSA plurality/mask context visual renders from declared alignment inputs
linear-plus-3D mask context visual renders from declared mask/structure inputs
ProteinMPNN sequence-diversity visuals render from candidate_table.parquet
cached ColabFold structure-review panel renders from local staged PDBs
Biohub ESMC feature-window heatmap renders from sae_feature_window_summary.parquet
candidate triage table exposes feasibility, SAE-window, ESMC, fold-review, and sequence-diversity fields without a combined rank
selection visuals render from materialized feasibility/selection inputs
selected protein sequence CSV is visible in the review notebook and referenced by handoff contracts
all visual manifests include alt text and interpretation limits
status.md, datasets.yaml, routes, and command groups name the new state
phase wording separates fold-check validation from downstream promotion
```

This state supports only this claim:

```text
These six rows form a global RT-only protein review panel with fold, feasibility, sequence, regional, chemistry, and local-structure evidence attached.
```

### RT-Only Handoff Done State

The downstream handoff slice is done only when:

```text
candidate_handoff.yaml exists and validates
candidate_handoff.yaml links the selected sequence CSV and upstream artifact hashes
handoff validator proves no construct subject was created
downstream acceptance remains required and separate from Eco1 selection readiness
```

Only that later state enables this claim:

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

| Source | Identifier | Role in this study |
| --- | --- | --- |
| Tao et al. 2026 | DOI [`10.1038/s41587-026-03149-6`](https://doi.org/10.1038/s41587-026-03149-6) | Fixed-backbone RT redesign method prior. |
| Mestre et al. 2020 | DOI [`10.1093/nar/gkaa1149`](https://doi.org/10.1093/nar/gkaa1149) | Retron RT roster and Ec86 clade context. |
| Simon et al. 2019 | DOI [`10.1093/nar/gkz865`](https://doi.org/10.1093/nar/gkz865) | RT motif grammar and annotation prior. |
| Wang et al. 2022 | DOI [`10.1038/s41564-022-01197-7`](https://doi.org/10.1038/s41564-022-01197-7); PDB `7V9U` | Ec86 RT-msDNA/msrRNA scaffold and substrate-contact context. |
| Inouye et al. 1999 | DOI [`10.1074/jbc.274.44.31236`](https://doi.org/10.1074/jbc.274.44.31236) | Primary Ec86 prior for primer-template RNA recognition and branch-G initiation context; not activity prediction. |
| Inouye et al. 2004 | DOI [`10.1074/jbc.M408462200`](https://doi.org/10.1074/jbc.M408462200) | Primary Ec86 prior for C-terminal/thumb primer-RNA binding context; not a reason to relax thumb-track protection. |
| ProteinMPNN | `dauparas/ProteinMPNN`; DOI [`10.1126/science.add2187`](https://doi.org/10.1126/science.add2187) | Public fixed-backbone inverse-folding CLI and helper-file workflow. |
| ColabFold | DOI [`10.1038/s41592-022-01488-1`](https://doi.org/10.1038/s41592-022-01488-1) | Public `colabfold_batch` fold-check command path. |
| Candido et al. 2026 | DOI [`10.64898/2026.06.03.729735`](https://doi.org/10.64898/2026.06.03.729735) | ESMC, ESMFold2, Atlas, and SAE representation context. |
| Biohub ESMC docs and notebooks | `/api/v1/encode`, `/api/v1/logits`, ESMC SAE and mutation-scoring notebooks | Public intended API and notebook patterns for query-time SAE and masked-marginal LLR review evidence. |

- Tao et al. supplies the fixed-backbone RT redesign method pattern:
  protect functional/conserved residues, generate RT sequence proposals, and
  structurally filter candidates. It does not define Eco1's biological objective.
- ProteinMPNN supplies the public fixed-backbone inverse-folding CLI and helper
  JSONL workflow.
- Wang et al. and 7V9U supply the Ec86 RT-msDNA/msrRNA cryoEM scaffold and
  substrate-contact context.
- ColabFold supplies the `colabfold_batch` structural-fidelity path used on
  BU SCC. LocalColabFold supplies the install/environment path for that CLI.
- Candido et al., Biohub ESMC, and ESM Atlas supply model-derived feature
  representation context. They do not supply biochemical processivity evidence.
- The Biohub ESMC mutation-scoring notebook supplies the masked-marginal
  sequence-logit pattern: mask one residue, compute per-position entropy, and
  compute zero-shot single-substitution LLRs. It does not supply experimental
  DMS data or an Eco1 mask rule.

---
doc_id: study-eco1-rt-repack-fold-validation-policy
surface: study-context
study_id: eco1_rt_repack
owner: dnadesign-maintainers
last_verified: 2026-07-06
---

## Fold Validation Policy

Fold validation is a computational QA gate, not proof of improved stability or
function. A candidate can advance only when the fold-check report states which
runtime produced the prediction, which starting structure was compared, and
which thresholds were applied.

The first Eco1 implementation runs the ColabFold `colabfold_batch` command on
BU SCC. `LocalColabFold` is the SCC install and environment path that exposes
that command through a pixi environment; it is not a separate modeling method.
This is command-line ColabFold execution, not the ColabFold notebook, not a
hosted API, and not a claim that the exact DeepMind AlphaFold2 distribution has
been run. ColabFold is the first backend because it accepts batch FASTA input,
fits the SCC runtime model, and covers the same structural-fidelity role that
AlphaFold2 served after ProteinMPNN design in Tao-style work. The contract stays
backend-neutral so later AlphaFold2, AlphaFold3, Boltz, or other fold runtimes
can write the same normalized report fields.

The materialized fold-check request is:

- `src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/foldcheck_request/input_sequences.fasta`
- `src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/foldcheck_request/foldcheck_request_manifest.yaml`
- `docs/bu-scc/jobs/eco1-colabfold-foldcheck.qsub` for SCC smoke/full
  ColabFold execution from that manifest.

Completed ColabFold output directories are normalized by
`src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/foldcheck_report/`,
which calls the generic `dnadesign.thread.adapters.colabfold` parser and writes
the compact `foldcheck_report.parquet` artifact. This parser does not run a
fold model and does not copy raw SCC output trees into the repository.

The full SCC run, job `6228979`, folded the WT baseline plus all 96 accepted
ProteinMPNN candidates with `--num-models 1`. Its raw output remains on SCC
project storage under
`/project/dunlop/esouth/foldcheck/eco1_rt/full_96_a4948b42/`. The compact
local report was normalized from that raw output and contains 97 accepted rows:
WT plus all 96 candidates. The earlier six-sequence smoke run, job `6224446`,
remains runtime-path history, not the current fold-coverage artifact.

The FASTA contains one WT baseline plus accepted ProteinMPNN candidates as
full 320-aa canonical Eco1 sequences. Terminal positions without 7V9U backbone
coordinates are retained as WT residues in the fold-check sequence; they were
not directly mutated by fixed-backbone ProteinMPNN.

### Methods-Ready Wording

Fold checks were run on BU SCC with the ColabFold `colabfold_batch` command,
installed through LocalColabFold in a pixi environment. The input was a FASTA
containing WT Ec86 RT and accepted ProteinMPNN candidate sequences as full
320-aa canonical sequences. For the first full screen, ColabFold was run with
`--num-models 1` to produce one ranked model per sequence. Raw ColabFold output
directories remained on SCC project storage. `dnadesign` normalized the PDB and
score outputs into `foldcheck_report.parquet`, recording runtime provenance,
parameter hashes, pLDDT, PAE summaries when present, and C-alpha RMSD against
the WT runtime baseline or declared reference. This step asks whether designed
sequences preserve the Ec86 RT fold; it does not measure RT activity,
processivity, strand displacement, or hairpin readthrough.

The review artifact uses explicit metric names. `wt_runtime_ca_rmsd` is the
candidate model's C-alpha RMSD to the WT ColabFold runtime model. It is useful
for finding candidate outliers inside the same ColabFold run, but it is not a
direct cryoEM comparison. `cryoem_mapped_ca_rmsd` is reserved for direct
mapped-residue comparison to the ec86kit/7V9U protein backbone and is populated
only when candidate PDBs are available locally. The current review bundle stages
one normalized PDB for WT plus each of the 96 candidates, so this direct
cryoEM-reference field is available for all candidate rows. If a future review
is run before staging model PDBs locally, the table records
`model_artifact_not_local` instead of silently substituting another reference.

Fold-review plots are generated from the same ranking table and Biohub ESMC
profile summary. Each plot is recorded in `review_visual_manifest.yaml` with alt
text, a plain description, data-source paths, and an interpretation limit. The
marimo notebook under `foldcheck_review/notebooks/` reads that manifest instead
of hard-coding figure paths. The notebook presents those plots through a
dropdown, then shows the selected image with source rows and an
interpretation-limit accordion. This keeps visual review scoped to the study
workspace and keeps the distinction clear: figures summarize
model-derived metrics, but selection still requires feasibility and handoff
review.

### Evidence Ladder

This study uses model outputs to answer different questions, not to collapse
all evidence into one score:

```text
cryoEM structure gives the scaffold
-> ProteinMPNN proposes fold-compatible sequence candidates on that scaffold
-> ColabFold asks whether those sequences still fold like the scaffold
-> Biohub ESMC/SAE asks how query-time model features change across candidates
-> ESM Atlas may add public-protein neighborhood context where available
-> biochemical assays decide processivity, strand displacement, and hairpin readthrough
```

ProteinMPNN is the sequence-proposal step. The Eco1 mask protects catalytic
motifs, Wang/Ec86 direct substrate-contact priors, Ec86 clade 9 conserved
positions, and mapped residues within 5 A of retained DNA/RNA before sampling.

ColabFold is the structural-fidelity gate. It compares full-length WT and
candidate sequences against the selected Ec86 cryoEM-backed RT scaffold. The
claim at this stage is fold preservation, not improved RT activity.

Biohub ESMC SAE is the query-time semantic annotation layer for WT and synthetic
candidates after fold checking. It can show whether model-derived feature
regions are retained or shifted. ESM Atlas remains public-protein neighborhood
context where a sequence or related public protein is present. WT ESMC
masked-marginal scoring is a separate model check over single
substitutions in the WT sequence context. None of these outputs measures strand
displacement, processivity, or structured-template readthrough.

### Runtime Ownership

Fold validation is split across three owners. Eco1 selects the sequences,
reference structure, and thresholds. BU SCC job templates own scheduler and
device execution details such as storage roots, queue resources, and the
LocalColabFold-backed `colabfold_batch` command.
`dnadesign.thread.adapters.colabfold` owns generic ColabFold output parsing, and
`dnadesign.thread.foldcheck` owns the normalized request/report fields so
ColabFold, AlphaFold-family, or later fold runtimes can write the same compact
artifact without importing Eco1 biology.

### Required Fields

- `candidate_id`
- `runtime_kind`
- `runtime_version`
- `input_sequence_hash`
- `reference_structure_id`
- `wt_baseline_artifact_id`
- `runtime_parameters_hash`
- `threshold_id`
- `threshold_values`
- `plddt`
- `pae_summary`
- `wt_runtime_ca_rmsd`
- `cryoem_mapped_ca_rmsd`
- `cryoem_mapped_ca_rmsd_status`
- `protected_contact_retention`
- `status`
- `rejection_reason`
- `missing_metric_reason`

### Metric Semantics

Fold validation compares each full-length candidate against the selected Eco1
RT structural authority. Wild type must be evaluated with the same runtime
settings before variant thresholds are interpreted.

| Metric | Meaning | Conservative use |
| --- | --- | --- |
| `plddt` | Global confidence summary from the fold runtime. | Reject substantial degradation from WT baseline or threshold-free acceptance. |
| `core_plddt` | Confidence over mapped RT core/palm/fingers/thumb regions. | Reject if the catalytic core is low confidence. |
| `pae_summary` | Domain-orientation uncertainty summary. | Reject high uncertainty between palm, fingers, and thumb. |
| `wt_runtime_ca_rmsd` | Candidate-to-WT ColabFold runtime C-alpha RMSD from the normalized fold-check report. | Flag candidates that diverge from the same-runtime WT baseline. |
| `cryoem_mapped_ca_rmsd` | Direct mapped-position C-alpha RMSD to the ec86kit/7V9U-backed reference when local model files are available. | Review whether a candidate still preserves the cryoEM-supported scaffold. |
| `cryoem_mapped_ca_rmsd_status` | Availability/status field for the direct cryoEM-reference comparison. | Prevent missing local structures or coordinate-basis failures from being read as measured RMSD. |
| `protected_region_rmsd` | RMSD over protected catalytic/contact regions. | Reject protected-region movement beyond the declared threshold. |
| `protected_contact_retention` | Whether modeled protected contacts remain geometrically plausible after superposition. | Reject loss of retained nucleic-acid/contact geometry. |

Thresholds belong in the fold-check report or profile fixture. This page names
required semantics; it does not bless universal numeric cutoffs. A fold-check
row cannot be accepted from raw metric values alone; it must point to the WT
baseline artifact, runtime parameters, and threshold policy used for the
decision.

### Fail-Fast Rules

- No fold-check run without a WT baseline sequence in the same request.
- No materialized fold-check report can pass validation unless the WT baseline
  row is `accepted`; candidate RMSD and degradation checks depend on that
  baseline.
- No accepted fold-check row without a candidate-table row.
- No fold-check row can carry an `input_sequence_hash` that disagrees with the
  current fold-check request manifest for the same candidate id.
- No accepted fold-check row without reference structure provenance.
- No accepted fold-check row without runtime kind, runtime version, runtime
  parameter hash, threshold id, and threshold values.
- No hidden fallback from missing real metrics to fixture metrics.
- No pooled-window handoff from candidates that lack fold-check coverage for
  the actual full sequence being proposed.
- A fixture fold-check row must be labeled `fixture` and cannot satisfy a
  materialized candidate handoff.
- Runtime failures are rows with `status: rejected` or `status: errored`, not
  silent omissions.
- Smoke subsets must leave candidates outside the subset as explicit
  `errored` rows, so a partial fold run cannot masquerade as complete
  fold-check coverage.
- Candidate handoff selection must require `accepted` fold-check rows for the
  selected candidate ids. Presence as an `errored` smoke-row is not acceptance.

### BU SCC Storage Posture

Heavy fold-model outputs should be created on BU SCC project storage, not in
the laptop checkout and not in git. The first SCC preflight is:

```bash
df -h /project/dunlop/esouth /projectnb/dunlop/esouth /scratch/$USER
```

Use `/project/dunlop/esouth` for the repo, environment, and compact run
outputs when live capacity allows. Use `/projectnb/dunlop/esouth` only after a
fresh capacity check. USR sync should move compact manifests and normalized
reports by default; raw ColabFold output directories should stay on SCC unless
a later handoff explicitly selects small structure files for transfer.

Future smoke runs should use `COLABFOLD_EXTRA_ARGS='--num-models 1'` when the
goal is runtime-path validation. Full candidate screens should declare model
count explicitly before submission.

### Biohub ESMC SAE And Atlas Context

Biohub ESMC SAE can be used after full fold-check coverage as a query-time
annotation layer for WT and fold-report rows accepted by the validator. It is not a
replacement for ColabFold/AlphaFold-family fold QA, and it is not evidence that
a candidate has improved function. The right study wording is:

> Biohub ESMC SAE results are model-derived feature activations and
> source-backed feature descriptions for the selected SAE dictionary. Atlas
> results, when available, are public-protein neighborhood context. Neither
> source measures processivity, strand displacement, or hairpin unwinding.

The first Eco1 use case is a semantic context check: ask whether
ProteinMPNN variants that passed the fold-report validator preserve or shift
polymerase-related model-feature activation patterns in controlled ways.
For structured RNA templates, the feature panel should be described as
processivity-related hypotheses:

| Feature context | How to use it |
| --- | --- |
| Thumb/palm nucleic-acid binding | Candidate context for duplex grip and C-terminal thumb integrity. |
| Motif B / primer grip | Candidate context for template-primer register during structured-template pausing. |
| N-terminal fingers/palm | Candidate context for template entry and possible hairpin destabilization. |
| DxD/YADD metal coordination | Catalytic competence context; protect from over-interpretation as processivity. |
| Pre-catalytic helix | Candidate context for open/closed active-site gating. |
| Broad RT/RdRp palm-core markers | Sanity checks that the sequence remains polymerase-like, not positive processivity scores. |

`RdRp` means RNA-dependent RNA polymerase. The label appears in some ESMC/SAE
feature names because RTs and RdRps share polymerase fold and motif geometry;
it does not mean Ec86 is an RdRp.

The Biohub ESMC SAE and Atlas context layers may be used for:

- QC: flag variants that no longer look RT-like in model-derived feature or
  public-neighborhood space.
- Annotation: expose thumb/palm, primer-grip, fingers/palm, catalytic, and
  gating feature shifts.
- Stratification: choose a balanced protein review panel with semantic-retained and
  semantic-shifted variants.
- Learning: provide model-derived input fields for supervised models after biochemical data
  exist.

It may not be used for:

- accepting candidates before full fold-check coverage exists;
- replacing cryoEM/ColabFold structural validation;
- claiming processivity, strand displacement, or hairpin readthrough;
- hiding a composite SAE score as an empirical fitness score.

Before biochemical data are inspected, freeze the feature panel, residue
windows, normalization rule, fold-gate thresholds, semantic flag definitions,
selection strata, assay endpoints, and primary analysis plan. The pre-assay
language is:

> SAE features will not be interpreted as direct measurements of processivity,
> strand displacement, or hairpin unwinding. They will be used to annotate
> fold-report-accepted Ec86 RT variants and to plan a stratified downstream
> review panel for later biochemical testing.

A first protein review panel should be a designed contrast, not a top-N SAE ranking. Use
fold-report-accepted variants to fill strata such as:

- WT Ec86 baseline.
- Fold-best / semantic-retained candidates.
- Fold-best / semantic-shifted candidates.
- Thumb-retained / fingers-shifted candidates.
- Primer-grip-shifted candidates with intact catalytic and fold gates.
- Random fold-report-accepted controls.

The [Biohub ESMC SAE feature interpretation notebook](https://colab.research.google.com/github/Biohub/esm/blob/main/cookbook/tutorials/esmc_sae_feature_interpretation.ipynb)
is the appropriate method reference for the SAE review step. In this study,
that means removing BOS/EOS positions before residue indexing, checking the
expected top-k sparsity, ranking features from the exact query-time SAE
dictionary by peak activation and prevalence, inspecting where they activate
over residues, and joining source-backed descriptions only for the same model,
layer, sparsity, and codebook. The current Eco1 all-97 SAE review path uses
the described `esmc-6b-2024-12-sae-layer60-k64-codebook16384` dictionary.
Candidate filtering should therefore stay staged:

```text
fold accepted
+ feasibility reviewed
+ optional exact-dictionary SAE stratum
-> review panel
```

The SAE stratum can help balance WT-like, shifted, depleted, and outlier
semantic profiles inside the review panel. It must not rescue a structurally
failed candidate or serve as a standalone acceptance rule.

Atlas and Biohub Platform surfaces should stay separate in documentation and
code. The ESM Atlas API is an alpha, currently no-auth lookup/search surface
for Atlas proteins, features, and similarity context. The authenticated Biohub
Platform `/api/v1/fold` endpoint is the ESMFold2 fold service. The
authenticated `/api/v1/logits` endpoint exposes ESMC logits/embeddings/SAE
outputs and is not a fold endpoint.

The first dnadesign implementation uses
`dnadesign.thread.adapters.esm_atlas` for reusable Atlas API normalization and a
thin Eco1 wrapper at
`src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/atlas_semantic_profile/`.
The wrapper selects WT plus fold-report rows accepted by the validator, calls the generic adapter, and
writes compact study-local artifacts:

- `atlas_semantic_profile.parquet`: one row per sequence, including query hash,
  Atlas hash, top feature indices/labels, raw response hash, status, and
  failure reason.
- `atlas_protein_activations.parquet`: sparse protein-level SAE activations.
- `atlas_residue_activations.parquet`: sparse per-residue SAE activations with
  zero-based Atlas residue indices and one-based sequence positions.
- `atlas_feature_catalog.parquet`: feature labels/descriptions once, not
  repeated per residue.
- `structure_predictions/structure_prediction_registry.parquet`: optional
  provenance rows for any Atlas/ESMFold-derived structures produced by an
  explicit on-demand fold request.

The current all-97 Atlas probe uses hash lookup with an explicit
`--allow-fold-on-miss --prediction-set-id ...` cap and resume by per-sequence
Atlas query hash. WT is accepted and has rich sparse Atlas rows plus one
Atlas/ESMFold-derived structure registry row. The first synthetic ProteinMPNN
candidates still return explicit 404 rows on this endpoint, and the remaining
synthetic candidates are capped as unattempted. Do not keep retrying this
hash-lookup path expecting rich query-level synthetic SAE rows unless the API
behavior changes. If no-auth Atlas context is needed for synthetic candidates,
use the sequence-similarity endpoint as a separate semantic-neighborhood
artifact. Do not merge Atlas/ESMFold structures with SCC ColabFold fold-check
structures, and do not route Atlas rows through fold validation or candidate
acceptance until a later policy explicitly says how that evidence is used.

Synthetic ProteinMPNN sequences should use Biohub ESMC/logits when rich
query-time SAE activations are needed. The implemented path is:

```text
same sequence
-> Biohub /api/v1/encode: amino-acid string to ESMC token ids
-> Biohub /api/v1/logits: ESMC SAE activations for those tokens
-> compact dnadesign Parquet tables keyed by candidate_id and sequence_hash
```

The current run uses `esmc-6b-2024-12` with
`esmc-6b-2024-12-sae-layer60-k64-codebook16384` and
`normalize_features=true`. It materialized WT plus all 96 fold-report
candidate rows accepted by the validator. All 97 selected sequences returned sparse query-time
SAE outputs with 64 active features per residue.
Store these rows as semantic annotation only.
Do not use them as a hidden processivity score or as a replacement for
ColabFold/AlphaFold-family structural checks.

Source roles:

- Tao et al. supplies the ProteinMPNN-to-fold-check design pattern, not an
  Atlas scoring rule.
- Wang et al. supplies the Ec86 cryoEM structural scaffold and direct
  substrate-contact context.
- ColabFold supplies the `colabfold_batch` fold-prediction CLI and output
  convention used for the current SCC run.
- LocalColabFold supplies the SCC-local install path for that CLI; it is not a
  second fold-validation method.
- Candido et al. supplies the ESMC, ESMFold2, Atlas, and SAE representation
  frame.
- Atlas API documentation supplies current endpoint behavior and alpha-status
  caveats.
- Biohub `/api/v1/encode` and `/api/v1/logits` documentation supplies the
  authenticated query-time ESMC/SAE API shape used for synthetic candidates.
- The Biohub ESMC SAE feature interpretation notebook supplies the feature
  ranking, residue activation-pattern, and feature-prevalence review pattern for
  exact-dictionary SAE interpretation.

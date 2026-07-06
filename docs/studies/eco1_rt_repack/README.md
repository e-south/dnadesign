---
doc_id: study-eco1-rt-repack
surface: study-root
study_id: eco1_rt_repack
owner: dnadesign-maintainers
last_verified: 2026-07-04
first_hop: routes/README.md
status_surface: record-only
preflight_surface: planned-contract-checks
---

## Eco1 RT Repack Study

This study is the checked-in planning and contract surface for Eco1
reverse-transcriptase fixed-backbone redesign.

Eco1 is the first anchor/profile. The reusable ontology is fixed-backbone
protein sequence design, not Eco1 and not reverse transcriptases in general.

Use `routes/README.md` first. Current state is in `record/status.md`; source and
artifact posture are in `record/datasets.yaml`; machine-readable planning
surfaces are under `operations/`; durable rationale and policy live in
`contexts/`; design-set meaning and provenance live in `workbench/`.

### Motivation And Funnel

Eco1/Ec86 has a cryoEM-supported RT/RNA/DNA structure that anchors this
fixed-backbone repacking study. The biological motivation is to prepare
fold-preserved reverse-transcriptase protein variants for downstream assays
that can test processivity, strand displacement, and structured-template
readthrough. The computational funnel does not claim those functions; it
selects a bounded protein sequence panel for review and handoff.

The method starts from the selected Ec86 cryoEM scaffold. The study defines
fixed and mutable positions using catalytic motifs, Wang/Ec86 direct-contact
priors, retained DNA/RNA proximity, and Tao-style homolog-conservation masks
from Mestre-derived MSA profiles. Different conservation and proximity
thresholds define design classes. ProteinMPNN samples protein sequences at the
unprotected positions for each design class. ColabFold predictions then remove
poor fold-model candidates. The remaining pool is triaged by MSA support,
localized mutation geography, near retained DNA/RNA or thumb-track chemistry, sequence
nonredundancy, and model-check annotations that are kept out of the acceptance
gate.

The current reviewer-facing endpoint is a six-row panel with one feasible,
fold-preserved representative per design class. The review notebook should let
a user inspect the full population, the selected six, py3Dmol structure views,
and the selected protein sequences. The flat
`candidate_handoff_sequences.csv` is the protein-sequence export for RT-only
handoff planning; codon optimization, restriction-site screening, and construct
subject creation remain downstream work.

This study intentionally separates three layers:

| Layer | Owner | Examples |
| --- | --- | --- |
| Study biology | `eco1_rt_repack` | Eco1 structure authority, catalytic masks, retron motif protection, first candidate-batch policy. |
| Reusable fixed-backbone mechanics | `thread` | Generic ProteinMPNN request, sample-ingest, candidate-table, ColabFold output normalization, and fold-check request/report contracts now; fold-model execution, feasibility, and handoff contracts are planned. |
| Downstream construct realization | `rt_lnrna_sponging_construct_triage` and `construct` | Pairing an accepted RT with lnRNA/TF-sponging construct subjects. |

The record must fail visibly when a layer is missing. It should not hide missing
structure authority, residue numbering, fold metrics, or downstream acceptance
behind fallback prose.

### Directory Ontology

```text
eco1_rt_repack/
  README.md
  record/
    campaign.yaml
    datasets.yaml
    status.md
  routes/
    README.md
  contexts/
    fixed-backbone-method.md
    msa-method.md
    implementation-roadmap.md
    residue-mask-policy.md
    fold-validation-policy.md
    synthesis-feasibility-policy.md
  operations/
    ops.study.yaml
    contract/
      fixtures/
      lifecycle/
      readiness/
      schemas/
      status/
      surfaces/
    runtime/
      command-groups/
  workbench/
    ontology/
    design_sets/
    provenance/
      conservation-source-discovery.md
      conservation-sources.yaml
      residue-numbering-policy.yaml
      structure-sources.yaml
```

Generated MPNN samples, fold predictions, large candidate tables, and runtime
sidecars do not belong in this checked-in record root.

### Naming Rules

- Use `eco1_rt_repack` for the checked-in study id.
- Use `eco1-rt-repack` only for human-facing slugs, skills, or dated plan
  filenames where kebab case is already the local convention.
- Use `eco1_rt_v1` for the first Eco1 profile id.
- Use neutral artifact names such as `mask_set.yaml` and
  `candidate_handoff.yaml` for reusable fixed-backbone contracts.
- Do not encode backend names into candidate ids; backend provenance belongs in
  candidate rows and upstream hashes.
- Do not create `permuter__var_id` or RT-lnRNA construct-subject ids until a
  later explicit handoff contract owns that promotion.

### Implementation Boundary

The current executable chain materializes structure, conservation, mask, thread
plan, ProteinMPNN request, sample table, candidate table, fold-check request,
fold-check reports, fold-check review bundles, local fold PDB staging, and
selected-panel ESM Atlas lookup. The baseline batch
`eco1_rt_p25_5a_n96_20260624` has 96 accepted ProteinMPNN samples and 96
accepted candidate rows. The expanded design-class bundle adds five
conservative or sensitivity classes and now contains 576 nonredundant synthetic
candidates plus WT for fold and SAE review. These are fold-preserved,
model-annotated candidates, not selected assay winners.

The review-deliverables bundle materializes additive WT-context ESMC LLR tables
and plots. That score compares candidate substitutions with the WT residue at
the same positions under a WT masked-marginal context; it is not a whole-protein
pseudo-likelihood and not an activity measurement. Whole-protein ESMC
pseudo-likelihood is not part of the v1 six-variant panel.

The protein review panel layer is now materialized for the expanded pool under
`outputs/thread/design_classes/selection/`. It contains a computational
full-gene feasibility report, a candidate triage table, and a six-row panel
with one feasible fold-preserved representative from each design class. The
triage table now records MSA support for designed residues, mutation geography,
near retained DNA/RNA or thumb-track chemistry, and sequence nonredundancy. ESMC
and SAE remain review evidence only. The review fields explain the panel; they
are not a single combined score.

Study code under `src/dnadesign/studies/units/eco1_rt_repack/` owns Eco1
policy and study paths. `dnadesign.thread.adapters.proteinmpnn` owns generic
ProteinMPNN request and sample-ingest mechanics, `dnadesign.thread.candidates`
owns generic candidate-table construction, and `dnadesign.thread.foldcheck`
owns generic fold-check request/report contracts. `dnadesign.thread.adapters.colabfold`
owns generic ColabFold output normalization, and
`dnadesign.thread.adapters.esm_atlas` owns generic Atlas lookup and sparse SAE
activation normalization. `dnadesign.thread.adapters.biohub_esmc` owns
authenticated Biohub ESMC `/api/v1/encode` -> `/api/v1/logits` query-time SAE
normalization for synthetic sequences that are not present in Atlas. These
Biohub ESMC rows are semantic annotation only; they are not fold validation,
processivity evidence, or candidate acceptance. Whole-protein ESMC
pseudo-likelihood is deferred and is not required for the v1 six-variant panel.
`dnadesign.thread.structure_predictions` owns the generic registry for
model-predicted structures, so an Atlas/ESMFold structure and a ColabFold
fold-check structure for the same sequence remain separate provenance records.
`dnadesign.thread.structure_views` owns the browser-embedded structure-view
contract used by the review notebook. Eco1 owns which structures appear in that
notebook, and ChimeraX remains the path for publication stills and pose
capture. `infer` may later own backend process execution if an explicit adapter
contract is added.

Fold-check execution currently means the ColabFold `colabfold_batch` CLI on BU
SCC, installed through LocalColabFold. LocalColabFold is an environment/install
path for the CLI, not a separate fold model or hosted API.

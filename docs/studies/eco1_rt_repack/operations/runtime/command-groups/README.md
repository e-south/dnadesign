## Eco1 RT Repack Command Groups

**Owner:** dnadesign-maintainers
**Last verified:** 2026-07-06

This directory is the reproducibility route for the study-owned Eco1 RT repack
materialization path. It is not a hidden run-all pipeline. Each lane in
`pipeline.yaml` names one owner, one artifact boundary, and one executable
command or planned future command.

Use `pipeline.yaml` as the machine-readable checklist for rerunning the current
study artifact chain:

```text
structure authority
structure preprocessing
contact evidence
contact geometry evidence
provider source cache
roster source cache
source FASTA bundles
source sufficiency
alignment bundles
MSA visualization sidecars
conservation profile
manual mask authority
mask set
contact-risk audit
phase validation
```

### Rerun Ladder

Use this order when regenerating the local Phase 1 state. Each step maps to a
lane id in `pipeline.yaml`; read that lane for the exact command and external
input placeholders.

1. `structure_authority` emits `backbone_bundle.yaml` and `residue_map.parquet`.
2. `structure_preprocessing` emits `structure_preprocessing_manifest.yaml` to
   make the raw 7V9U dimer to selected protomer-1 chain ontology explicit.
3. `contact_profile` emits retained DNA/RNA nearest-proximity evidence.
4. `contact_geometry_profile` emits atom-class side-chain/backbone/contact-density
   evidence from the selected protomer mmCIF.
5. `conservation_provider_sources` ingests the hash-pinned Mestre S1 roster and
   explicit provider source files or writes unresolved-provider ledgers.
6. `conservation_roster_cache` emits selected Ec86 clade 9 and II-A3/`42_1`
   source records with declared QC metadata.
7. `conservation_source_bundles` emits unaligned FASTA bundles and inserts the
   ec86kit target row.
8. `conservation_source_sufficiency` must pass before any MSA backend runs.
9. `conservation_alignments` runs the declared Clustal Omega backend through
   `dnadesign.aligner.msa`.
10. `evidence_profiles` emits `conservation_profile.parquet`.
11. `manual_mask_authority` emits the runtime manual mask-authority artifact
   from the checked-in ontology.
12. `mask_contract` emits the simple clade9-plurality-25/direct-contact-5 A
   `mask_set.yaml`.
13. `contact_risk_profile` emits a contact evidence review from the contact,
   conservation, manual-mask, and mask-set evidence chain.
14. `phase1_contract_validation` must pass before sampling-plan work starts.

Phase 1 validation is not a presence-only check. It validates
`structure_preprocessing_manifest.yaml` as the raw 7V9U-to-protomer authority and
re-checks `contact_geometry_profile.parquet` upstream hashes against the current
structure-source policy, preprocessing manifest, backbone bundle, residue map,
and ec86kit model.

The contact-geometry implementation is split by responsibility so this evidence
surface remains easy to audit before mask work: `structure_io.py` owns
mmCIF parsing and retained-chain extraction, `rows.py` owns atom-distance and
contact-density row construction, `writer.py` owns Parquet schema/metadata
emission, and `pipeline.py` owns orchestration only.

The Phase 1 state is materialized locally through
`contact_geometry_profile.parquet`, `contact_risk_profile.yaml`, and
`mask_set.yaml`. Evidence-review artifacts do not decide protected residues.
The current mask rule is
`eco1_rt_clade9_plurality25_direct_contact5a_v1`: protect NAxxH/YADD/VTG,
Wang/Ec86 direct substrate-contact priors, Ec86 clade 9 >=25% WT-plurality
conservation calls, and mapped residues within 5 A of retained DNA/RNA.
Terminal residues `1`, `2`, and `312-320` are `non_fixed_missing_backbone`.

### Backend and Device Boundary

The request plan is materialized:

```text
thread_plan.yaml
proteinmpnn_request/request_manifest.yaml
```

`thread_plan.yaml` declares backend, seed, temperature, request hash,
fixed/non-fixed position source, terminal missing-backbone exclusions, and
no-fallback policy. The Eco1 `proteinmpnn_request` command resolves study paths
and selected 7V9U/ec86kit structure provenance, then calls
`dnadesign.thread.adapters.proteinmpnn` to export the protein-only backbone,
convert canonical residues to chain-local ProteinMPNN positions, write
helper-compatible parsed-PDB, assigned-chain, and fixed-position JSONL payloads,
and build the request manifest. The Eco1 `proteinmpnn_sample_ingest` command
then calls the same generic adapter with an explicit ProteinMPNN checkout,
verifies official helper parity, runs `protein_mpnn_run.py` for declared seeds,
temperatures, `num_seq_per_target`, and omitted amino acids, and writes
`sample_table.parquet`. `candidate_table` then converts accepted backend rows
into canonical-position mutation summaries and rejects protected-position edits.
`foldcheck_request` reconstructs full 320-aa WT/candidate sequences and writes a
ColabFold CLI request manifest without running a fold model. The device boundary
is explicit: this study owns the request and threshold policy, `docs/bu-scc`
owns scheduler/runtime templates, and `thread` owns the normalized fold-check
report contract. The SCC execution lane is
`docs/bu-scc/jobs/eco1-colabfold-foldcheck.qsub`: submit a small
`FOLDCHECK_SEQUENCE_LIMIT=6` smoke first, then `all` only after the smoke
outputs can be normalized into `foldcheck_report.parquet`. A changed mask rule
must be opened as an explicit policy change before it can feed sampling.

The SCC runtime uses the ColabFold `colabfold_batch` command. LocalColabFold is
only the pixi-based install path that provides this command on BU SCC.

`foldcheck_review` ranks the baseline 96-candidate fold report and writes a selected
structure-panel manifest, full local structure-set manifest, ChimeraX scripts,
Atlas subset manifest, visual manifest, SVG review plots, and a scoped marimo
notebook. It does not launch ChimeraX by default and does not copy the full SCC
ColabFold output tree.
The local structure set is one normalized PDB per accepted fold row, suitable
for full ChimeraX viewing while preserving SCC source paths in the manifest. The
review plots include alt text and interpretation limits; they summarize model
metrics and SAE coverage for inspection, not candidate acceptance.

### Generation-Policy V2 Lane

The active v2 cleanup uses complete generation policies rather than the nested
distance-mask design classes. The request lane writes
`generation_policies_v2/generation_policy_manifest.yaml`, position and alphabet
manifests, and one `proteinmpnn_request/request_manifest.yaml` per policy:

```bash
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.materialization.generation_policies --repo-root . all
```

The SCC template runs one complete policy per array task and does not combine
mutations across policies:

```bash
qsub -t 1 \
  -v DNADESIGN_REPO=<dnadesign_repo>,PROTEINMPNN_ROOT=<dnadesign_repo>/.var/tools/proteinmpnn \
  docs/bu-scc/jobs/eco1-proteinmpnn-generation-policy.qsub
qsub -t 2-3 \
  -v DNADESIGN_REPO=<dnadesign_repo>,PROTEINMPNN_ROOT=<dnadesign_repo>/.var/tools/proteinmpnn \
  docs/bu-scc/jobs/eco1-proteinmpnn-generation-policy.qsub
```

Each task writes that policy's `sample_table.parquet` and
`candidate_table.parquet`. When the SCC run is complete, pull
`generation_policies_v2/` back to the local clone before preparing local
fold-check inputs:

```bash
rsync -avz \
  esouth@scc1.bu.edu:/project/dunlop/esouth/dnadesign/src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/generation_policies_v2/ \
  src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/generation_policies_v2/
```

ProteinMPNN generation ends at the per-policy sample and candidate tables. It
does not by itself run fold checks, selection, or review deliverable
regeneration.

After `generation_policies_v2/` has been pulled back to the local clone,
aggregate complete generated sequences by policy and write a local ColabFold
request:

```bash
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.materialization.generation_policies \
  --repo-root . \
  candidate-pool
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.materialization.generation_policies \
  --repo-root . \
  foldcheck-request
```

The candidate pool deduplicates exact protein sequences and records source
policy provenance. It does not combine mutations across policy outputs. The
fold-check request writes `generation_policies_v2/foldcheck_request/` with a
WT-plus-candidate FASTA and a request manifest for the external ColabFold CLI.

For a local smoke run, first materialize a bounded FASTA from that request:

```bash
uv run python -m dnadesign.thread.foldcheck.subset \
  --request-manifest src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/generation_policies_v2/foldcheck_request/foldcheck_request_manifest.yaml \
  --sequence-limit 6 \
  --sequence-start 1 \
  --input-fasta src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/generation_policies_v2/foldcheck_local_runs/smoke_6/input_sequences.fasta \
  --run-manifest src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/generation_policies_v2/foldcheck_local_runs/smoke_6/colabfold_run_manifest.yaml \
  --output-dir src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/generation_policies_v2/foldcheck_local_runs/smoke_6/colabfold_outputs \
  --schema-id eco1_rt.colabfold_local_run_manifest \
  --execution-status planned_local_colabfold_cli
```

Then run the local ColabFold binary against the staged subset:

```bash
colabfold_batch --num-models 1 \
  src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/generation_policies_v2/foldcheck_local_runs/smoke_6/input_sequences.fasta \
  src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/generation_policies_v2/foldcheck_local_runs/smoke_6/colabfold_outputs
```

The active study record is the v2 generation-policy selection root.

### Protein Review Panel Preparation

The next local summaries prepare the active v2 candidate pool for a six-variant
protein review panel. They explain computational feasibility, local structure,
charge sensitivity, and selection-readiness evidence; they do not predict strand
displacement.

The SAE window summary uses existing Biohub ESMC sparse tables and does not make
new Biohub requests:

```bash
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.materialization.sae_window_summary --repo-root .
```

Run v2 selection readiness explicitly against the v2 root:

```bash
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness \
  --repo-root . \
  --output-root src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/generation_policies_v2
```

`review_deliverables` builds the manuscript/review bundle from existing
artifacts. It writes `review_deliverable_manifest.yaml`, a Mestre-derived clade
9 scaffold/mask-evidence panel, fold-review links, WT ESMC model-check SVGs,
exact-dictionary Biohub ESMC SAE review plots when present, interactive
py3Dmol-backed structure-browser manifests, and a scoped marimo notebook
organized by progressive analysis sections. The notebook presents constraint
evidence for the design mask, active v2 fold and panel-selection evidence, and
ESMC/SAE checks when those manifests exist.
The active panel-selection tables are materialized under
`outputs/thread/generation_policies_v2/selection/`; the review notebook links
the selection-readiness SVGs from that manifest, including local RMSD threshold
audits, C-terminal primer-RNA recognition review axes, near-region charge
sensitivity, and region-wise MSA support. WT ESMC
masked-marginal scoring is shown with the constraint
evidence as a model check, not as a mask input.
Static plots and interactive structure views are selected through the same
section/visual controls. The scaffold/mask browser highlights one mask or motif
category at a time on the off-white ec86kit/7V9U reference using a single
high-contrast highlight color. The ColabFold browser reuses local ColabFold
PDBs, fits the selected query to the reference in memory over mapped C-alpha
atoms, and displays a compact metric strip for pLDDT, RMSD, sequence identity,
and mutation burden. Structure-control labels are section-specific, and
molecule-visibility toggles stay stable when switching between visual sections.
It does not rewrite or duplicate the raw ColabFold PDB files; ChimeraX remains
the still-render and pose-capture path. The command
does not launch ChimeraX unless an operator passes the explicit render flag. It
does not rerun ProteinMPNN, ColabFold, Biohub, Atlas, or candidate selection.
The review notebook does not compute selection; it reads the selection manifest
and pre-rendered SVGs.

Use the active v2 selection root when regenerating review deliverables:

```bash
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables \
  --repo-root . \
  --selection-root src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/generation_policies_v2/selection
```

`atlas_semantic_profile` queries ESM Atlas with `fold_on_miss=false` unless an
operator explicitly opts into on-demand folding. Use `--selection-manifest` for
declared panels, `--resume-existing` to reuse rows with matching per-sequence
Atlas query hashes, and `--max-new-requests` plus `--request-sleep-seconds` for
bounded batch progression. The current all-97 on-demand probe accepted WT,
returned explicit 404 rows for the first four synthetic ProteinMPNN candidates,
and left the remaining 92 rows unattempted. Any returned PDB is recorded as an
Atlas/ESMFold-derived structure prediction, separate from ColabFold fold-check
evidence. Synthetic-candidate Atlas context through the no-auth API should use a
separate sequence-similarity artifact rather than retrying the hash-lookup path.

`biohub_esmc_sae_profile` uses the authenticated Biohub `POST /api/v1/encode`
then `POST /api/v1/logits` path for query-time ESMC SAE activations. Keep the
key in the sibling `../key.md` file or another operator-supplied path; the
materializer records the key label and redacted authorization only. The baseline
all-97 profile and expanded WT-plus-576 design-class profile are materialized;
this command is the safe resumable form:

```bash
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.materialization.biohub_esmc_sae_profile \
  --repo-root . \
  --sequence-limit all \
  --key-file ../key.md \
  --model esmc-6b-2024-12 \
  --sae-model esmc-6b-2024-12-sae-layer60-k64-codebook16384 \
  --normalize-features \
  --fetch-feature-descriptions \
  --resume-existing \
  --max-new-requests 5 \
  --request-sleep-seconds 1.5 \
  --request-timeout-seconds 180
```

Use `--sequence-limit all --resume-existing` only after deciding to spend the
remaining hosted requests. These rows are model annotation, not fold
validation or processivity evidence.

`biohub_esmc_wt_mutation_scoring` is a WT-only masked-marginal model check.
A final uncapped run must use all 320 WT positions; short position
ranges are smoke tests only and must be capped with `--max-new-requests`. The
default 300M run writes to `biohub_esmc/mutation_scoring/`. Non-default models
write under a model-specific subdirectory, so a 6B rescore does not overwrite
the 300M grid.

```bash
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.materialization.biohub_esmc_wt_mutation_scoring \
  --repo-root . \
  --positions all \
  --key-file ../key.md \
  --model esmc-300m-2024-12 \
  --resume-existing \
  --request-sleep-seconds 1.5 \
  --request-timeout-seconds 180
```

To rescore the same WT masked contexts with the 6B model:

```bash
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.materialization.biohub_esmc_wt_mutation_scoring \
  --repo-root . \
  --positions all \
  --key-file ../key.md \
  --model esmc-6b-2024-12 \
  --request-sleep-seconds 1.5 \
  --request-timeout-seconds 180
```

That command writes to
`biohub_esmc/mutation_scoring/esmc_6b_2024_12/`. The review-deliverables
materializer derives a separate 6B additive candidate-LLR table and plot when
that directory exists, then renders a 300M-versus-6B score-comparison plot. These
scores remain WT-context masked-marginal additive LLR values, not whole-protein
likelihoods.

Do not expect a capped position smoke test to resume into `--positions all`.
The request hash includes the selected position set. Use `--resume-existing`
only when rerunning the same model and same position set, such as a completed
all-position 6B grid.

The review-deliverables command now also renders a lightweight SAE
interpretation section from the existing sparse Biohub ESMC tables. That pass
does not call Biohub again. It follows the
[Biohub ESMC SAE feature interpretation notebook](https://colab.research.google.com/github/Biohub/esm/blob/main/cookbook/tutorials/esmc_sae_feature_interpretation.ipynb)
at a small scale: rank WT-active features, inspect residue-localized
activations, compare candidate retention of the same exact-dictionary
features, and render a joint review panel that arranges rows by SAE
similarity to WT for inspection while showing ColabFold pLDDT and summed WT
masked-marginal single-substitution LLR side markers. Feature labels and descriptions are
joined only from the exact 6B layer-60 16k dictionary; the LLR side marker is
not a joint protein likelihood.

### Source-Role Guardrails

- Tao is the masking-method prior: homolog MSA conservation, fixed functional
  residues, fixed-backbone RT redesign, and fold-check triage.
- ProteinMPNN is the backend request-format and execution prior:
  helper-compatible parsed-PDB JSONL, assigned-chain JSONL, fixed-position JSONL,
  chain-local 1-indexed positions, explicit seed, temperature,
  `num_seq_per_target`, omitted-amino-acid, and no-fallback execution fields.
- Mestre is the source ontology: the full S1 roster is a candidate/context
  surface, while Ec86 RT clade 9 and II-A3/`42_1` are the active conservation
  denominators.
- Simon is the annotation grammar for RT regions and motif visualization.
- Wang is the Eco1/Ec86 structural prior for the selected cryo-EM context,
  active-site/motif spans, and candidate interface residues.
- Inouye et al. 1999 and Inouye et al. 2004 are Ec86 C-terminal/thumb
  specificity priors. They support tracking thumb and C-terminal primer-RNA
  recognition context in review plots, but they are not active mask sources in
  the current policy.
- Paired-protomer dimerization is not a retention objective for the current
  monomeric RT-msDNA-msrRNA design profile; alpha-1/pre-RT1 residues are not
  fixed solely for dimer preservation.
- `manual-mask-authority.yaml` is the source for NAxxH, YADD, VTG, RT1-RT7
  review labels, and Wang/Ec86 direct contact priors. Under the selected mask
  policy, NAxxH/YADD/VTG and Wang direct contacts are protected; RT1-RT7 labels
  do not blanket hard-fix residues.

Keep these lanes independently runnable when executable commands are
introduced. A future orchestration command may call them in order only after
each lane has its own validator and negative-path fixture. Do not collapse them
into a single hidden pipeline.

External data inputs remain explicit. The provider and roster cache lanes need
a hash-pinned Mestre S1 table plus explicit provider FASTA source roots. The
checked-in study record should never infer source rows from review figures,
public Eco1 accessions that disagree with the ec86kit target hash, or transient
local FASTA files.

## Eco1 RT Repack Command Groups

**Owner:** dnadesign-maintainers
**Last verified:** 2026-06-25

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

`foldcheck_review` ranks the full 96-candidate fold report and writes a selected
structure-panel manifest, full local structure-set manifest, ChimeraX scripts,
Atlas subset manifest, visual manifest, SVG review plots, and a scoped marimo
notebook. It does not launch ChimeraX by default and does not copy the full SCC
ColabFold output tree.
The local structure set is one normalized PDB per accepted fold row, suitable
for full ChimeraX viewing while preserving SCC source paths in the manifest. The
review plots include alt text and interpretation limits; they summarize model
metrics and SAE coverage for inspection, not candidate acceptance.

### Design-Class Expansion

The 5 A class remains the baseline. The design-class materializer adds request
surfaces for five additional classes without overwriting the baseline artifacts:
clade 9 p25 contact 6/8/10 A, clade 9 p50 contact 5 A, and II-A3/`42_1` p50
contact 5 A.

```bash
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.materialization.design_classes --repo-root . requests
```

Each generated class gets its own `mask_set.yaml`, `thread_plan.yaml`, and
ProteinMPNN request sidecars under `outputs/thread/design_classes/<class-id>/`.
Run ProteinMPNN for those classes on BU SCC through the submit-ready job
template in `docs/bu-scc/jobs/eco1-proteinmpnn-design-class.qsub`. Submit one
class first, then the remaining array after the smoke writes its per-class
`candidate_table.parquet`:

```bash
qsub -t 1 \
  -v DNADESIGN_REPO=<dnadesign_repo>,PROTEINMPNN_ROOT=<dnadesign_repo>/.var/tools/proteinmpnn \
  docs/bu-scc/jobs/eco1-proteinmpnn-design-class.qsub
qsub -t 2-5 \
  -v DNADESIGN_REPO=<dnadesign_repo>,PROTEINMPNN_ROOT=<dnadesign_repo>/.var/tools/proteinmpnn \
  docs/bu-scc/jobs/eco1-proteinmpnn-design-class.qsub
```

After those runs are ingested into per-class `candidate_table.parquet` files,
rebuild the nonredundant pool:

```bash
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.materialization.design_classes --repo-root . candidate-pool
```

The pool keeps one row per `sequence_hash` and records duplicate class
provenance. Only after the pool includes at least one generated class should the
expanded fold-check request be written:

```bash
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.materialization.design_classes --repo-root . foldcheck-request
```

The expanded fold-check, ColabFold normalization, Biohub ESMC SAE profile, and
ESMC additive LLR review should then use the `design_classes/` output root so
the new variants carry the same feature families as the current 96-candidate
baseline. After the expanded ColabFold report is normalized, stage the shared
non-mask inputs that downstream fold-review and ESMC lanes need:

```bash
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.materialization.design_classes --repo-root . downstream-inputs
```

This writes `design_classes/candidate_table.parquet` from the nonredundant
candidate pool and copies shared residue-map, backbone, MSA, source-manifest,
and WT ESMC mutation-scoring inputs. It does not write a root-level
`mask_set.yaml`: the expanded pool contains multiple `mask_policy_id` values,
so mask-specific review must read the per-class mask sets instead of treating
one mask as the whole pool.

After `foldcheck_review` is materialized against the `design_classes/` root,
derive expanded 300M and 6B additive ESMC candidate-preference outputs without
new Biohub requests:

```bash
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.materialization.design_classes --repo-root . esmc-sequence-preference
```

These outputs are WT-context masked-marginal additive LLR review covariates.
They are not whole-protein likelihoods and are not assay measurements.

`review_deliverables` builds the first broader manuscript/review bundle from
existing artifacts. It writes `review_deliverable_manifest.yaml`, a
Mestre-derived clade 9 scaffold/mask-evidence panel, ProteinMPNN diversity
SVGs, a Tao-style ColabFold RMSD/pLDDT joint plot for the current single mask
policy, a ChimeraX mask-context script, WT ESMC model-constraint audit
SVGs, exact-dictionary Biohub ESMC SAE review plots, interactive
py3Dmol-backed structure-browser manifests, and a scoped marimo notebook
organized by progressive analysis sections. The notebook presents constraint
evidence for the design mask, ProteinMPNN designs with fold triage, ESMC
feature review, and a planned feasibility/handoff gate. WT ESMC
masked-marginal scoring is shown with the constraint evidence as a review-only
model-constraint audit, not as a mask input.
Static plots and interactive structure views are selected through the same
section/visual controls. The scaffold/mask browser highlights one mask or motif
category at a time on the off-white ec86kit/7V9U reference using a single
high-contrast highlight color. The ColabFold browser reuses local ColabFold
PDBs, fits the selected query to the reference in memory over mapped C-alpha
atoms, and displays a compact metric strip for pLDDT, RMSD, sequence identity,
and mutation burden. It does not rewrite or duplicate the raw ColabFold PDB
files; ChimeraX remains the still-render and pose-capture path. The command
does not launch ChimeraX unless an operator passes the explicit render flag. It
does not rerun ProteinMPNN, ColabFold, Biohub, Atlas, or candidate selection.
WT SAE structure frames, feature-window heatmaps, SAE-to-structure overlays,
and feasibility/selection matrices are planned follow-ons, not part of the
materialized foundation bundle.

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
materializer records the key label and redacted authorization only. The current
all-97 profile is materialized; this command is the safe resumable form:

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
remaining hosted requests. These rows are semantic annotation, not fold
validation or processivity evidence.

`biohub_esmc_wt_mutation_scoring` is a WT-only masked-marginal model-constraint
audit. A final uncapped run must use all 320 WT positions; short position
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
that directory exists, then renders a 300M-versus-6B rank-stability plot. These
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
features, and render a joint review panel that orders variants by SAE
similarity to WT while showing ColabFold pLDDT and summed WT masked-marginal
single-substitution LLR side markers. Feature labels and descriptions are
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

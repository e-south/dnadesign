---
doc_id: study-eco1-rt-repack-implementation-roadmap
surface: study-context
study_id: eco1_rt_repack
owner: dnadesign-maintainers
last_verified: 2026-06-26
---

## Implementation Roadmap

This roadmap is the implementation-facing checklist for the Eco1 RT repack
study and the reusable `thread` tool. It is intentionally staged so
each slice can be changed, tested, or replaced without rewriting the full
campaign.

### Owner Split

| Surface | Owns | First implementation obligation |
| --- | --- | --- |
| `eco1_rt_repack` | Eco1 profile authority, manual protected-residue policy, study selection decisions, and downstream promotion policy. | Validate that pending Eco1 structure and numbering decisions block runtime artifacts. |
| `thread` | Generic fixed-backbone artifact contracts, residue maps, mask algebra, backend request/result normalization, candidate ids, fold-check interpretation, and handoff hash closure. | Materialize and validate the artifact chain against fixtures and negative cases. |
| `infer` | Optional model-process execution and writeback for declared backends. | Expose backend run ids and result manifests if it executes MPNN or fold-check jobs. |
| `construct` | Later sequence realization and placement/window feasibility. | Accept only explicit downstream promotion records, not raw `thread` candidates. |
| `permuter` | DMS and explicit variant records through public APIs. | Import redesigned RT candidates only through a later public handoff contract. |

### Slices

| Slice | Code home | Contract input | Contract output | Negative path |
| --- | --- | --- | --- | --- |
| `profile-validator` | study unit, then `thread.contracts` | `eco1_rt_v1.profile.yaml` | validated profile object | pending structure authority is accepted past Phase 0 |
| `artifact-chain-validator` | `thread.contracts` | artifact-chain schema plus fixture paths | ordered artifact contract | required fields disagree across schema surfaces |
| `structure-authority-materializer` | `thread.structure` | selected PDB/mmCIF, chain policy, reference FASTA | `backbone_bundle.yaml`, `residue_map.parquet` | chain, retained context, numbering origin, or sequence hash is missing |
| `evidence-profile-ingest` | `thread.evidence` | residue map, contact source, MSA source declarations | `contact_profile.parquet`, `conservation_profile.parquet` | evidence source hash or per-position mapping is missing |
| `mask-builder` | `thread.masks` | evidence profiles and manual Eco1 policy | `mask_set.yaml` | missing evidence becomes designable or mask conflict is silently resolved |
| `sampling-request-builder` | `thread.adapters` | mask set and backend policy | `thread_plan.yaml`, backend request manifest | backend, seed, temperature, fixed positions, or fallback policy is implicit |
| `sample-ingest` | `thread.adapters` or `infer` handoff | backend result manifest | `sample_table.parquet` | samples lack run id, request hash, score, sequence hash, seed, or status |
| `candidate-builder` | `thread.candidates` | sample table and mask set | `candidate_table.parquet` | duplicate ids, non-deterministic ordering, or mask violations are accepted |
| `foldcheck-normalizer` | `thread.adapters` or `infer` handoff | candidate table and fold runtime output | `foldcheck_report.parquet` | WT baseline, thresholds, runtime parameters, or errored rows are missing |
| `atlas-semantic-profiler` | `thread.adapters` plus study wrapper | accepted fold-check rows and candidate sequences | `atlas_semantic_profile.parquet` | Atlas/SAE affiliations are treated as function proof, query hashes are missing, or API schema drift is hidden |
| `feasibility-assessor` | `thread.candidates` plus study policy | accepted full-sequence candidates | `feasibility_report.parquet` | windowed candidates lack nearest-parent or structural-coupling evidence |
| `candidate-handoff-builder` | `thread.handoffs` plus study selection policy | candidate, fold, and feasibility reports | `candidate_handoff.yaml` | upstream hashes, nonfixture fold acceptance, or downstream target are missing |
| `rt-lnrna-promotion-check` | downstream study | RT-only candidate handoff | downstream accept/reject record | construct-subject ids are preclaimed before downstream binding |

### Implemented Slice: Phase 0 Contract Validator

`src/dnadesign/studies/units/eco1_rt_repack/operations/contract_validation.py`
is the first executable study-owned CLI entrypoint. Domain validation now lives
under
`src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/`, with
separate modules for profile contracts, artifact-chain contracts, structure
authority, conservation-source authority, and materialized runtime artifacts.
Keep new checks in those semantic modules instead of growing the CLI entrypoint.

Supported commands:

```bash
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.contract_validation --repo-root . --phase phase0_scaffold
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.contract_validation --repo-root . --phase phase1_thread_contract
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.contract_validation --repo-root . --phase phase2_real_backend_ingest
```

The validator owns only study-local scaffold and runtime-artifact checks. It
does not hide MPNN or fold execution behind a run-all command. The current
`dnadesign.thread` surface covers generic ProteinMPNN request, sample-ingest,
candidate-table, fold-check report, and ColabFold output-normalization
mechanics. Fold-model execution, feasibility, and handoff mechanics remain
separate contract promotions.

### Implemented Slice: Structure Authority v1a

The selected authority is `ec86kit_7v9u_protomer1`: PDB `7v9u`, protomer 1,
RT chain `A`, retained nucleic-acid context chains `D`, `E`, and `F`, and no
effector-retention policy for the first repacking profile. The selected
numbering policy is `ec86kit_protomer1_chain_a_numbering_v1`, backed by the
sibling `ec86kit` chain-A amino-acid map. That map covers 309 of 320 canonical
reference positions.

### Implemented Slice: Structure Authority Materializer v1

`src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/structure/`
materializes the selected authority into local runtime artifacts:

```text
src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/backbone_bundle.yaml
src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/residue_map.parquet
```

The backbone bundle records typed chain inventory: RT chain `A` as the design
backbone, DNA chain `D` as retained context, and RNA chains `E` and `F` as
retained context. The residue map records all 320 canonical Eco1 RT positions:
309 mapped positions and 11 terminal positions without coordinates. The current
mask treats those terminals as unprotected but not directly fixed-backbone
mutable until coordinates exist.

At that point in the slice sequence, Phase 1 failed on the next
conservation/contact/mask blockers until contact profile materialization was
run:

```text
eco1_rt.evidence.conservation_profile_not_materialized
eco1_rt.mask.mask_set_not_materialized
```

### Implemented Slice: Contact Profile Materializer v1

`src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/contact/`
materializes the retained-context contact evidence into a local runtime
artifact:

```text
src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/contact_profile.parquet
```

The profile is joined through `residue_map.parquet` and uses the pinned
`ec86kit` per-residue DNA/RNA minimum-distance table. It is retained as
distance evidence, not as the active mask policy. The active mask uses the
5 A direct-contact rule described below; broad 15-20 A shells remain diagnostic
history.

After contact materialization, Phase 1 failed on the next conservation/mask
blockers until the MSA-derived conservation profile was materialized:

```text
eco1_rt.evidence.conservation_profile_not_materialized
eco1_rt.mask.mask_set_not_materialized
```

### Source Discovery: Conservation Profile

`docs/studies/eco1_rt_repack/workbench/provenance/conservation-source-discovery.md`
records the method and roster priors for the MSA-derived conservation profile.
The discovery pass found enough prior art to define source authority without
inventing a bespoke source model: Tao et al. 2026 supplies the conservation
rule shape, and Mestre et al. 2020 supplies a 1928-entry retron RT roster with
Eco1/Ec86 in clade 9, subtype II-A3, cluster `42_1`.

This is not a materialized MSA and does not satisfy
`conservation_profile.parquet`.

### Implemented Slice: Conservation Source Contract v1

`docs/studies/eco1_rt_repack/workbench/provenance/conservation-sources.yaml`
declares the selected MSA source authority for `ec86_clade9_conservation_v1` and
`ec86_iia3_cluster42_1_conservation_v1`. It uses Mestre et al. 2020 Supplementary Table S1 as
the accession roster and candidate pool, declares NCBI and BV-BRC provider
policies, pins the target row to the ec86kit Eco1 RT reference sequence hash,
and records the T301/A301 mismatch in public `WP_099010551.1` as a fail-fast
target-row issue. The full Mestre roster is context/candidate-pool evidence,
not the Phase 1 conservation denominator.

`docs/studies/eco1_rt_repack/contexts/msa-method.md` explains the future
reproduction procedure for roster filtering, sequence retrieval, alignment,
residue-map joining, and Tao-style plurality/frequency scoring.

The Phase 1 validator checks source hashes, target-sequence identity, sequence
providers, filters, alignment command, gap denominator, threshold, and
plurality rule. It still fails until the generated
`conservation_profile.parquet` exists.

### Implemented Slice: Conservation Profile Materializer v1

`src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/conservation/`
materializes a long-form local runtime artifact when supplied explicit aligned
FASTA inputs for each selected profile id:

```text
src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/conservation_profile.parquet
```

The materializer reads `conservation-sources.yaml`, joins through
`residue_map.parquet`, requires the `eco1_rt_ec86kit_reference` target row,
rejects target-row drift such as the T301/A301 public-accession mismatch, and
scores each canonical Eco1 position with the Tao-style WT plurality/frequency
rule. It writes one row per `(profile_id, canonical_position)` and records
aligned-FASTA hashes in parquet metadata.

This slice deliberately does not fetch provider sequences or run MSA backends. Provider
source acquisition and source-sequence sufficiency are study-owned upstream
steps; aligned FASTA materialization remains the next blocker.

### Implemented Cross-Tool Slice: Aligner MSA Backend v1

`src/dnadesign/aligner/msa/` provides the generic alignment API for the next
source-data slice: FASTA validation, MAFFT/Clustal Omega preflight/execution,
and aligned FASTA bundle manifests. It is intentionally not Eco1-aware. It does not fetch
Mestre/NCBI/BV-BRC source sequences, adjudicate the T301/A301 target mismatch,
score conservation, or build masks.

After the selected source FASTA bundles pass sufficiency, the next Eco1
alignment slice should use this public API to produce:

```text
ec86_clade9_conservation_v1.aligned.fasta
ec86_iia3_cluster42_1_conservation_v1.aligned.fasta
```

with bundle manifests, then pass those aligned FASTA files into the study-owned
conservation-profile materializer.

### Implemented Slice: Conservation Provider Source v1

`src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/source_sequences/provider_sources/`
materializes explicit provider FASTA source files from the hash-pinned Mestre
S1 roster and declared provider endpoints. It writes NCBI Protein and BV-BRC
protein FASTA sources plus a provider-source manifest. Provider-missing rows
are not silently dropped; they are recorded in an explicit unresolved-provider
ledger before roster-cache can mark them excluded. Provider accession shapes
are compiled from `conservation-sources.yaml` so roster-cache and sufficiency
cannot drift apart.

The current local real-data run produced 350/350 NCBI Protein records,
1464/1577 BV-BRC protein records, and 113 explicit BV-BRC unresolved-provider
entries.

### Implemented Slice: Conservation Roster Cache v1

`src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/source_sequences/roster_cache/`
materializes local source records from selected roster rows and explicit
provider FASTA source files. Under the revised broad profile, it rejects
direct full-Mestre roster materialization for `ec86_clade9_conservation_v1`;
the local real-data run now emits clade 9 source records after declared QC.
It still rejects uncontracted roster hashes by default,
unsupported accession providers, missing included provider sequences without a
passed unresolved-provider ledger, and public `WP_099010551.1` target-row
leakage by excluding that accession with the declared T301/A301 mismatch
reason.

This slice deliberately does not fetch live NCBI/BV-BRC records, run MSA backends,
score conservation, or build masks.

### Implemented Slice: Conservation Source-Sequence Bundle v1

`src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/source_sequences/`
materializes unaligned source FASTA bundles from explicit local provider caches
and a `source_records.yaml` ledger. It validates declared provider ids, rejects
operator-supplied target rows, requires explicit exclusion reasons, inserts the
ec86kit target row itself, and writes profile-level and bundle-level manifests.

This slice deliberately does not fetch live provider records, run MSA backends, score
conservation, or build masks.

### Implemented Slice: Conservation Source-Sequence Sufficiency v1

`src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/source_sequences/sufficiency/`
is the pre-alignment gate for source FASTA bundles. It validates that provider
caches exist, `source_records.yaml` and provider FASTA hashes match the bundle
manifests, target rows are inserted and hash-pinned, profile bundles meet the
declared `min_non_gap_count` support floor, provider accessions look real for
their declared source, and exclusions remain explicit.

The sufficiency gate is split by ontology: context/report models, manifest
checks, cache/hash checks, provider accession validation, and FASTA content
checks. Shared conservation-source contract parsing and provider accession
policy live under
`operations/materialization/source_sequences/contracts/` so roster-cache,
source-bundle, and sufficiency code do not duplicate the source-authority
schema.

The previous local full-Mestre broad source bundle had 1814 included and 114
excluded rows, while `ec86_iia3_cluster42_1_conservation_v1` had 46 included
and 1 excluded row. That broad bundle is now superseded candidate-pool
context. The current selected source bundles pass sufficiency locally after
clade 9 / II-A3 QC. Alignment and conservation-profile materialization have
since completed for the selected profiles; the current runtime gate is the
post-mask sampling-plan policy, not source reselection.

### Implemented Slice: Conservation Clade 9 Source Cache v1

`conservation-clade9-source-cache-v1` consumes the full Mestre
candidate-pool provider cache and emit `ec86_clade9_conservation_v1` source
records from Mestre RT clade 9 after declared QC. The source-record QC computes
pairwise target coverage, pairwise identity, length status, motif-QC markers,
hard-reject status, and explicit exclusion metadata before any MSA backend run.
The local real-data run produced 302 included and 22 excluded
`ec86_clade9_conservation_v1` rows, plus 44 included and 3 excluded
`ec86_iia3_cluster42_1_conservation_v1` rows. A runtime cap or diversity subset
may be introduced only by
an explicit benchmark/contract update; it must not be the default explanation
for the broad panel.

After this slice, the regenerated source FASTA sufficiency gate passes.

### Implemented Slice: Conservation Alignment and Profile Acceptance v1

`src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/conservation_alignments/`
now accepts both selected source FASTA bundles through the public
`dnadesign.aligner.msa` Clustal Omega backend. The accepted local alignment bundle
contains `ec86_clade9_conservation_v1.aligned.fasta` with 303 records and
`ec86_iia3_cluster42_1_conservation_v1.aligned.fasta` with 45 records; both
include the pinned `eco1_rt_ec86kit_reference` target row and hash-linked
manifests.

`src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/conservation/`
then materializes `conservation_profile.parquet` from those accepted alignments.
The local profile has 640 rows, one per selected profile id and canonical Eco1
position. Generic `aligner.msa` visualization sidecars have also been
regenerated for both selected profiles.

### Implemented Slice: Manual Motif And RT Interval Authority v1

`docs/studies/eco1_rt_repack/workbench/ontology/manual-mask-authority.yaml`
is the study-owned source ontology for mask-authoritative motif anchors. It is
distinct from `rt-annotation-tracks.yaml`, which remains visualization-only.
`src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/manual_mask_authority/`
materializes local `manual_mask_authority.yaml` from the ontology and
`residue_map.parquet`. NAxxH, YADD, and VTG are protected motif anchors.
RT1-RT7 spans remain annotation/review labels under the current mask and do not
blanket-protect their intervals. The same ontology records Wang/Ec86
RT-msDNA/msrRNA interface candidates as direct substrate-contact priors.

### Implemented Slice: Mask Set Materializer v1a

`src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/mask_set/`
composes `residue_map.parquet`, `contact_profile.parquet`,
`conservation_profile.parquet`, and `manual_mask_authority.yaml` into
`mask_set.yaml`. The old 20 A version is all-fixed: 320 fixed positions and
zero directly mutable positions. That result showed that broad retained
nucleic-acid proximity is too blunt for Eco1. The next mask should use the
plurality25 direct-contact rule instead.

### Implemented Slice: Structure Preprocessing and Contact Geometry v1

`src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/structure_preprocessing/`
materializes `structure_preprocessing_manifest.yaml` from the checked-in
`structure-preprocessing.yaml` provenance policy. It records raw RCSB 7V9U
dimer context, selected sibling `ec86kit` protomer-1 model provenance, retained
RT/DNA/RNA chains, excluded paired-protomer context, the explicit non-objective
of preserving paired-protomer dimerization, and upstream hashes.

`src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/contact_geometry/`
materializes `contact_geometry_profile.parquet` by parsing the selected
ec86kit mmCIF model with the study-owned Biopython backend. The profile keeps
one row per canonical Eco1 position and measures all-atom, side-chain,
backbone, DNA/RNA split, contact-density, and retained-chain-count evidence.
Raw structure residue ids remain joined through `residue_map.parquet`; they do
not become design positions.

### Implemented Slice: Contact Risk Profile v1

`src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/contact_risk/`
materializes `contact_risk_profile.yaml` as a contact evidence review after the
conservative mask set. It joins the retained-context nearest-distance profile,
atom-class contact geometry, conservation masks, manual mask authority,
and Wang/Ec86 candidate priors.

`src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/contact_risk/`
validates the artifact shape and prevents missing evidence-status records from
being treated as permission to mutate a residue. This slice improves evidence
legibility; it does not choose the mask.

After this slice, Phase 1 passes locally. The old diagnostic 20 A mask has
been replaced by the simpler plurality25 direct-contact rule.

### Completed Slice: Plurality25 Direct-Contact Mask v1

The current rule is `eco1_rt_clade9_plurality25_direct_contact5a_v1`:

```text
protected =
  NAxxH / YADD / VTG
  OR Wang/Ec86 direct substrate-contact prior
  OR Eco1 amino acid is evolutionarily conserved at >=25% WT plurality in the Ec86 clade 9 MSA
  OR mapped residue is within 5 A of retained DNA/RNA

non_fixed = NOT protected
```

Terminal residues `1`, `2`, and `312-320` are
`non_fixed_missing_backbone`: unprotected, but not directly mutable by
fixed-backbone ProteinMPNN until coordinates exist.

This slice:

1. Regenerates `mask_set.yaml` under
   `eco1_rt_clade9_plurality25_direct_contact5a_v1`.
2. Emits row-level classes `protected`, `non_fixed`, and
   `non_fixed_missing_backbone`.
3. Preserves the current evidence roles: Tao for fixed-backbone redesign and
   homolog-MSA plurality, Mestre for source ontology, Wang for direct Ec86
   substrate-contact priors, and Simon for motif grammar.
4. Excludes contact-density classes, retained-chain-count classes,
   contact-class tiers, and RT1-RT7 blanket hard fixing from the mask call.
5. Keeps evidence-review artifacts out of the sampling plan unless a future
   task explicitly changes the mask policy.

### Completed Slice: ProteinMPNN Request Adapter v1

`thread_plan.yaml` is materialized from the simple mask. The study
`proteinmpnn_request/` materializer now delegates generic fixed-backbone
request mechanics to `dnadesign.thread.adapters.proteinmpnn`: protein-only
backbone export, chain-local position conversion, helper JSONL payloads,
request manifest construction, and generic request validation. Eco1 keeps the
study path, structure-source, mask, and provenance decisions.

The request uses the public ProteinMPNN command-line contract, not a private
parser: parsed-PDB JSONL, assigned-chain JSONL, fixed-position JSONL, explicit
designed chain, sampling temperature, seed, `num_seq_per_target`, and
omitted-amino-acid fields. ProteinMPNN fixed positions are chain-local
1-indexed sequence positions, so canonical Eco1 residue numbers are converted
before backend execution.

### Completed Slice: Backend Sample Ingest v1

`sample_table.parquet` is materialized from the validated ProteinMPNN request.
The active batch `eco1_rt_p25_5a_n96_20260624` uses an explicit local
ProteinMPNN checkout, runs official helper parity checks before sampling,
preserves request hash, seed, temperature, fixed-position source, and
no-fallback policy, and records a separate backend-run manifest.
The actual model call is `protein_mpnn_run.py` from official ProteinMPNN commit
`8907e6671bfbfc92303b5f79c4b5e6ce47cdef57`; `dnadesign` only prepares inputs,
checks parity, runs the declared command, and normalizes the resulting sequences
into sample rows.

### Completed Slice: Candidate Table v1

Build `candidate_table.parquet` from accepted `sample_table.parquet` rows. It
assigns stable candidate ids, keeps request/sample provenance, summarizes
mutations against the selected backbone sequence, rejects edits at protected
positions, and keeps terminal missing-backbone residues out of fixed-backbone
mutation accounting. The active table contains 96 accepted candidate rows.

### Completed Slice: Fold Check Request v1

`foldcheck_request/input_sequences.fasta` and
`foldcheck_request/foldcheck_request_manifest.yaml` are materialized from the
accepted candidate table. The request contains one WT baseline plus 96 accepted
ProteinMPNN candidates as full 320-aa canonical Eco1 sequences. It is a planned
ColabFold `colabfold_batch` CLI request for BU SCC execution; no fold model is
run by this materializer.

Generic fold-check request/report contracts now live in
`dnadesign.thread.foldcheck`. Eco1 owns candidate selection, WT sequence
reconstruction, threshold policy, and SCC storage posture.

### Implemented Slice: ColabFold Output Normalizer v1

`dnadesign.thread.adapters.colabfold` now normalizes completed ColabFold output
directories into generic fold-check rows. It discovers model files by request
sequence id, extracts pLDDT from PDB B-factors, summarizes PAE JSON when
available, computes C-alpha RMSD against the WT runtime baseline or an explicit
reference PDB, and emits failure rows for missing outputs or missing metrics.
Output discovery uses a one-pass ColabFold output index and rank-token parsing
so numeric candidate ids cannot be mistaken for model ranks.

Eco1's thin wrapper lives at
`src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/foldcheck_report/`.
It locates the current study request manifest, calls the generic adapter, and
writes `foldcheck_report.parquet` under the study workspace. Phase 3 validation
now fails fast when that report is absent or lacks WT/candidate coverage.

### Completed Slice: SCC Fold Smoke and Report v1

LocalColabFold is installed on BU SCC under
`/projectnb/dunlop/esouth/tools/localcolabfold`, and the
`colabfold_batch --help` preflight succeeds when the pixi environment `lib/`
directory is first on `LD_LIBRARY_PATH`. LocalColabFold is the install/runtime
environment for the ColabFold CLI, not a separate fold model or API. The SCC
clone was fast-forwarded to the current branch, the materialized fold-check
request was regenerated there, and BU SCC job `6224446` ran the WT baseline
plus five accepted candidates.

The smoke raw output lives on SCC project storage under
`/project/dunlop/esouth/foldcheck/eco1_rt/eco1_colabfold_foldcheck.6224446/`.
After the residue-map missing-backbone policy was renamed, the fold-check
request hash changed and the compact local `foldcheck_report.parquet` was
re-normalized from that SCC raw output. The report now validates against the
current request hash, with six `accepted` rows and 91 explicit `errored` rows
for candidates outside the smoke subset.

The smoke also exposed one generic portability bug: fold-check request hashes
must not include host-local FASTA/output paths. `dnadesign.thread.foldcheck`
now hashes request intent, sequence hashes, threshold policy, and upstream
artifact hashes while leaving local paths as operator metadata.
Smoke reports may include errored rows for candidates outside the subset.
Downstream selection must require accepted fold-check rows for selected
candidates.

### Completed Slice: Full Foldcheck Batch v1

The full WT plus 96-candidate ColabFold request was run through
`colabfold_batch` on BU SCC as job `6228979`, using `--num-models 1`. Raw output
remains under `/project/dunlop/esouth/foldcheck/eco1_rt/full_96_a4948b42/`.
The compact `foldcheck_report.parquet` was normalized on SCC, synced back to
the study workspace, and validates locally with 97 accepted rows. Reserve
heavier multi-model checks for selected candidates after this first coverage
pass. The owner split remains explicit: `docs/bu-scc` owns scheduler/runtime templates,
`dnadesign.thread.adapters.colabfold` owns output normalization,
`dnadesign.thread.foldcheck` owns the normalized report contract, and
`eco1_rt_repack` owns candidate selection, WT sequence reconstruction, and
threshold policy.

### Implemented Slice: Foldcheck Review v1

The study now materializes `foldcheck_review/` under the study workspace. This
review bundle writes:

- `foldcheck_candidate_ranking.parquet`
- `foldcheck_structure_panel.yaml`
- `chimerax/ec86_fold_panel.cxc`
- `atlas_subset_manifest.yaml`

The ranking table joins ProteinMPNN candidate metrics with ColabFold metrics and
uses explicit RMSD fields. `wt_runtime_ca_rmsd` is the candidate-to-WT
ColabFold runtime-model comparison from the normalized fold report.
`cryoem_mapped_ca_rmsd` is direct mapped-position RMSD to the ec86kit/7V9U
backbone from the local normalized full structure set. Raw ColabFold outputs
remain on SCC project storage; the study-local review bundle keeps one PDB per
accepted fold row for ChimeraX and direct structure review.

The current review classes are `strong_fold_preserved: 17`,
`good_fold_preserved: 53`, `low_confidence: 9`, `review_band: 14`, and
`structural_outlier: 3`. The structure-panel manifest selects WT, best-folded
candidates, high-RMSD outliers, low-pLDDT rows, intermediate rows, and
deterministic controls. The Atlas subset manifest records a contrastive panel
for later semantic annotation; it is not a candidate acceptance gate.

### Next Slice: SAE Window Summary, Feasibility, Selection, And RT-Only Handoff v1

Inspect the selected and full ChimeraX review scripts before candidate
selection. The Atlas hash-lookup/on-demand probe selected WT plus all 96
candidates and allowed five new requests. WT was accepted with rich sparse Atlas
data and one Atlas/ESMFold-derived structure registry row; the first four
synthetic candidates still returned explicit 404 rows, and the remaining 92
synthetic candidates were left unattempted. Do not continue retrying that hash
lookup path for synthetic candidates unless the API behavior changes. If Atlas
context is needed through the no-auth API, add a separate sequence-similarity
artifact for semantic-neighborhood review. Then materialize the
assembly/synthesis feasibility report and only then prepare downstream RT-only
handoff records.

The authenticated Biohub ESMC/logits path is now implemented as a separate
query-time SAE lane. The current conservative run selected WT plus all 96
fold-accepted ProteinMPNN candidates, and the materialization accepted all
97 query rows. It writes
`biohub_esmc_sae_profile.parquet`, `biohub_esmc_protein_features.parquet`,
`biohub_esmc_residue_features.parquet`,
`biohub_esmc_feature_catalog.parquet`, and a redacted
`biohub_esmc_request_manifest.yaml`. This lane is for synthetic sequence SAE
annotation, not Atlas lookup and not fold validation.

The next implementation should not treat current Biohub SAE feature indices as
the interpreted Atlas feature panel. The all-97 Biohub run used
`esmc-300m-2024-12-sae-layer23-k64-codebook65536`, so feature ids are tied to
that exact model, layer, sparsity, and dictionary. The next table should first
summarize sparse activations over declared Eco1 residue windows, then attach
biological labels only when a source-backed interpretation exists for the exact
SAE model.

The materialized review-deliverable foundation makes the first visuals explicit
rather than free-floating illustrations. It writes a single
`review_deliverables/review_deliverable_manifest.yaml` with manifest-relative
figure paths, input hashes, alt text, plain descriptions, interpretation
limits, and a lightweight or optional/heavy flag. The marimo notebook reads the
manifest and pre-rendered assets; dogfood validation includes static notebook
checks and HTML export. The deliverable sequence is:

1. MSA plurality/mask context: Eco1/Ec86 target row first, declared display
   subset for the 303-row clade 9 alignment, full-row inspection in HTML or
   marimo, and background markings for `>=25%` WT-plurality protected columns.
2. Linear-plus-3D mask context: an off-white Ec86 RT scaffold with categorical
   overlays for catalytic anchors, Wang/Ec86 direct-contact priors, retained
   DNA/RNA 5 A contacts, clade 9 plurality protection, and mutable design canvas.
3. ProteinMPNN candidate diversity: score/global score, mutation count,
   sequence identity to WT, sampling temperature/seed, and mutation density.
4. ColabFold structure review: a cached top/bottom/control structure panel first
   and an all-97 contact sheet only as an optional/heavy cached artifact.
5. WT Biohub ESMC SAE feature frames: one feature per frame on the WT structure,
   labelled by exact SAE model and feature index, with source-backed names only.
6. Biohub ESMC feature-window heatmap: WT plus candidates sorted by structural
   review metrics, with a declared feature/window subset instead of all features.
7. Feasibility and handoff matrix after feasibility and selection tables exist.

After the SAE summary and review-deliverable foundation, materialize
`feasibility_report.parquet`, `selection/candidate_selection_panel.parquet`, and
the RT-only `candidate_handoff.yaml`.

### Implemented Slice: ESM Atlas Semantic Audit v1

The repo now has a small Atlas semantic-audit lane for WT plus fold-accepted
Eco1 candidates. The generic API boundary is
`dnadesign.thread.adapters.esm_atlas`; the Eco1 wrapper is
`operations/materialization/atlas_semantic_profile/`. The current command uses
the all-97 request, explicit on-demand provenance, and a bounded request cap:

```bash
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.materialization.atlas_semantic_profile \
  --repo-root . \
  --sequence-limit all \
  --allow-fold-on-miss \
  --prediction-set-id atlas_esmfold_on_miss_all97_20260626 \
  --resume-existing \
  --max-new-requests 5 \
  --request-sleep-seconds 1.5
```

The materializer writes compact study-local Parquet artifacts:

- `atlas_semantic_profile.parquet`
- `atlas_protein_activations.parquet`
- `atlas_residue_activations.parquet`
- `atlas_feature_catalog.parquet`
- `structure_predictions/structure_prediction_registry.parquet`

The all-97 hash-lookup/on-demand probe is materialized locally. It produced 97
profile rows: WT accepted, four synthetic ProteinMPNN candidates as explicit
404 rows, and 92 synthetic candidates marked
`atlas_request_not_attempted_due_to_max_new_requests`. The accepted WT row
carries 2,095 sparse protein-level activations, 20,480 sparse per-residue
activations, and 100 feature-catalog rows. Sparse activation tables avoid dense
16,384-feature matrices and avoid repeating long feature descriptions per
residue. The WT Atlas-generated PDB is written through the generic
`thread.structure_predictions` registry instead of being folded into the Atlas
semantic-profile table.

This is not a fold-validation replacement and not a functional assay. The
plain claim is:

```text
ProteinMPNN = fold-compatible sequence proposal
ColabFold = structural fidelity gate
Biohub ESMC/SAE = query-time semantic annotation
ESM Atlas = public-protein neighborhood context where available
Assay = functional truth
```

For Ec86, the first processivity-oriented feature panel should be treated as a
hypothesis panel around polymerase mechanics, not a measured processivity score:

- thumb/palm nucleic-acid binding and C-terminal thumb context;
- motif B / primer-grip context;
- N-terminal fingers/palm context for structured-template handling;
- DxD/YADD metal-coordination context;
- pre-catalytic helix and open/closed gating context;
- broad RT/RdRp palm-core features as fold/class sanity checks only.

Do not rank candidates by a hidden composite "processivity" score. Use Atlas
features to stratify a small experimental panel after fold acceptance:

- semantic-retained candidates;
- semantic-shifted candidates;
- thumb-retained / fingers-shifted candidates;
- primer-grip-shifted candidates;
- random fold-accepted controls.

Before biochemical data are inspected, freeze the feature panel, residue
windows, normalization rule, fold thresholds, semantic flag definitions,
selection strata, assay endpoints, and primary analysis plan. Atlas/SAE may
prioritize review or assay design; it may not claim processivity, strand
displacement, or hairpin readthrough.

Keep the implementation pragmatic. Do not add a wider `semantic_profile`
framework until a second semantic backend needs the same report contract. The
Eco1 wrapper owns which candidates are queried and how the feature panel is
interpreted in the study record.

### Implementation Rules

- Start with validators and fixture materializers before backend execution.
- Keep each slice callable and testable without running later slices.
- Use explicit state values: `scaffold`, `fixture`, `materialized`,
  `accepted`, or `rejected`.
- Do not add a `run_all` command until every slice has a deterministic
  validator and negative-path test.
- Do not add hidden migration shims from the study-local schemas to future
  `thread` contracts. Promotion is a breaking contract update with an explicit
  migration note.
- Do not let backend selection fall through from ProteinMPNN to LigandMPNN, or
  from real fold metrics to fixtures. A changed backend or fixture mode is a
  new run id and an operator-visible decision.
- Do not let Atlas, ESMC, or SAE output become an acceptance gate without a
  separate policy slice. Model-derived affiliations can prioritize review or
  assay candidates; they do not prove strand displacement, processivity, or
  hairpin unwinding.

### Phase Gates

| Phase | Minimum acceptance |
| --- | --- |
| Phase 0 scaffold | Study records, schema stubs, policy pages, and negative fixture cases exist. |
| Phase 1 contract | Profile, artifact-chain, mask, fold, feasibility, and handoff validators reject the known negative cases. |
| Phase 2 backend ingest | Materialized runtime artifacts exist with nonfixture states, schema-valid fields, and upstream hashes. |
| Phase 3 fold-check report | Full fold-check report and fold-review artifacts validate for WT plus the accepted candidate set. This is not handoff readiness. |
| Phase 4 downstream promotion | RT-only candidate handoff is accepted or rejected by the downstream RT-lnRNA contract without creating construct subjects implicitly. |

### Current Known Gaps

- Structure source, chain policy, retained context, and residue-numbering
  policy are selected and materialized locally as structure artifacts.
- Contact evidence is materialized locally from the retained DNA/RNA context.
- Conservation source discovery and source authority are documented and
  selected, provider candidate-pool acquisition exists, selected source FASTAs
  pass sufficiency, and both selected conservation alignments are accepted under
  the current Clustal Omega policy. `conservation_profile.parquet`, generic
  `aligner.msa` MSA visualization sidecars, and the conservative diagnostic
  `mask_set.yaml`, `thread_plan.yaml`, and
  `proteinmpnn_request/request_manifest.yaml`, `sample_table.parquet`,
  `candidate_table.parquet`, and the ColabFold CLI
  `foldcheck_request/foldcheck_request_manifest.yaml` are materialized locally.
  A full WT plus 96-candidate ColabFold `foldcheck_report.parquet` is
  materialized and validates. Fold-check review, the local full-fold PDB set,
  selected-panel Atlas lookup, and all-97 Biohub ESMC query-time SAE profile
  are materialized. Feasibility report and candidate handoff are not
  materialized.
- Phase 1 now has content validators for the local structure, contact,
  conservation, mask, thread-plan, ProteinMPNN request, and fold-check request
  artifacts. Phase 2 backend ingest passes locally through the candidate table,
  and Phase 3 fold-check report validation passes for the full report. Optional
  Atlas semantic annotation, feasibility review, and downstream selection are
  the next gates.

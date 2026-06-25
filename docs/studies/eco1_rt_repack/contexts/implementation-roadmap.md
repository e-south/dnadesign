---
doc_id: study-eco1-rt-repack-implementation-roadmap
surface: study-context
study_id: eco1_rt_repack
owner: dnadesign-maintainers
last_verified: 2026-06-24
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

The validator owns only study-local scaffold and structure-artifact checks. It
does not hide MPNN/fold execution behind a run-all command. The current
`dnadesign.thread` surface covers generic ProteinMPNN request, sample-ingest,
and candidate-table mechanics; fold-check, feasibility, and handoff mechanics
may graduate to `thread` only through separate contract promotions.

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
outputs/thread/eco1_rt_conservative_v1/backbone_bundle.yaml
outputs/thread/eco1_rt_conservative_v1/residue_map.parquet
```

The backbone bundle records typed chain inventory: RT chain `A` as the design
backbone, DNA chain `D` as retained context, and RNA chains `E` and `F` as
retained context. The residue map records all 320 canonical Eco1 RT positions:
309 mapped positions and 11 unresolved terminal positions fixed by policy.

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
outputs/thread/eco1_rt_conservative_v1/contact_profile.parquet
```

The profile is joined through `residue_map.parquet`, uses the pinned
`ec86kit` per-residue DNA/RNA minimum-distance table, encodes the conservative
20 A contact threshold from `eco1_rt_v1.profile.yaml`, and keeps unresolved
terminal residues non-designable. The Phase 1 validator now checks the contact
profile source hash, threshold, canonical row coverage, unresolved policy, and
mask-threshold consistency.

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
outputs/thread/eco1_rt_conservative_v1/conservation_profile.parquet
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
`residue_map.parquet`, fixing audited NAxxH, YADD, VTG, and RT1-RT7 canonical
interval spans. The same ontology records Wang/Ec86 RT-msDNA/msrRNA interface
candidates as candidate-prior rows. Those candidates guide contact review; they
do not create additional `manual_mask=true` rows by themselves.

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

### Completed Slice: Backend Sample Ingest v1

`sample_table.parquet` is materialized from the validated ProteinMPNN request.
The active batch `eco1_rt_p25_5a_n96_20260624` uses an explicit local
ProteinMPNN checkout, runs official helper parity checks before sampling,
preserves request hash, seed, temperature, fixed-position source, and
no-fallback policy, and records a separate backend-run manifest.

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
ColabFold/AlphaFold-family CLI request for BU SCC execution; no fold model is
run by this materializer.

Generic fold-check request/report contracts now live in
`dnadesign.thread.foldcheck`. Eco1 owns candidate selection, WT sequence
reconstruction, threshold policy, and SCC storage posture.

### Next Slice: ColabFold Smoke and Report Ingest v1

Fast-forward the BU SCC clone to the pushed branch, verify Phase 2 there, then
run a small ColabFold smoke job from the materialized request with
`docs/bu-scc/jobs/eco1-colabfold-foldcheck.qsub`. Normalize the smoke output
into `foldcheck_report.parquet` before scaling to the full 96-candidate request.
The report must include WT baseline, runtime parameter hash, candidate ids,
explicit pass/fail thresholds, and rows for runtime failures before any
downstream promotion.

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

### Phase Gates

| Phase | Minimum acceptance |
| --- | --- |
| Phase 0 scaffold | Study records, schema stubs, policy pages, and negative fixture cases exist. |
| Phase 1 contract | Profile, artifact-chain, mask, fold, feasibility, and handoff validators reject the known negative cases. |
| Phase 2 backend ingest | Materialized runtime artifacts exist with nonfixture states, schema-valid fields, and upstream hashes. |
| Phase 3 downstream promotion | RT-only candidate handoff is accepted or rejected by the downstream RT-lnRNA contract without creating construct subjects implicitly. |

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
  `candidate_table.parquet`, and the ColabFold-planned
  `foldcheck_request/foldcheck_request_manifest.yaml` are materialized locally.
  The fold-check runtime report, feasibility report, and candidate handoff are
  not materialized.
- Phase 1 now has content validators for the local structure, contact,
  conservation, mask, thread-plan, and ProteinMPNN request artifacts. Phase 2
  backend ingest passes locally through the candidate table. Fold-check runtime
  validation is the next gate.

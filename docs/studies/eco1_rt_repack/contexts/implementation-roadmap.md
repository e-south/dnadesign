---
doc_id: study-eco1-rt-repack-implementation-roadmap
surface: study-context
study_id: eco1_rt_repack
owner: dnadesign-maintainers
last_verified: 2026-06-20
---

## Implementation Roadmap

This roadmap is the implementation-facing checklist for the Eco1 RT repack
study and the planned reusable `thread` tool. It is intentionally staged so
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
is the first executable study-owned CLI seam. Domain validation now lives under
`src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/`, with
separate modules for profile contracts, artifact-chain contracts, structure
authority, conservation-source authority, and materialized runtime artifacts.
Keep new checks in those semantic modules instead of growing the CLI entrypoint.

Supported commands:

```bash
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.contract_validation --repo-root . --phase phase0_scaffold
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.contract_validation --repo-root . --phase phase1_thread_contract
```

The validator owns only study-local scaffold and structure-artifact checks. It
does not run MPNN/fold backends or create `dnadesign.thread`. Reusable
artifact-chain mechanics may graduate to `thread` only through a separate
breaking-contract promotion.

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

Phase 1 now fails on the next conservation/contact/mask blockers until contact
profile materialization is run:

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

Phase 1 now fails on the next conservation/mask blockers:

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

`src/dnadesign/aligner/msa/` provides the generic alignment seam for the next
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
clade 9 / II-A3 QC, so the next runtime gate is alignment, not source
reselection.

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

After this slice, the regenerated source FASTA sufficiency gate passes. The next
runtime slice is to run both selected source FASTAs through
`dnadesign.aligner.msa` using the selected Clustal Omega backend, and then
materialize `conservation_profile.parquet`.

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
  selected, provider candidate-pool acquisition exists, and a conservation-profile
  materializer exists for explicit aligned FASTA inputs. The generic MSA
  execution seam lives in `dnadesign.aligner.msa`, and the study-owned
  conservation-alignment materializer now orchestrates source sufficiency,
  declared MSA args, selected profile ids, and alignment bundle manifests. The
  generic runner stages stdout to a temporary FASTA, records stderr, and
  publishes an accepted aligned FASTA only after validation. The former
  full-roster broad profile ran for roughly four hours under the declared
  high-sensitivity MAFFT policy without a complete broad alignment and is no
  longer the active denominator. The selected `ec86_iia3_cluster42_1_conservation_v1` profile now
  has accepted local profile-level alignment
  evidence, and generic `aligner.msa` MSA visualization sidecars are
  implemented and materialized locally for that accepted profile. Those
  sidecars can render Eco1-owned motif anchors, explicit exemplar-row windows,
  selected-row overview panels, and plurality/gap histograms, but they remain
  display-only and do not satisfy conservation evidence. The complete
  two-profile aligned FASTA bundle, materialized conservation profile, mask set,
  sampling plan, sample table, candidate table, fold-check report, feasibility
  report, and candidate handoff are not materialized.
- Readiness checks are still scaffold-level study preflight checks. They must
  grow content validators before Phase 1 can be marked accepted.

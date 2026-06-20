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
declares the selected MSA source authority for `broad_retron_rt` and
`eco1_like_retron_rt`. It uses Mestre et al. 2020 Supplementary Table S1 as
the accession roster, declares NCBI and BV-BRC provider policies, pins the
target row to the ec86kit Eco1 RT reference sequence hash, and records the
T301/A301 mismatch in public `WP_099010551.1` as a fail-fast target-row issue.

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

This slice deliberately does not fetch provider sequences or run MAFFT. Those
remain explicit source-alignment curation work because no checked-in FASTA,
A3M, or MSA source artifact currently exists.

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
  selected, and a conservation-profile materializer exists for explicit aligned
  FASTA inputs. No real aligned FASTA bundle, materialized conservation profile,
  mask set, sampling plan, sample table,
  candidate table, fold-check report, feasibility report, or candidate handoff
  is materialized.
- Readiness checks are still scaffold-level study preflight checks. They must
  grow content validators before Phase 1 can be marked accepted.

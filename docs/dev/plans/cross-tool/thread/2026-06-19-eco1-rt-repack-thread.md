## Eco1 RT Repack Thread

**Status:** proposed development specification
**Owner:** dnadesign-maintainers
**Last verified:** 2026-06-19
**Study id:** `eco1_rt_repack`
**Primary study surface:** `docs/studies/eco1_rt_repack/`
**Planned reusable tool surface:** `thread`

### 0. Document Framing And Authority

This proposal defines the first contract for an Eco1 reverse-transcriptase
fixed-backbone redesign campaign. The campaign is large enough to be a
standalone checked-in study, but it must keep downstream collaboration with
`rt_lnrna_sponging_construct_triage` explicit rather than duplicating that
study's paired RT-lnRNA construct semantics.

The planned reusable tool is `thread`: a short, punchy package name for
fixed-backbone protein sequence design intent. `thread` should own backbone
intake, residue maps, mask sets, inverse-folding sampling plans, candidate
tables, fold-check normalization, and candidate handoff contracts once those
mechanics are no longer Eco1-specific.

Eco1 is the first profile and anchor, not the reusable ontology.

### 1. Problem Statement

The study needs a conservative, inspectable computational path for redesigning
Eco1 RT sequence variants against a fixed structural context while protecting
catalytic, substrate-contact, and conserved residues. The immediate goal is a
design and QA surface, not a wet-lab protocol and not a prime-editing campaign.

The downstream biological setting is transcription-factor sponging through
RT-lnRNA constructs. This study therefore emits RT candidate handoffs that the
RT-lnRNA study can choose to promote into paired construct subjects. It does
not own lnRNA slots, Construct context views, Reader SPOP labels, LatentDNA
interpretation, or OPAL campaign state.

### 2. Owner Boundaries

| Surface | Owns | Does not own |
| --- | --- | --- |
| `thread` | Generic fixed-backbone protein design contracts, masks, request/result normalization, candidate ids, ranking, fold-check interpretation, and candidate handoff bundles. | Eco1 biological policy, Construct realization, model-process execution ledgers, Infer sidecars, RT-lnRNA labels. |
| `infer` | Optional model-process execution and writeback for ProteinMPNN, LigandMPNN, AlphaFold, or ColabFold once explicit adapter contracts exist. | Redesign ontology, mask algebra, candidate identity, or study-specific acceptance policy. |
| `aligner` | Pairwise nucleotide alignment and generic MSA bundle mechanics through the public `dnadesign.aligner` API. | Eco1 roster curation, provider fetching, target-row adjudication, conservation scoring, or hidden mask machinery. |
| `construct` | Sequence realization, named slots, placement/window feasibility, and construct-context views. | Deciding which redesigned RT variants are worth ordering or assaying. |
| `permuter` | DMS and explicit variant intent through public `dnadesign.permuter` APIs. | ProteinMPNN-generated design proposals unless they are intentionally imported as variant records through a later contract. |
| `eco1_rt_repack` | Eco1 profile, protected-residue policy, source structure choice, candidate batch policy, and downstream handoff decisions. | Reusable MPNN mechanics, generic fold-check adapters, or RT-lnRNA paired-construct semantics. |

Backend rule: Phase 1 treats `thread` as the design-domain owner and `infer` as
an optional execution provider. `thread` may build backend requests and ingest
declared backend results, but it must not hide process execution, retry policy,
or model fallback inside design-domain code. If `infer` executes a backend run,
`thread` records only the public run id, request hash, result hash, and adapter
kind.

Boundary rule: a generic contract may mention `BackboneBundle`,
`ResidueMaskSet`, or `ThreadCandidate`; it may not mention Eco1 residues,
retron motifs, prime editing, TF sponging, or study-specific synthesis choices.
Those terms belong in `eco1_rt_repack` policy or downstream studies.

### 3. Information Architecture

The study record owns the current campaign state:

```text
docs/studies/eco1_rt_repack/
  README.md
  record/
  routes/
  contexts/
  operations/
  workbench/
```

Study folders have these meanings:

| Directory | Meaning | Mutation policy |
| --- | --- | --- |
| `record/` | Current state, dataset registry, and campaign step list. | Small, factual, human-readable records only. |
| `routes/` | One-hop navigation and owner handoffs. | Update whenever a new authoritative surface is added. |
| `contexts/` | Durable study policy and rationale. | Study-specific biology belongs here, not in `thread`. |
| `operations/contract/` | Machine-readable providerless contracts, readiness groups, fixtures, and schemas. | Keep split by object or phase; avoid catch-all YAML files. |
| `operations/runtime/` | Planned command grouping and future runtime lanes. | No generated runtime payloads in the repo record root. |
| `workbench/ontology/` | Vocabulary and naming semantics. | Prefer short object names with precise definitions. |
| `workbench/design_sets/` | Named candidate-batch intent and inclusion policy. | No large candidate tables; link to runtime artifacts. |
| `workbench/provenance/` | Source authority, numbering, and evidence audits. | Every mutable residue must be traceable here before sampling. |

The planned reusable tool, once promoted, should use this shape:

```text
src/dnadesign/thread/
  README.md
  docs/
  workspaces/
  src/
    contracts/
    structure/
    evidence/
    masks/
    adapters/
    candidates/
    handoffs/
  tests/
```

`thread` source modules should remain small and contract-oriented:

| Module | Owns | First public seam |
| --- | --- | --- |
| `contracts/` | Pydantic/dataclass contracts and serialization boundaries. | Validate `BackboneBundle`, `ResidueMaskSet`, `ThreadPlan`, and `CandidateHandoff`. |
| `structure/` | Structure intake, chain selection, residue-id normalization. | Emit `BackboneBundle` and `ResidueMap`. |
| `evidence/` | Conservation/contact profile ingestion and normalization. | Emit `ConservationProfile` and `ContactProfile`; do not perform hidden searches. |
| `masks/` | Deterministic mask algebra and conflict checks. | Emit `ResidueMaskSet` or fail with named conflicts. |
| `adapters/` | Thin request/result adapters for declared model backends. | Emit backend request manifests and normalize already-declared results; no process runner and no implicit fallback backend. |
| `candidates/` | Deduplication, ids, mutation tables, and ranking. | Emit deterministic `ThreadCandidate` rows. |
| `handoffs/` | Downstream handoff bundles and hash closure. | Emit `CandidateHandoff`. |

Do not create `thread` source modules until a tracer bullet needs executable
contracts. Until then, the study docs and fixtures are the stable planning
surface.

Promotion checklist before `src/dnadesign/thread/` becomes executable:

- Public imports are exposed from `dnadesign.thread` only after contracts pass
  fixture and negative-path tests.
- CLI stance is explicit: Phase 1 may provide `thread validate` and
  `thread materialize-fixture`; it should not provide `thread run-all`.
- Tests mirror module ownership: `contracts/`, `structure/`, `evidence/`,
  `masks/`, `adapters/`, `candidates/`, and `handoffs/` each get focused
  contract tests before shared integration tests.
- Package data is limited to schemas, small fixtures, and docs; model weights,
  PDB/mmCIF files, fold predictions, and candidate tables stay in workspaces or
  runtime outputs.
- Architecture-boundary checks must reject cross-tool imports from sibling
  internals such as `dnadesign.infer.src.*`, `dnadesign.construct.src.*`, or
  `dnadesign.permuter.src.*`.
- The first implementation slice is now a study-owned Phase 0/1 contract
  validator at
  `src/dnadesign/studies/units/eco1_rt_repack/operations/contract_validation.py`.
  This is the CLI entrypoint; domain validators live under
  `src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/` so profile,
  structure authority, conservation-source, materialized-artifact, and mask-case
  checks do not accrete into a single module.
- The second implementation slice is a study-owned structure materializer at
  `src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/structure/`.
  It emits thread-shaped local runtime artifacts for `backbone_bundle.yaml` and
  `residue_map.parquet`. It is not a model-execution framework and does not
  create `dnadesign.thread`.
- The third implementation slice is a study-owned contact materializer at
  `src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/contact/`.
  It emits thread-shaped local runtime contact evidence for
  `contact_profile.parquet` from the retained DNA/RNA context; Phase 1 now
  fails on missing conservation evidence and `mask_set.yaml`.
- The fourth implementation slice is a study-owned conservation source
  contract at
  `docs/studies/eco1_rt_repack/workbench/provenance/conservation-sources.yaml`.
  It validates the declared MSA roster authority, provider policy, target-row
  sequence hash, gap denominator, threshold, and plurality rule without
  materializing `conservation_profile.parquet`.
- The fifth implementation slice is a study-owned conservation materializer at
  `src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/conservation/`.
  It consumes explicit aligned FASTA inputs and emits thread-shaped long-form
  `conservation_profile.parquet` evidence. It does not fetch provider
  sequences, run MAFFT, or create `dnadesign.thread`; the real aligned FASTA
  bundle remains the next source-data blocker.
- The Aligner modernization slice adds public `dnadesign.aligner.msa` contracts
  for FASTA validation, MAFFT preflight/execution, atomic aligned FASTA
  publication, stderr sidecars, and aligned FASTA bundle manifests. Eco1 still
  owns the source roster, provider policy, target-row hash, and conservation
  scoring.
- The conservation alignment slice adds
  `src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/conservation_alignments/`.
  It validates Eco1 source FASTA sufficiency, parses declared MAFFT args, calls
  the public `dnadesign.aligner.msa` runner, supports selected profile ids for
  operational runs, and writes an Eco1 alignment bundle index. The former
  full-Mestre broad-profile run with the declared high-sensitivity MAFFT policy
  was stopped after roughly four hours and is no longer the active broad
  denominator. The next study-owned source-data slice is bounded homolog
  selection for `broad_tao_homolog_rt`; only after that should broad alignment
  be rerun through `dnadesign.aligner.msa`.
- The conservation source-sequence bundle slice adds
  `src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/source_sequences/`.
  It consumes explicit local provider FASTA caches plus `source_records.yaml`,
  inserts the ec86kit target row, rejects undeclared providers and unexplained
  exclusions, and emits unaligned source FASTA bundles plus manifests. It does
  not fetch live provider rows, run MAFFT, score conservation, or create
  `dnadesign.thread`.
- The conservation provider-source slice adds
  `src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/source_sequences/provider_sources/`.
  It remains study-owned because it derives accessions from the Eco1/Mestre
  source contract and writes explicit unresolved-provider ledgers. It does not
  belong in `aligner` or future `thread`; those tools consume declared FASTA
  bundles and evidence artifacts after source authority is already resolved.
- The broad conservation source policy was revised from a whole-roster Mestre
  denominator to `broad_tao_homolog_rt`: a Tao-like target-centered bounded
  homolog panel drawn from Mestre S1. The full Mestre roster remains
  candidate-pool/classification context, not Phase 1 conservation evidence.

### 4. Contract Objects

Generic contract names should not include Eco1:

| Object | Purpose |
| --- | --- |
| `BackboneBundle` | Structure source, chain selection, sequence hash, ligand/cofactor policy, and fixed-backbone declaration. |
| `ResidueMap` | Mapping among structure residue ids, canonical protein positions, CDS positions, and design positions. |
| `ConservationProfile` | MSA-derived per-position conservation values and mapping quality. |
| `ContactProfile` | Structure-derived protected contacts to substrate, metal, RNA/DNA, ligand, or protein interfaces. |
| `ResidueMaskSet` | Final mutable/fixed/protected/unresolved residue masks with source contributions. |
| `ThreadPlan` | Sampling backend, seed set, temperature schedule, fixed positions, tied positions, and expected sample count. |
| `ThreadSample` | Raw sampled sequence with backend provenance and score fields. |
| `ThreadCandidate` | Deduplicated candidate with stable id, mutation list, mask checks, rank, and provenance. |
| `FoldCheckReport` | Fold-validation metrics, thresholds, and explicit per-candidate pass/fail state. |
| `AssemblyFeasibilityReport` | Computational mutation-window and haplotype feasibility for full-sequence candidates. Study economics and Construct placement remain outside the generic object. |
| `CandidateHandoff` | Selected candidates plus upstream artifact hashes and downstream target contract. |

Contract state must be explicit:

| State | Meaning |
| --- | --- |
| `scaffold` | Shape exists for planning but is not runtime-valid. |
| `fixture` | Deterministic example exists and is allowed only in fixture lanes. |
| `materialized` | Runtime artifact exists and records provenance. |
| `accepted` | Artifact passed the declared checks for its phase. |
| `rejected` | Artifact failed with named reasons. |

No code path may silently promote `scaffold` or `fixture` artifacts to
`materialized` evidence.

### 5. Artifact Chain

Use generic artifact names inside `thread`-style fixtures:

```text
backbone_bundle.yaml
residue_map.parquet
conservation_profile.parquet
contact_profile.parquet
mask_set.yaml
thread_plan.yaml
sample_table.parquet
candidate_table.parquet
foldcheck_report.parquet
feasibility_report.parquet
candidate_handoff.yaml
```

Study-specific fixtures may include `eco1_rt_v1` in profile ids and filenames,
but reusable object names remain generic.

Authority rules:

- Runtime table authority should be Parquet for typed, bounded columns.
- Review exports may be CSV, but they are not the source of truth unless the
  contract explicitly says so.
- Every artifact must record `schema_id`, `schema_version`, `artifact_id`,
  `created_by`, `created_at`, and upstream artifact hashes.
- A table with zero rows is valid only when the schema declares an empty-output
  state and a reason.
- A missing optional metric is encoded as null plus a reason column; it is never
  inferred from another metric.
- Fold-check reports must record the wild-type baseline artifact, runtime
  parameter hash, threshold id, threshold values, and per-candidate pass/fail
  reason. A raw pLDDT or RMSD value without those fields is not acceptance
  evidence.
- Candidate handoffs are emitted by the design-domain contract, but selected
  candidate policy is study-owned. Downstream studies accept or reject the
  handoff through their own promotion contract.

### 5.1 Implementation Roadmap

Implementation should proceed in narrow, replaceable slices. Each slice has a
single owner, explicit input and output artifacts, and a negative-path test.

| Slice | Owner | Input | Output | Must fail when |
| --- | --- | --- | --- | --- |
| `thread-contracts-v1` | `thread` | checked-in schemas and fixtures | importable contract models and validators | required fields conflict, unknown state is accepted, or fixture artifacts satisfy materialized contracts |
| `structure-authority-v1` | study then `thread.structure` | selected structure source and reference FASTA | `backbone_bundle.yaml`, `residue_map.parquet` | source, chain, retained context, numbering origin, or sequence hash is pending |
| `conservation-source-contract-v1` | study then `thread.evidence` | declared MSA roster and sequence-provider authority | selected source contract | target row, provider, gap denominator, threshold, or plurality rule is missing |
| `evidence-profiles-v1` | `thread.evidence` | residue map plus declared MSA/contact source artifacts | `conservation_profile.parquet`, `contact_profile.parquet` | source hashes, gap policy, retained context, or per-position mapping is missing |
| `mask-algebra-v1` | `thread.masks` plus study policy | evidence profiles and manual mask groups | `mask_set.yaml` | missing evidence implies designability, conflicts are unresolved, or mutable set is empty/all-mutable |
| `sampling-request-v1` | `thread.adapters` | mask set and backend policy | `thread_plan.yaml`, backend request manifest | backend selection, seeds, temperature, fixed positions, or fallback policy is implicit |
| `sample-ingest-v1` | `thread.adapters` or `infer` provider | backend result manifest | `sample_table.parquet` | backend result lacks run id, request hash, sequence hash, seed, temperature, score, or status |
| `candidate-qa-v1` | `thread.candidates` | sample table and mask set | `candidate_table.parquet` | duplicate ids, mask violations, unstable row ordering, or missing mutation windows appear |
| `foldcheck-normalize-v1` | `thread.adapters` or `infer` provider | candidate table and fold runtime output | `foldcheck_report.parquet` | WT baseline, thresholds, runtime parameters, provenance, or failure rows are missing |
| `window-feasibility-v1` | `thread.candidates` plus study policy | accepted full-sequence candidates | `feasibility_report.parquet` | structural coupling or nearest-parent distance is unreported for windowed candidates |
| `candidate-handoff-v1` | `thread.handoffs` plus study selection policy | candidate, fold, and feasibility reports | `candidate_handoff.yaml` | upstream hash closure, nonfixture fold acceptance, full-sequence validation, or downstream target is missing |
| `rt-lnrna-acceptance-v1` | downstream study | RT-only candidate handoff | accepted/rejected promotion record | construct subject ids are preclaimed or required RT-only fields are absent |

Do not collapse these slices into one module or command. A future orchestration
command may run them in order only after each slice has its own validator.

### 6. Conservative Tracer Bullet

The first executable slice should be deliberately small:

1. Ingest one Eco1/Ec86 RT structure source and one Eco1 RT reference sequence.
2. Emit `residue_map.parquet`.
3. Emit one conservative `mask_set.yaml`.
4. Ingest or run eight deterministic ProteinMPNN or LigandMPNN samples.
5. Deduplicate to a small `candidate_table.parquet`.
6. Attach fixture or real fold-check metrics.
7. Emit `candidate_handoff.yaml`.
8. Verify that mutable positions are mapped, mask violations are zero,
   candidate ids are deterministic, and every emitted artifact records upstream
   hashes.

The tracer bullet should be split into five fail-fast gates:

| Gate | Required output | Fails when |
| --- | --- | --- |
| Structure authority | `backbone_bundle.yaml`, `residue_map.parquet` | Chain, sequence hash, numbering, or ligand policy is unresolved. |
| Mask contract | `conservation_profile.parquet`, `contact_profile.parquet`, `mask_set.yaml` | Any mutable residue lacks mapping or any fixed reason conflicts. |
| Sampling plan | `thread_plan.yaml`, `sample_table.parquet` | Backend, seed set, temperature, fixed positions, or sample count is implicit. |
| Candidate QA | `candidate_table.parquet`, `foldcheck_report.parquet` | Mask violations, duplicate ids, missing fold provenance, or threshold-free acceptance appears. |
| Handoff | `feasibility_report.parquet`, `candidate_handoff.yaml` | Selected ids are not full-sequence validated or downstream target is undeclared. |

Current checked-in readiness records use the supported study preflight kinds
available today, mostly `path_exists`. Those checks are scaffolding only. Phase
1 acceptance requires content validators for schema fields, Parquet columns,
artifact state, upstream hashes, fixture/materialized separation, and the
negative cases listed in the Eco1 study fixtures.

The current Phase 0 validator covers checked-in YAML/Markdown scaffold content:
profile fields, forbidden cross-tool identity fields, artifact-chain schema
shape, no-fallback and fixture/materialized invariants, conservative mask-case
coverage, structure-source selection status, retained-context policy,
profile/source consistency, residue-numbering policy status, and the
materialized structure-artifact content for the Eco1 tracer bullet. Runtime
artifact mechanics may still graduate to `thread` only through a later explicit
promotion.

### 7. Fail-Fast Policy

- No design run without a valid `BackboneBundle`.
- No mutable residue without a `ResidueMap` row.
- No MSA conservation mask without a target-row mapping and gap policy.
- No contact mask without structure provenance and distance threshold policy.
- No sample table without backend version, seed, and mask provenance.
- No fold-check acceptance without explicit thresholds and runtime provenance.
- No pooled-window handoff unless every recombined candidate has preserved
  haplotype/window linkage, nearest-parent distance, structural-coupling flags,
  and computational QA.
- No RT-lnRNA promotion without the downstream study's construct-subject
  contract.
- No hidden backcompat shim from a study-local schema to a future `thread`
  schema. Migration must be an explicit breaking contract update.
- No fallback from ProteinMPNN to LigandMPNN, or from real fold metrics to
  fixture metrics, without a new run id and an explicit operator decision.
- No monolithic `run_all` command before the five gates above can be run and
  verified independently.

### 8. Source Evidence

The motivating external method is Tao et al., "AI-guided redesign of
laboratory-evolved reverse transcriptases enhances prime editing", Nature
Biotechnology, published 2026-05-21, DOI `10.1038/s41587-026-03149-6`.
The paper cites ProteinMPNN as the core fixed-backbone sequence-design method
and provides GitHub/SRA resources for the prime-editor RT redesign campaign.
This study adapts the computational pattern to Eco1 RT for sponging use; it
does not adopt the prime-editing objective.

### 9. Open Questions For Refinement

- Which first backend result format should `thread` ingest: direct
  ProteinMPNN/LigandMPNN files, `infer` result manifests, or both through
  separate explicit adapters?
- Which parts of protein MSA evidence should promote from the Eco1 materializer
  into `thread.evidence` after `dnadesign.aligner.msa` emits aligned FASTA
  bundles?
- Which Eco1/Ec86 structural source is the first profile authority?
- Which fold-check metrics are required for the first real candidate batch?
- What exact downstream acceptance record will
  `rt_lnrna_sponging_construct_triage` emit after it accepts or rejects RT-only
  candidates?

# Study Surfaces

Use these surfaces in this order for Eco1 RT repack status or routing.

## Checked-In Study Record

- `docs/studies/eco1_rt_repack/README.md`: first-hop ontology and directory
  contract.
- `docs/studies/eco1_rt_repack/record/status.md`: factual current phase,
  blockers, and next actions.
- `docs/studies/eco1_rt_repack/record/datasets.yaml`: declared source,
  planned output, and downstream dataset posture.
- `docs/studies/eco1_rt_repack/record/campaign.yaml`: planned campaign and
  phase manifest.
- `docs/studies/eco1_rt_repack/routes/README.md`: one-hop handoff map by owner
  surface.

## Contract Shelves

- `operations/ops.study.yaml`: split contract entrypoint for lifecycle,
  artifacts, snapshot, and readiness parts.
- `operations/contract/lifecycle/`: phase and mode declarations.
- `operations/contract/surfaces/artifacts.yaml`: planned artifact names and
  ownership boundaries.
- `operations/contract/readiness/`: record-only readiness checks for profile,
  structure authority, mask contract, sampling plan, fold-check runtime,
  assembly feasibility, candidate handoff, and downstream RT-lnRNA handoff.
- `operations/contract/fixtures/thread/`: Eco1 profile and conservative-mask
  fixture stubs.
- `operations/contract/schemas/`: profile, artifact-chain, candidate-handoff,
  and RT-only downstream acceptance schemas.
- `operations/runtime/command-groups/pipeline.yaml`: planned command-group
  posture; it is not an executable pipeline.
- `workbench/provenance/structure-sources.yaml`: selected `ec86kit`-backed
  Eco1/Ec86 protomer authority.
- `workbench/provenance/residue-numbering-policy.yaml`: selected numbering
  policy that remains separate from the local materialized
  `residue_map.parquet`.
- `workbench/provenance/conservation-sources.yaml`: selected MSA source
  contract for `broad_retron_rt` and `eco1_like_retron_rt`.
- `workbench/provenance/conservation-source-discovery.md`: source-discovery
  note for method and roster priors before MSA materialization.
- `src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/source_sequences/roster_cache/`:
  study-owned roster-cache materializer for Mestre roster plus explicit
  provider FASTA source inputs.
- `src/dnadesign/aligner/msa/`: generic aligned FASTA bundle API for MAFFT
  preflight/execution; it is not Eco1 source authority or conservation scoring.

## Context Pages

- `contexts/fixed-backbone-method.md`: computational method lane and
  ProteinMPNN/LigandMPNN posture.
- `contexts/implementation-roadmap.md`: implementation slice order, code homes,
  artifact inputs/outputs, and negative paths.
- `contexts/msa-method.md`: MSA reproduction method, Tao-style conservation
  scoring rule, and T301/A301 source-mismatch handling.
- `contexts/residue-mask-policy.md`: catalytic/contact/conservation mask
  policy.
- `contexts/fold-validation-policy.md`: fold-check acceptance and no-go
  signals.
- `contexts/synthesis-feasibility-policy.md`: full-gene versus bounded-window
  computational handoff policy.

## Study-Owned Source

- `src/dnadesign/studies/units/eco1_rt_repack/`: reserved for future
  study-owned helpers only.
- `src/dnadesign/studies/units/eco1_rt_repack/operations/contract_validation.py`:
  Phase 0/1 CLI entrypoint for checked-in contract validation.
- `src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/`:
  semantic validator package for profile, artifact-chain, mask-case, structure
  authority, conservation-source, and materialized-artifact checks.
- `src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/structure/`:
  study-owned materializer for local `backbone_bundle.yaml` and
  `residue_map.parquet`.
- `src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/contact/`:
  study-owned materializer for local `contact_profile.parquet` from retained
  DNA/RNA context distances.
- `src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/conservation/`:
  study-owned materializer for local `conservation_profile.parquet` from
  explicit aligned FASTA inputs.
- `src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/source_sequences/`:
  study-owned materializer for unaligned source FASTA bundles from explicit
  provider caches and exclusion ledgers.
- `src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/source_sequences/contracts/`:
  shared parser/accessor package for `conservation-sources.yaml` so
  source-sequence materializers do not duplicate source-authority semantics.
- `src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/source_sequences/sufficiency/`:
  pre-alignment gate package for cache/hash/accession/support and target-row
  checks before MAFFT.
- `src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/conservation_artifacts.py`:
  semantic validator for materialized conservation-profile metadata, source
  hashes, residue-map joins, and Tao-style conservation rule consistency.
- Reusable fixed-backbone mechanics belong in a future `src/dnadesign/thread/`
  package after a tracer bullet makes the executable contract real.

## Router Rule

If a follow-up asks for RT-lnRNA construct state, leave this skill and route to
`docs/studies/rt_lnrna_sponging_construct_triage/routes/README.md`.

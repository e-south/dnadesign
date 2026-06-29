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
- `operations/runtime/command-groups/pipeline.yaml`: sequential command-group
  contract for independently runnable study-owned materialization lanes; it is
  not a hidden run-all pipeline.
- `workbench/provenance/structure-sources.yaml`: selected `ec86kit`-backed
  Eco1/Ec86 protomer authority.
- `workbench/provenance/structure-preprocessing.yaml`: raw 7V9U
  dimer-to-selected-protomer provenance policy, including selected protein,
  DNA, RNA, excluded paired-protomer context, and the explicit non-objective of
  preserving paired-protomer dimerization.
- `workbench/provenance/residue-numbering-policy.yaml`: selected numbering
  policy that remains separate from the local materialized
  `residue_map.parquet`.
- `workbench/provenance/conservation-sources.yaml`: selected MSA source
  contract for `ec86_clade9_conservation_v1` and `ec86_iia3_cluster42_1_conservation_v1`; full Mestre
  rows are candidate/context only, not the broad conservation denominator.
- `workbench/provenance/conservation-source-discovery.md`: source-discovery
  note for method and roster priors, updated as the MSA and conservation
  profile materialization state advances.
- `src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/source_sequences/provider_sources/`:
  study-owned provider-source acquisition package for declared NCBI Protein and
  BV-BRC FASTA source files plus explicit unresolved-provider ledgers.
- `src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/source_sequences/roster_cache/`:
  study-owned roster-cache materializer for Mestre roster plus explicit
  provider FASTA source inputs; source records now include pairwise target
  coverage, pairwise identity, length, motif-marker, and hard-reject QC
  metadata before MSA execution.
- `src/dnadesign/aligner/msa/`: generic aligned FASTA bundle API for MAFFT and
  Clustal Omega preflight/execution, atomic output publication, stderr sidecars,
  and run provenance; it is not Eco1 source authority or conservation scoring.
- `src/dnadesign/aligner/msa/visualization/`: generic MSA QC and
  visualization sidecar API for accepted aligned FASTA files; it is not Eco1
  source authority, provider-cache policy, conservation scoring, or mask
  designability.
- `src/dnadesign/aligner/msa/visualization/contracts/`: generic request/result
  models and YAML readers for annotation tracks, exemplar rows, and panel
  specs.
- `src/dnadesign/aligner/msa/visualization/materialization/`: generic
  visualization orchestration, QC calculations, bundle manifests, CSV, and HTML
  report writers.
- `src/dnadesign/aligner/msa/visualization/renderers/`: generic SVG renderers
  and label-placement helpers; study-owned biological annotations are inputs,
  not renderer constants.
- `workbench/ontology/rt-annotation-tracks.yaml`: Eco1-owned target-position
  motif annotation tracks that the generic MSA visualization API can render;
  they are not mask sources or designability rules.
- `workbench/ontology/manual-mask-authority.yaml`: Eco1-owned
  manual mask ontology. Under `eco1_rt_clade9_plurality25_direct_contact5a_v1`
  it protects NAxxH, YADD, VTG, and Wang/Ec86 direct substrate-contact priors; RT1-RT7
  intervals remain annotation/review labels, not blanket hard masks.
- `workbench/ontology/msa-exemplar-rows.yaml`: Eco1-owned explicit row
  selections that ground motif-window visualization; they are not conservation
  denominators or representative-set claims.
- `workbench/ontology/msa-panel-spec.yaml`: Eco1-owned display contract for
  selected-row whole-alignment overview and plurality/gap histogram panels; it
  is not a conservation denominator, mask source, or designability rule.
- `src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/conservation_alignments/`:
  study-owned orchestration package that validates Eco1 source-bundle
  sufficiency, parses declared MSA backend args, calls `dnadesign.aligner.msa`,
  supports selected profile ids, and writes the Eco1 alignment bundle index.

## Context Pages

- `contexts/fixed-backbone-method.md`: computational method lane and
  ProteinMPNN/LigandMPNN posture.
- `contexts/implementation-roadmap.md`: implementation slice order, code homes,
  artifact inputs/outputs, and negative paths.
- `contexts/msa-method.md`: MSA reproduction method, Tao-style conservation
  scoring rule, and T301/A301 source-mismatch handling.
- `contexts/residue-mask-policy.md`: catalytic/contact/conservation mask
  policy.
- `contexts/fold-validation-policy.md`: fold-check request, runtime acceptance,
  and no-go signals.
- `contexts/synthesis-feasibility-policy.md`: full-gene versus bounded-window
  computational handoff policy.

## Study-Owned Source

- `src/dnadesign/studies/units/eco1_rt_repack/`: reserved for future
  study-owned helpers only.
- `src/dnadesign/studies/units/eco1_rt_repack/operations/contract_validation.py`:
  Phase 0/1 CLI entrypoint for checked-in contract validation.
- `src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/`:
  shared validator package for profile, artifact-chain, evidence aggregation,
  models, constants, and the Phase 0/1 validation suite.
- `src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/conservation/`:
  conservation-specific contract package for MSA source authority, source-set
  selection rules, materialized conservation-profile metadata, source hashes,
  residue-map joins, and Tao-style conservation rule consistency.
- `src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/masks/`:
  mask-specific contract package for conservative mask cases, checked-in manual
  mask source ontology helpers, RT1-RT7 interval source contracts, generated
  `manual_mask_authority.yaml` validation, and generated `mask_set.yaml`
  validation.
- `src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/structure/`:
  structure-specific contract package for selected authority, residue-numbering
  policy, profile/source consistency, materialized `backbone_bundle.yaml`,
  materialized `residue_map.parquet`, materialized
  `structure_preprocessing_manifest.yaml`, shared structure-provenance hash
  closure, and materialized `contact_geometry_profile.parquet`.
- `src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/structure/`:
  study-owned materializer for local `backbone_bundle.yaml` and
  `residue_map.parquet`.
- `src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/structure_preprocessing/`:
  study-owned materializer for `structure_preprocessing_manifest.yaml`; it
  hash-checks selected ec86kit model provenance and machine-checks retained
  RT/DNA/RNA chain ontology before contact geometry.
- `src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/contact/`:
  study-owned materializer for local `contact_profile.parquet` from retained
  DNA/RNA context distances.
- `src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/contact_geometry/`:
  study-owned Biopython/mmCIF materializer for local
  `contact_geometry_profile.parquet` from the selected 7V9U/ec86kit protomer
  context, including side-chain, backbone, contact-density, and
  retained-chain-count evidence. The package is intentionally split so
  `structure_io.py` owns mmCIF/chain parsing, `rows.py` owns atom-distance row
  math, `writer.py` owns Parquet schema/metadata emission, and `pipeline.py`
  stays orchestration-only.
- `src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/conservation/`:
  study-owned materializer for local `conservation_profile.parquet` from
  explicit aligned FASTA inputs.
- `src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/conservation_alignments/`:
  study-owned materializer for alignment bundle manifests from
  sufficiency-passing source FASTA inputs. The selected
  `ec86_clade9_conservation_v1` and
  `ec86_iia3_cluster42_1_conservation_v1` source FASTAs now pass sufficiency,
  and a complete accepted two-profile Clustal Omega aligned FASTA bundle is
  materialized locally.
- `src/dnadesign/studies/units/eco1_rt_repack/operations/masking/`:
  study-owned mask-row algebra package for protected/non-fixed row
  composition, source-count summaries, and missing-backbone handling. Runtime
  writers and Phase 1 validators route through this package instead of
  duplicating row math.
- `src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/mask_set/`:
  study-owned materializer for `mask_set.yaml`. The artifact is materialized
  under `eco1_rt_clade9_plurality25_direct_contact5a_v1`, with row classes
  `protected`, `non_fixed`, and `non_fixed_missing_backbone`.
- `src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/thread_plan/`:
  study-owned materializer for local `thread_plan.yaml` from the validated
  simple mask. It emits an explicit planned `proteinmpnn` request with seeds,
  temperatures, request hash, fixed and mutable positions, terminal
  missing-backbone exclusions, and no backend fallback.
- `src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/proteinmpnn_request/`:
  thin study-owned wrapper that resolves Eco1 paths, validates selected
  structure provenance, and delegates generic ProteinMPNN request mechanics to
  `src/dnadesign/thread/adapters/proteinmpnn/`. It does not execute
  ProteinMPNN.
- `src/dnadesign/thread/adapters/proteinmpnn/`:
  generic fixed-backbone ProteinMPNN adapter. It owns
  canonical-to-chain-local conversion, helper JSONL payloads, protein-only
  backbone export, request manifests, official-checkout preflight, helper
  parity checks, backend run manifests, sample-table writing, request/sample
  validation, and generic hashing. It must not contain Eco1, Ec86, Mestre,
  Wang, or motif policy.
- `src/dnadesign/thread/candidates/`:
  generic candidate-table builders. ProteinMPNN candidate rows are built from
  normalized sample tables and request manifests; canonical mutation accounting
  must use the request's canonical-to-chain-local position map.
- `src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/proteinmpnn_sample_ingest/`:
  thin study-owned wrapper that resolves Eco1 paths and delegates generic
  ProteinMPNN execution/result mechanics to
  `src/dnadesign/thread/adapters/proteinmpnn/`. It writes
  `sample_table.parquet` from accepted backend rows.
- `src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/candidate_table/`:
  thin study-owned wrapper that resolves Eco1 paths and delegates generic
  candidate-table construction to `src/dnadesign/thread/candidates/`. It writes
  `candidate_table.parquet` from accepted backend rows.
- `src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/foldcheck_request/`:
  thin study-owned wrapper that reconstructs full canonical WT/candidate
  sequences from the residue map and candidate table, then writes a
  ColabFold-planned fold-check FASTA and request manifest without running a fold
  model.
- `src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/foldcheck_report/`:
  thin study-owned wrapper that resolves Eco1 paths and delegates completed
  ColabFold output parsing to `src/dnadesign/thread/adapters/colabfold/`. It
  writes `foldcheck_report.parquet` from completed runtime outputs and does not
  submit jobs or copy raw model directories.
- `src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/foldcheck_review/`:
  study-owned fold-review wrapper. It ranks full fold-check rows, separates
  WT-runtime RMSD from direct ec86kit/7V9U mapped-residue RMSD, stages a small
  structure-panel manifest and ChimeraX script, stages the full local PDB set for
  ChimeraX review, writes an Atlas subset manifest, and emits SVG review plots,
  a selected-structure ChimeraX overlay PNG when the executable is available,
  plus a scoped marimo notebook through `review_visual_manifest.yaml`. It does
  not copy full raw ColabFold output trees or accept candidates.
- `src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/`:
  study-owned visual-deliverable wrapper. It writes
  `review_deliverable_manifest.yaml`, canonical-coordinate MSA plurality/mask
  context, linear mask tracks, a ChimeraX mask-context script/render,
  ProteinMPNN diversity panels, linked foldcheck_review SVG/PNG visuals, linked
  WT ESMC masked-marginal plots, MSA-vs-ESMC model-constraint audit plots, and
  a manifest-backed marimo notebook. Manifest paths are relative to the manifest
  location, and notebook dogfood includes static checks plus HTML export. It does
  not rerun ProteinMPNN, ColabFold, Biohub, Atlas, or candidate
  selection.
- `src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/foldcheck/`:
  fold-check request/report contract package. It validates the request
  manifest, FASTA sequence ids, full 320-aa Eco1 sequence length, accepted
  candidate coverage, request hash, upstream artifact hashes, and report
  coverage once runtime output exists.
- `src/dnadesign/thread/adapters/colabfold/`:
  generic ColabFold output normalizer. It discovers model files by request
  sequence id, extracts pLDDT/PAE/RMSD-style fields, and emits failure rows
  without importing Eco1 policy.
- `src/dnadesign/thread/adapters/esm_atlas/`:
  generic ESM Atlas API adapter and sparse-activation normalizer. It owns
  bounded no-auth Atlas lookup calls, query/raw-response hashes, top feature
  summaries, sparse protein-level activations, sparse per-residue activations,
  compact feature catalog rows, and explicit error rows. It must not interpret
  Eco1 function or act as a candidate acceptance gate.
- `src/dnadesign/thread/adapters/biohub_esmc/`:
  generic authenticated Biohub ESMC adapter for query-time SAE activations. It
  owns runtime-only credential loading, the documented `/api/v1/encode` ->
  `/api/v1/logits` flow, encoded SAE tensor decoding, sparse protein/residue
  feature rows, redacted manifests, and explicit error rows. It must not
  interpret Eco1 function, claim processivity, or act as a candidate acceptance
  gate.
- `src/dnadesign/thread/structure_predictions/`:
  generic registry for model-predicted structures. It keeps Atlas/ESMFold,
  ColabFold, Biohub Fold, Boltz, or other future structure predictions
  provenance-separated by backend, model, request hash, raw-response hash, and
  structure hash.
- `src/dnadesign/thread/foldcheck/`:
  generic fold-check request/report contract package. It owns WT-baseline
  request manifests, fold-check FASTA writing, report schemas, and report
  validation; it does not run ColabFold or choose Eco1 thresholds.
- `src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/atlas_semantic_profile/`:
  thin study-owned wrapper that selects WT plus fold-accepted Eco1 sequences and
  delegates Atlas lookup/normalization to `src/dnadesign/thread/adapters/esm_atlas/`.
  It writes compact semantic-profile and sparse SAE activation artifacts, plus a
  structure-prediction registry for any explicitly authorized Atlas on-demand
  structures. These are not fold-check or processivity evidence.
- `src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/biohub_esmc_sae_profile/`:
  thin study-owned wrapper that selects fold-accepted Eco1 sequences and
  delegates authenticated ESMC/logits request normalization to
  `src/dnadesign/thread/adapters/biohub_esmc/`. It writes compact query-time SAE
  artifacts for synthetic sequences and keeps the Biohub token out of manifests,
  logs, docs, and generated artifacts.
- `src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/biohub_esmc_wt_mutation_scoring/`:
  thin study-owned wrapper that selects the accepted WT fold-check sequence,
  delegates single-mutant grid semantics to `dnadesign.permuter`, and delegates
  authenticated Biohub ESMC sequence-logit calls to the Biohub adapter. It
  writes WT-only masked-marginal position entropy and substitution LLR tables,
  plus a mask-context join. This is model-derived mutation-scoring evidence for
  review, not experimental DMS and not a current-mask update.
- `src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/sampling/`:
  sampling artifact contract package. Phase 2 validates `thread_plan.yaml` and
  `proteinmpnn_request/request_manifest.yaml` separately from
  `sample_table.parquet` and `candidate_table.parquet`, so request planning,
  backend adaptation, backend ingest, and candidate construction remain distinct
  gates. `sampling/thread_plan/` owns thread-plan metadata, upstream-hash,
  expected-field, and request-hash checks, while `sampling/sample_table.py` and
  `sampling/candidate_table.py` validate normalized backend rows and candidates.
- `src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/contact_risk/`:
  study-owned materializer for `contact_risk_profile.yaml`. It is a
  review artifact that joins nearest-distance contact evidence, atom-class
  contact geometry, conservation masks, manual-mask authority, Wang/Ec86
  candidate priors, and selected simple-mask row status.
- `src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/contact_risk/`:
  contact-risk artifact contract package. It validates review metadata, required
  evidence-availability statuses, and per-row risk-class fields without making
  review artifacts control protected residues.
- `src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/manual_mask_authority/`:
  study-owned materializer for local `manual_mask_authority.yaml` from the
  checked-in manual mask-authority ontology and `residue_map.parquet`.
- `src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/source_sequences/`:
  study-owned materializer for unaligned source FASTA bundles from explicit
  provider caches and exclusion ledgers.
- `src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/source_sequences/provider_sources/`:
  study-owned materializer for provider FASTA source roots before source-cache
  filtering; current local run produced 350 NCBI records, 1464 BV-BRC records,
  and 113 explicit unresolved-provider ledger rows.
- `src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/source_sequences/contracts/`:
  shared parser/accessor package for `conservation-sources.yaml`, including
  provider accession policy, so source-sequence materializers do not duplicate
  source-authority semantics.
- `src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/source_sequences/sufficiency/`:
  pre-alignment gate package for cache/hash/accession/support and target-row
  checks before MSA execution. It passes locally for the regenerated Ec86
  clade 9 and II-A3/`42_1` source FASTA bundles.
- Broader reusable fold-model execution, feasibility, and handoff mechanics are
  still planned; the current executable `thread` surfaces are the generic
  ProteinMPNN adapter, candidate-table package, and fold-check contract package.

## Router Rule

If a follow-up asks for RT-lnRNA construct state, leave this skill and route to
`docs/studies/rt_lnrna_sponging_construct_triage/routes/README.md`.

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
- `src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/contact_risk/`:
  study-owned materializer for `contact_risk_profile.yaml`. It is a
  review artifact that joins nearest-distance contact evidence, atom-class
  contact geometry, conservation masks, manual-mask authority, Wang/Ec86
  candidate priors, and selected simple-mask row status.
- `src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/surface_accessibility/`:
  study-owned materializer for `surface_accessibility_profile.parquet`. It
  computes complex-context Biopython Shrake-Rupley SASA from the selected Ec86
  RT-msDNA-msrRNA structure and records one canonical row per Eco1 RT position.
  It is an earlier check, not an input to the current mask rule.
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
- Reusable fixed-backbone mechanics belong in a future `src/dnadesign/thread/`
  package after a tracer bullet makes the executable contract real.

## Router Rule

If a follow-up asks for RT-lnRNA construct state, leave this skill and route to
`docs/studies/rt_lnrna_sponging_construct_triage/routes/README.md`.

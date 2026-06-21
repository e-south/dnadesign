## Eco1 RT Repack Status

**Owner:** dnadesign-maintainers
**Last verified:** 2026-06-20
**Status surface:** record-only

### Current Phase

Phase 1 contract: structure-authority artifacts, contact evidence,
conservation source authority, provider-candidate source acquisition, the
alignment-bundle materializer, and the conservation-profile materializer are
implemented, selected, or materialized; the Eco1-like aligned FASTA profile is
accepted locally and has partial generic `aligner.msa` visualization sidecars.
The broad conservation denominator has been revised to a Tao-like bounded
homolog profile, so the complete aligned FASTA bundle, conservation profile,
and mask artifacts remain blocked until bounded-homolog source records are
materialized.

The study has a checked-in record root, planned contract fixtures, and a
selected Eco1/Ec86 structure-authority policy derived from the sibling
`ec86kit` protomer-1 workspace. The first structure artifacts are now
materialized under `outputs/thread/eco1_rt_conservative_v1/`: a typed
`backbone_bundle.yaml`, all-position `residue_map.parquet`, and a retained
nucleic-acid-context `contact_profile.parquet`. The study-owned validator
checks profile, artifact-chain, mask-case, structure-authority,
retained-context, residue-numbering policy, structure-artifact content, and
contact-profile provenance/content. It also validates the selected MSA source
contract and materialized conservation-profile content when an explicit aligned
FASTA bundle is supplied. It does not yet have executable `thread` code, MPNN
adapter execution, fold-check adapters, or generated candidate outputs.

Local contact-profile sanity read: 320 canonical positions are represented;
309 resolved positions have retained DNA/RNA distance evidence; 11 terminal
unresolved positions remain fixed. Under the current 20 A conservative contact
threshold, all 309 resolved positions pass the contact-mask predicate, so the
first mask-set slice should either prove an all-fixed conservative mask or
explicitly introduce a relaxed threshold profile.

Conservation source read: local retron-prior PDFs provide method and context
priors, but no checked-in MSA. Mestre et al. 2020 provides the roster authority
and candidate pool, Tao et al. 2026 provides the target-centered homolog MSA
and WT/plurality/frequency conservation-mask method, and Simon et al. 2019
provides the RT-region visual annotation grammar. The selected Phase 1
conservation profiles are now `broad_tao_homolog_rt` and
`eco1_like_retron_rt`. The full Mestre S1 roster remains
`full_mestre_retron_rt` context and candidate-pool evidence; it is not the
Phase 1 conservation scoring denominator.

The checked-in source contract pins the MSA target row to the ec86kit reference
sequence and rejects silent replacement by the mismatched public NCBI
`WP_099010551.1` row. The provider-source acquisition code may fetch full
Mestre candidate-pool records so the next selector can compute coverage,
identity, motif support, and diversity, but roster-cache/source-record
materialization now refuses `broad_tao_homolog_rt` until the bounded homolog
selector exists. The previous local full-roster run fetched 350 NCBI Protein
records and 1464 BV-BRC feature protein records, recorded 113 unresolved
BV-BRC rows as explicit exclusions, and produced a 1814-row broad source FASTA
bundle. That output is superseded candidate-pool context under the revised
policy, not accepted conservation-denominator evidence.

The study-owned alignment-bundle materializer routes accepted source FASTAs
through the public `dnadesign.aligner.msa` MAFFT bundle contract while keeping
Eco1 profile ids, source paths, and target-row policy in this study. The
generic MAFFT seam stages stdout to a temporary FASTA, records stderr as a
sidecar, and publishes an accepted aligned FASTA only after validation; the
study orchestrator can run either all declared profiles or selected profile
ids. An interactive real-data attempt using the former full-roster broad
profile and declared `mafft --globalpair --maxiterate 1000 --reorder` policy
was interrupted after roughly four hours of active CPU before a complete
`broad_tao_homolog_rt.aligned.fasta` was produced. A selected-profile run of
`eco1_like_retron_rt` completed locally through the same declared MAFFT policy
with 47 aligned records, one aligned length of 560 aa, the pinned
`eco1_rt_ec86kit_reference` target row present, MAFFT v7.526, return code 0,
and hash-linked stdout/stderr manifests. This is accepted profile-level
evidence, not a complete two-profile aligned FASTA bundle. The conservation
materializer accepts explicit aligned FASTA inputs, but it does not fetch
provider records, align, or invent source sequences. The generic
`dnadesign.aligner.msa.visualization` API now writes MSA QC YAML,
per-position QC CSV, SVG, and HTML reports from accepted aligned FASTA inputs.
It can also render optional target-position annotation tracks, explicit
exemplar-row motif windows, selected-row whole-alignment overview panels, and
plurality/gap histograms. Eco1 owns a study ontology annotation file for the
audited `NAxxH`, `YADD`, and `VTG` motif anchors, informed by Simon et al.'s
validated retron RT MSA visual grammar and Mestre et al.'s RT0-RT7/clade
annotation context. Eco1 also owns an explicit exemplar-row selection and an
MSA panel spec so the report can show accession-grounded named rows, bordered
context spans, motif boxes, and consensus-style summaries beside the target
instead of relying only on an opaque global heatmap. These display records are
context only and do not define plurality or designability. Simon-style
cross-family controls are documented as a future display-reference bundle, not
silently inserted into the conservation MSA. The current local visualization output is
intentionally partial: it covers `eco1_like_retron_rt` and records
`broad_tao_homolog_rt` as missing. Eco1 owns only the profile IDs, target-row hash
policy, annotation/exemplar/panel data, and generated study output location;
the plotting/QC mechanics stay in `aligner.msa`.

Validator command:

```bash
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.contract_validation --repo-root . --phase phase0_scaffold
```

Phase 1 readiness is intentionally fail-fast at the next evidence/mask gate:

```bash
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.contract_validation --repo-root . --phase phase1_thread_contract
```

### Current Next Actions

1. Implement `conservation-bounded-homolog-selector-v1`: consume the full
   Mestre candidate-pool provider cache, compute target-centered coverage,
   identity, motif support, and deterministic diversity/cap metadata, then
   materialize bounded `broad_tao_homolog_rt` source records. Do not fall back
   to raw roster order.
2. Regenerate source FASTA sufficiency for the bounded broad profile, then run
   `conservation-alignment-bundle-v1` through the hardened
   `dnadesign.aligner.msa` seam. The smaller `eco1_like_retron_rt` profile has
   an accepted local aligned FASTA.
3. Regenerate generic `aligner.msa` MSA visualization sidecars with the Eco1
   RT annotation-track, exemplar-row, and panel-spec YAML once the broad profile
   is accepted so both profiles are visible in the same QC report.
4. Materialize `conservation_profile.parquet` from the declared aligned FASTA
   bundle.
5. Materialize the conservative `mask_set.yaml` only after contact and
   conservation evidence are mapped onto `residue_map.parquet`.
6. Promote only reusable artifact-chain mechanics into `thread` after the
   study-local validator proves the contract shape.
7. Generate a small explicit sampling plan with backend, seed, temperature,
   fixed-position, request-hash, and no-fallback policy.
8. Define the downstream RT-lnRNA candidate handoff accepted by
   `rt_lnrna_sponging_construct_triage`.

### Blockers

- No executable `thread` package exists.
- Structure authority, chain policy, retained context policy, and residue
  numbering policy are selected and materialized locally as structure artifacts;
  contact evidence is materialized locally from the retained DNA/RNA context;
  reusable `thread.structure` and `thread.evidence` code does not exist yet.
- MSA/conservation source discovery and source authority are documented and
  selected. Provider candidate-pool acquisition exists, but the revised
  `broad_tao_homolog_rt` bounded-homolog source records are not materialized.
  The selected `eco1_like_retron_rt` profile has an accepted local aligned
  FASTA, but the complete aligned FASTA bundle and materialized conservation
  profile do not yet exist.
- Generic `aligner.msa` visualization sidecars are implemented and
  materialized locally for the accepted Eco1-like profile, with an Eco1-owned
  motif-anchor annotation track, exemplar-row windows, selected-row overview
  panels, and plurality/gap histograms available for richer
  publication-oriented inspection; the report is partial until the broad
  alignment is accepted.
- No conservative mask set exists.
- No sampling plan or backend-ingest contract is materialized.
- No sample table, candidate table, or backend result manifest exists.
- No fold-check runtime report with WT baseline, thresholds, and runtime
  parameter hash exists.
- No assembly feasibility report exists.
- No RT-only candidate handoff or RT-lnRNA acceptance record exists.

### Non-Goals

- Wet-lab protocol execution.
- Prime-editing campaign ownership.
- Replacing the RT-lnRNA sponging construct study.
- Hiding Eco1-specific biology inside a reusable tool package.

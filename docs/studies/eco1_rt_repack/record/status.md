## Eco1 RT Repack Status

**Owner:** dnadesign-maintainers
**Last verified:** 2026-06-20
**Status surface:** record-only

### Current Phase

Phase 1 contract: structure-authority artifacts, contact evidence,
conservation source authority, provider-candidate source acquisition, the
alignment-bundle materializer, and the conservation-profile materializer are
implemented, selected, or materialized; the previously accepted Eco1-like
aligned FASTA profile has partial generic `aligner.msa` visualization sidecars
but predates the current Clustal Omega policy. The broad conservation
denominator has been revised to the Mestre Ec86-containing RT clade 9 panel, so
the complete aligned FASTA bundle, conservation profile, and mask artifacts
remain blocked until the selected source FASTAs are aligned under the current
Clustal Omega backend.

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
and clade/type hierarchy, Tao et al. 2026 provides the homolog MSA
and WT/plurality/frequency conservation-mask method, and Simon et al. 2019
provides the RT-region visual annotation grammar. The selected Phase 1
conservation profiles are now `ec86_clade9_conservation_v1` and
`ec86_iia3_cluster42_1_conservation_v1`. The full Mestre S1 roster remains
`mestre_all_retron_rt_context` context and candidate-pool evidence; it is not the
Phase 1 conservation scoring denominator.

The checked-in source contract pins the MSA target row to the ec86kit reference
sequence and rejects silent replacement by the mismatched public NCBI
`WP_099010551.1` row. The provider-source acquisition code may fetch full
Mestre candidate-pool records, but conservation source-record materialization
now defines `ec86_clade9_conservation_v1` by the natural Mestre RT clade 9 unit
rather than by a cap-first selector. The local provider-source run fetched 350
NCBI Protein records and 1464 BV-BRC feature protein records, with 113
unresolved BV-BRC rows recorded as explicit exclusions. The selected source
cache has now been regenerated from Mestre RT clade 9 under declared QC:
`ec86_clade9_conservation_v1` includes 302 rows and excludes 22 rows, while
`ec86_iia3_cluster42_1_conservation_v1` includes 44 rows and excludes 3 rows.
QC records include pairwise target coverage, pairwise identity range status,
length status, motif-marker calls, and hard-reject filters. The superseded
1814-row full-roster broad bundle remains candidate-pool context only, not
accepted conservation-denominator evidence.

The study-owned alignment-bundle materializer routes accepted source FASTAs
through the public `dnadesign.aligner.msa` bundle contract while keeping Eco1
profile ids, source paths, and target-row policy in this study. The generic MSA
backend seam stages output, records stderr as a sidecar, and publishes an
accepted aligned FASTA only after validation; the study orchestrator can run
either all declared profiles or selected profile ids. An interactive real-data
attempt using the former full-roster broad profile and previous
`mafft --globalpair --maxiterate 1000 --reorder` policy was interrupted after
roughly four hours of active CPU before a complete broad aligned FASTA was
produced. The current selected backend is Clustal Omega:
`clustalo --force --outfmt=fasta --threads=1 -i <input_fasta> -o <output_fasta>`.
The earlier `ec86_iia3_cluster42_1_conservation_v1` aligned FASTA is retained
as local historical evidence but should be regenerated under the selected
backend before conservation-profile materialization. The conservation
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
intentionally partial: it covers `ec86_iia3_cluster42_1_conservation_v1` and records
`ec86_clade9_conservation_v1` as missing. Eco1 owns only the profile IDs, target-row hash
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

1. Run `conservation-alignment-bundle-v1` for the selected
   `ec86_clade9_conservation_v1` and
   `ec86_iia3_cluster42_1_conservation_v1` source FASTAs through
   the hardened `dnadesign.aligner.msa` Clustal Omega seam.
2. Regenerate generic `aligner.msa` MSA visualization sidecars with the Eco1
   RT annotation-track, exemplar-row, and panel-spec YAML once the broad profile
   is accepted so both profiles are visible in the same QC report.
3. Materialize `conservation_profile.parquet` from the declared aligned FASTA
   bundle.
4. Materialize the conservative `mask_set.yaml` only after contact and
   conservation evidence are mapped onto `residue_map.parquet`.
5. Promote only reusable artifact-chain mechanics into `thread` after the
   study-local validator proves the contract shape.
6. Generate a small explicit sampling plan with backend, seed, temperature,
   fixed-position, request-hash, and no-fallback policy.
7. Define the downstream RT-lnRNA candidate handoff accepted by
   `rt_lnrna_sponging_construct_triage`.

### Blockers

- No executable `thread` package exists.
- Structure authority, chain policy, retained context policy, and residue
  numbering policy are selected and materialized locally as structure artifacts;
  contact evidence is materialized locally from the retained DNA/RNA context;
  reusable `thread.structure` and `thread.evidence` code does not exist yet.
- MSA/conservation source discovery and source authority are documented and
  selected. Provider candidate-pool acquisition, clade-9 source-record QC, and
  source FASTA sufficiency are locally materialized for the selected
  `ec86_clade9_conservation_v1` and
  `ec86_iia3_cluster42_1_conservation_v1` profiles, but the accepted aligned
  FASTA bundle must still be regenerated under the current Clustal Omega policy
  before the conservation profile can be materialized.
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

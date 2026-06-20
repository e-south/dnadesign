## Eco1 RT Repack Status

**Owner:** dnadesign-maintainers
**Last verified:** 2026-06-20
**Status surface:** record-only

### Current Phase

Phase 1 contract: structure-authority artifacts, contact evidence,
conservation source authority, and the conservation-profile materializer are
implemented or selected; real conservation profile and mask artifacts still
blocked.

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
for both `broad_retron_rt` and `eco1_like_retron_rt`, while Tao et al. 2026
provides the conservation-mask method. The checked-in source contract pins the
MSA target row to the ec86kit reference sequence and rejects silent replacement
by the mismatched public NCBI `WP_099010551.1` row. The study-owned
source-sequence materializer now accepts explicit local provider caches and an
exclusion ledger to emit unaligned source FASTA bundles. The conservation
source-sequence package also exposes a sufficiency preflight that rejects
missing cache roots, placeholder accessions, undersized profile bundles, and
hash drift before MAFFT. The conservation materializer accepts explicit
aligned FASTA inputs, but it does not fetch, align, or invent source sequences.
Generic aligned FASTA execution routes through the public
`dnadesign.aligner.msa` MAFFT bundle contract; Eco1 still owns the source
roster and target-row policy.

Validator command:

```bash
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.contract_validation --repo-root . --phase phase0_scaffold
```

Phase 1 readiness is intentionally fail-fast at the next evidence/mask gate:

```bash
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.contract_validation --repo-root . --phase phase1_thread_contract
```

### Current Next Actions

1. Curate explicit provider FASTA caches and a `source_records.yaml` ledger for
   `broad_retron_rt` and `eco1_like_retron_rt`.
2. Materialize unaligned source FASTA bundles with
   `operations/materialization/source_sequences/`.
3. Run the source-sequence sufficiency gate; do not align bundles that fail it.
4. Align sufficiency-passing source FASTA bundles through `dnadesign.aligner.msa`.
5. Materialize `conservation_profile.parquet` from that declared aligned FASTA
   bundle.
6. Materialize the conservative `mask_set.yaml` only after contact and
   conservation evidence are mapped onto `residue_map.parquet`.
7. Promote only reusable artifact-chain mechanics into `thread` after the
   study-local validator proves the contract shape.
8. Generate a small explicit sampling plan with backend, seed, temperature,
   fixed-position, request-hash, and no-fallback policy.
9. Define the downstream RT-lnRNA candidate handoff accepted by
   `rt_lnrna_sponging_construct_triage`.

### Blockers

- No executable `thread` package exists.
- Structure authority, chain policy, retained context policy, and residue
  numbering policy are selected and materialized locally as structure artifacts;
  contact evidence is materialized locally from the retained DNA/RNA context;
  reusable `thread.structure` and `thread.evidence` code does not exist yet.
- MSA/conservation source discovery and source authority are documented and
  selected; source-sequence bundle, source-bundle sufficiency, and
  conservation-profile materialization code exists, but no real provider cache,
  sufficiency-passing source FASTA bundle, aligned FASTA bundle, or
  materialized conservation profile exists.
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

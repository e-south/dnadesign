---
doc_id: study-eco1-rt-repack-msa-method
surface: study-context
study_id: eco1_rt_repack
owner: dnadesign-maintainers
last_verified: 2026-06-20
---

## MSA Method

This page explains how the Eco1 RT conservation profile should be built. It is
not an MSA, and it does not satisfy `conservation_profile.parquet`.

The machine-readable source contract is:

```text
docs/studies/eco1_rt_repack/workbench/provenance/conservation-sources.yaml
```

The source-discovery note is:

```text
docs/studies/eco1_rt_repack/workbench/provenance/conservation-source-discovery.md
```

### Current Authority

Eco1/Ec86 RT is treated as the ec86kit-pinned reference sequence used by the
selected structure and residue map. The conservation target row must match:

```text
sha256:429a9c9894501e04f48803b96307cea45955f63b85f1461dc25c017e94b7eaeb
```

Mestre et al. 2020 Supplementary Table S1 is the roster authority, not a
finished alignment. The Eco1/Ec86 row is Node 1550, RT clade 9, retron subtype
II-A3, cluster/domain `42_1`, accession `WP_099010551.1`.

The first source groups are:

```text
broad_retron_rt       all Mestre S1 retron RT records after declared filters
eco1_like_retron_rt   Mestre II-A3 cluster 42_1 after declared filters
```

### Procedure

1. Start from the Mestre S1 roster declared in `conservation-sources.yaml`.
2. Split the roster into `broad_retron_rt` and `eco1_like_retron_rt`.
3. Fetch protein sequences through declared providers only:
   `ncbi_protein_efetch` for `WP_*` accessions and
   `bv_brc_feature_protein_fasta` for `fig|*` feature ids.
4. Exclude unresolved provider rows only with an explicit reason; do not
   silently drop them.
5. Materialize unaligned source FASTA bundles from explicit local provider
   caches and `source_records.yaml`; each bundle must insert the ec86kit Eco1
   RT sequence as the explicit target FASTA row.
6. Reject `WP_099010551.1` as the target row unless the T301/A301 discrepancy
   is explicitly adjudicated.
7. Run the source-sequence sufficiency gate; reject missing cache roots,
   placeholder accessions, undersized profile bundles, missing source hashes,
   provider hash drift, and exclusions without reasons.
8. Apply the declared filters: query coverage, identity range, length range,
   required RT/retron motifs, and excluded RT families.
9. Align proteins with the declared MAFFT command from the source contract
   through `dnadesign.aligner.msa`.
10. Map alignment columns back to `residue_map.parquet` through canonical Eco1
   positions, not raw PDB residue ids.
11. Compute conservation using non-gap rows as the denominator.
12. Emit `conservation_profile.parquet` only after every row has source hashes,
    target-row provenance, profile id, WT amino acid, plurality amino acid, WT
    frequency, non-gap count, and pass/fail status.

### Tao-Style Conservation Rule

The Eco1 profile follows the Tao et al. rule shape:

```text
fixed_by_conservation =
  wt_aa_is_plurality_aa
  AND wt_frequency >= conservation_threshold
```

For the first conservative profile, `conservation_threshold` is `0.25`.

The MSA is evidence for masking, not an activity model. It cannot make a
residue designable. Missing MSA evidence fails closed until a later operator
explicitly changes the policy.

### T301/A301 Handling

The selected ec86kit/structure authority has T301. A direct NCBI fetch of
`WP_099010551.1` observed A301. Position 301 is near the C terminus but is
resolved in the selected structure and already contact-proximal under the 20 A
retained-context policy.

This is a source-authority mismatch, not a biological conclusion. The MSA
target must be the ec86kit sequence unless a future contract explicitly
declares a substitution and updates all linked hashes.

### Fail-Fast Rules

- No conservation profile without `conservation-sources.yaml`.
- No target row inferred from a public accession with a sequence mismatch.
- No provider fallback outside the declared provider ids.
- No MAFFT alignment from a source bundle that fails the sufficiency gate.
- No figure-level or prose-only MSA used as materialized evidence.
- No conservation count with gaps in the denominator.
- No fixed-position conservation rule unless WT is the plurality amino acid.
- No designability from missing conservation evidence.

### Source-Sequence Bundle Materializer

The source-sequence bundle materializer is:

```text
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/source_sequences/
```

It consumes explicit local source caches:

```text
<source-cache-root>/source_records.yaml
<source-cache-root>/provider_caches/ncbi_protein_efetch.fasta
<source-cache-root>/provider_caches/bv_brc_feature_protein_fasta.fasta
```

The ledger records `profile_id`, `record_id`, `provider_id`, `accession`,
`status`, and an `exclusion_reason` for excluded rows. The materializer inserts
`eco1_rt_ec86kit_reference` itself, rejects operator-supplied target rows, and
writes unaligned source FASTA plus manifests:

```text
outputs/thread/eco1_rt_conservative_v1/conservation_sources/
```

It does not fetch live provider records and it does not run MAFFT.

Before alignment, run the sufficiency preflight:

```bash
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.materialization.source_sequences.sufficiency --repo-root .
```

This command is expected to fail until real provider caches exist under:

```text
outputs/thread/eco1_rt_conservative_v1/conservation_source_cache/
```

It rejects source bundles that are fixture-like, under-supported relative to
`min_non_gap_count`, not hash-linked to `source_records.yaml` and provider
caches, or populated with placeholder accessions such as synthetic `WP_BROAD`
or `fig|BROAD` records.

### Conservation Profile Materializer

The study-owned materializer is:

```text
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/conservation/
```

It consumes explicit aligned FASTA files, one per selected profile id:

```text
<alignment-root>/broad_retron_rt.aligned.fasta
<alignment-root>/eco1_like_retron_rt.aligned.fasta
```

Each aligned FASTA must include the target row:

```text
eco1_rt_ec86kit_reference
```

The materializer writes:

```text
outputs/thread/eco1_rt_conservative_v1/conservation_profile.parquet
```

It validates the target row against `residue_map.parquet`, records aligned
FASTA source hashes, and emits long-form rows keyed by
`profile_id + canonical_position`.

This materializer does not fetch provider sequences or run MAFFT. The next
source-data slice must use the source-sequence bundles and
`dnadesign.aligner.msa` to materialize the aligned FASTA bundle before this can
create real conservation evidence.

### Next Slice

The next data slice is real provider-cache curation until the source-sequence
sufficiency gate passes. After that, run the explicit source FASTA bundles
through the public `dnadesign.aligner.msa` API to create aligned FASTA bundle
manifests, then run the conservation materializer and confirm Phase 1 advances
to the `mask_set.yaml` blocker only.

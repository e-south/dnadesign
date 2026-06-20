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
   `ncbi_protein_efetch` for NCBI Protein accessions in S1, including
   `WP_*` and GenBank-style protein ids such as `EIJ70524.1`, and
   `bv_brc_feature_protein_fasta` for `fig|*` feature ids.
4. Exclude unresolved provider rows only with an explicit reason; do not
   silently drop them.
5. Materialize provider FASTA source files from the hash-pinned Mestre roster
   table. Provider-missing accessions must be written to an explicit failure
   ledger before they can become excluded source records.
6. Materialize the local roster cache from the hash-pinned Mestre roster table
   and explicit provider FASTA sources. This writes `source_records.yaml`,
   filtered provider cache FASTAs, and a cache manifest.
7. Materialize unaligned source FASTA bundles from the local provider caches
   and `source_records.yaml`; each bundle must insert the ec86kit Eco1 RT
   sequence as the explicit target FASTA row.
8. Reject `WP_099010551.1` as the target row unless the T301/A301 discrepancy
   is explicitly adjudicated.
9. Run the source-sequence sufficiency gate; reject missing cache roots,
   placeholder accessions, undersized profile bundles, missing source hashes,
   provider hash drift, and exclusions without reasons.
10. Apply the declared filters: query coverage, identity range, length range,
   required RT/retron motifs, and excluded RT families.
11. Align proteins with the declared MAFFT command from the source contract
   through `dnadesign.aligner.msa`.
12. Map alignment columns back to `residue_map.parquet` through canonical Eco1
   positions, not raw PDB residue ids.
13. Compute conservation using non-gap rows as the denominator.
14. Emit `conservation_profile.parquet` only after every row has source hashes,
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

### Provider-Source Materializer

The provider-source materializer is:

```text
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/source_sequences/provider_sources/
```

It consumes the hash-pinned Mestre S1 roster table, derives declared NCBI and
BV-BRC provider accessions from the selected source groups, resolves provider
identity through `sequence_providers[*].accession_patterns` in
`conservation-sources.yaml`, and writes explicit provider FASTA source files:

```text
outputs/thread/eco1_rt_conservative_v1/conservation_provider_sources/ncbi_protein_efetch.fasta
outputs/thread/eco1_rt_conservative_v1/conservation_provider_sources/bv_brc_feature_protein_fasta.fasta
outputs/thread/eco1_rt_conservative_v1/conservation_provider_sources/provider_source_manifest.yaml
```

If a declared provider does not return requested records, those records may
only be carried forward through an explicit failure ledger:

```text
outputs/thread/eco1_rt_conservative_v1/conservation_provider_sources/provider_source_failures.yaml
```

Current local real-data counts:

```text
ncbi_protein_efetch requested 350, returned 350
bv_brc_feature_protein_fasta requested 1577, returned 1464, unresolved 113
```

Command shape:

```bash
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.materialization.source_sequences.provider_sources \
  --repo-root . \
  --roster-table <mestre-s1-roster.xlsx> \
  --write-unresolved-ledger
```

### Roster-Cache Materializer

The roster-cache materializer is:

```text
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/source_sequences/roster_cache/
```

It consumes the Mestre S1 roster table plus explicit provider FASTA sources:

```text
<roster-table>.csv|tsv|xlsx
<provider-source-root>/ncbi_protein_efetch.fasta
<provider-source-root>/bv_brc_feature_protein_fasta.fasta
```

Roster tables may carry optional `source_cache_status` and
`exclusion_reason` columns. Rows default to `included`; rows marked
`excluded` must include a reason and do not require a provider FASTA sequence.
Provider accession shapes are not hard-coded in roster-cache; they are compiled
from the checked-in conservation source contract and reused by the sufficiency
gate.

By default it requires the roster-table hash to match
`conservation-sources.yaml`. Test fixtures may use
`--allow-uncontracted-roster-hash`, but real study data should not. It writes
the local source cache:

```text
outputs/thread/eco1_rt_conservative_v1/conservation_source_cache/source_records.yaml
outputs/thread/eco1_rt_conservative_v1/conservation_source_cache/provider_caches/
outputs/thread/eco1_rt_conservative_v1/conservation_source_cache/source_cache_manifest.yaml
```

Command shape:

```bash
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.materialization.source_sequences.roster_cache \
  --repo-root . \
  --roster-table <mestre-s1-roster.csv-or-xlsx> \
  --provider-source-root <provider-fasta-source-root> \
  --provider-failure-ledger <provider-source-root>/provider_source_failures.yaml
```

The materializer does not perform live NCBI or BV-BRC network retrieval. It
ingests explicit provider FASTA source files so provider drift remains visible.

### Source-Sequence Bundle Materializer

The source-sequence bundle materializer is:

```text
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/source_sequences/
```

It consumes explicit local source caches produced by the roster-cache layer:

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

This command must pass before MAFFT. The current local real-data source bundle
passes with:

```text
broad_retron_rt included 1814, excluded 114
eco1_like_retron_rt included 46, excluded 1
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

The next data slice is `conservation-alignment-bundle-v1`: run the explicit
source FASTA bundles through the public `dnadesign.aligner.msa` API to create
aligned FASTA bundle manifests, then run the conservation materializer and
confirm Phase 1 advances to the `mask_set.yaml` blocker only.

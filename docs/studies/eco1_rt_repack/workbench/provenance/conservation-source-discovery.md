---
doc_id: study-eco1-rt-repack-conservation-source-discovery
surface: study-provenance-note
study_id: eco1_rt_repack
owner: dnadesign-maintainers
last_verified: 2026-06-20
---

## Conservation Source Discovery

This note records source-discovery evidence for the first Eco1 RT conservation
profile. It is not a materialized MSA, and it must not satisfy
`conservation_profile.parquet`.

### Decision

There are sufficient method and source priors to proceed with a
`conservation-source-contract-v1` slice before any alignment materializer.

The follow-on source contract now declares source authority, target-sequence
identity, sequence-provider policy, filtering policy, alignment command policy,
and hash closure. It does not generate `conservation_profile.parquet`; that
profile remains a later materialized artifact.

### Local Source Inventory

The local paper cache at `../resources/retron-prior-papers/` contains primary
method and retron-context PDFs, but no checked-in FASTA, A3M, Stockholm,
aligned MSA, HMM, or supplementary Excel/ZIP files. Supplementary files must be
downloaded or fetched as explicit source artifacts before materialization.

### Method Prior

Tao et al. 2026 is the direct method prior for the conservation mask. Their RT
redesign workflow computes residue conservation from an MSA of homologs found
by querying UniRef50 with the parent RT sequence. Residues are fixed when the
parent amino acid is the plurality residue and its frequency passes the
declared threshold. Their tested conservation thresholds were 0.25 and 0.50.

Eco1 should keep the same rule shape:

```text
fixed_by_conservation =
  wt_aa_is_plurality_aa
  AND wt_frequency >= conservation_threshold
```

The study profile currently declares `conservation_threshold: 0.25` and two
profile ids:

```text
broad_retron_rt
eco1_like_retron_rt
```

### Best Broad Source Prior

Mestre et al. 2020 is the strongest source prior for a retron RT sequence
roster. The article analyzes a 1928-entry retron/retron-like RT dataset and
uses RT0-RT7 alignments to define retron RT clades. The OUP supplement exposes
`Supplementary_Table S1_R1.xlsx` and `Supplementary FiLE S1_RT_Tree.newick.nwk`.

Transient inspection on 2026-06-20 found:

| File | Role | SHA-256 |
| --- | --- | --- |
| `gkaa1149_supplemental_files.zip` | OUP supplemental archive | `4b8882485ebaf03b49d071a6b91f0fe92e20127e70e810b7fe39a517c0c6b9c9` |
| `Supplementary_Table S1_R1.xlsx` | 1928-entry RT roster | `e85635f98ae07808e49b0042096313cd5b8422cdba62d3b200344f10af129aee` |
| `Supplementary FiLE S1_RT_Tree.newick.nwk` | RT tree | `7a9a11296dafa3fff7137bfe51cdd1f1365aa53dc87457400b105e2f7d0134ce` |

Useful observed roster facts:

| Subset | Count | Notes |
| --- | ---: | --- |
| All Mestre S1 RT records | 1928 | Candidate `broad_retron_rt` roster before filtering. |
| RT clade 9 records | 324 | Broad clade containing Eco1-like type II-A3 records. |
| Type II-A3 records | 59 | Candidate `eco1_like_retron_rt` roster. |
| Type II-A3, cluster `42_1` records | 47 | Tighter Eco1-like source candidate. |

The Eco1/Ec86 row is:

```text
Node: 1550
RT/Clade: 9
Retron subtype: II-A3
msr/msd family: IIA3 (Proteobacteria)
Cluster/domain: 42_1
Accession: WP_099010551.1
Retron name: Retron-Eco1 (Ec86)
Species/strain: Escherichia coli
```

Mestre S1 is therefore appropriate as an accession-roster authority, but it is
not by itself a materialized protein sequence alignment.

### Sequence Provider Feasibility

The source contract can avoid bespoke scraping:

- NCBI E-utilities can fetch `WP_*` protein accessions as FASTA.
- BV-BRC can fetch `fig|...` Mestre/PATRIC feature ids as protein FASTA using
  `Accept: application/protein+fasta` against the `genome_feature` API.

Observed examples:

```text
NCBI protein: WP_099010551.1 -> 320 aa Eco1/Ec86 RT-like sequence
BV-BRC feature: fig|1343738.3.peg.2232 -> Retron-Vch3/Vc137 RT sequence
```

The contract should treat these as provider policies, not hard-coded fallback
behavior. If a provider cannot resolve an accession, the row should fail or be
explicitly excluded with a reason.

### Target Sequence Caution

The current Eco1 study structure authority is keyed to the ec86kit reference
sequence hash:

```text
sha256:429a9c9894501e04f48803b96307cea45955f63b85f1461dc25c017e94b7eaeb
```

Direct NCBI fetch of `WP_099010551.1` produced a 320-aa sequence with hash:

```text
sha256:49220a2d5627a561264eca48027a7e943fe409d702e39799074120a63171a7db
```

The two sequences differ at canonical position 301:

```text
ec86kit reference: T301
NCBI WP_099010551.1: A301
```

This is a hard source-contract issue. The conservation materializer must map
alignment columns to the ec86kit target sequence used by
`residue_map.parquet`, not silently use the NCBI Eco1 row as the target
authority. Acceptable resolutions include an explicit ec86kit target FASTA row,
a declared target-row substitution before alignment, or a rejected source
contract until the sequence discrepancy is adjudicated.

### Secondary Priors

| Source | Use | Not enough for |
| --- | --- | --- |
| Simon et al. 2019 | Small curated retron RT alignment and RT1-RT7/X/Y motif sanity checks. | Primary broad conservation profile. |
| Wang et al. 2022 | Ec86 structural/motif sanity checks and contact/motif residue context. | Broad conservation scoring. |
| Khan et al. 2025 | Functional retron roster, RT phylogeny coverage, and accession/ncRNA context. | Direct conservation profile without sequence alignment. |
| Crawford et al. 2024 | Eco1 ncRNA design constraints and functional-context guardrails. | RT protein MSA source. |

### Recommended Source Contract

`conservation-source-contract-v1` should declare two source groups:

| Profile id | Roster authority | Sequence providers | Purpose |
| --- | --- | --- | --- |
| `broad_retron_rt` | Mestre S1, all 1928 records after filters | NCBI E-utilities for `WP_*`; BV-BRC protein FASTA for `fig|*`; explicit reject/exclude for unresolved ids | Protect broadly conserved retron RT positions. |
| `eco1_like_retron_rt` | Mestre S1 type II-A3 cluster `42_1`, or type II-A3 if support is too low | Same providers | Protect Eco1-like recognition/scaffold positions. |

Minimum contract fields:

```text
profile_id
roster_source_ref
roster_source_sha256
target_sequence_hash
target_sequence_policy
accession_field
allowed_sequence_providers
provider_resolution_policy
filter_policy_id
min_query_coverage
min_non_gap_count
identity_range
length_range_aa
required_motifs
excluded_families
alignment_tool
alignment_command
gap_denominator_policy
threshold
plurality_rule
```

Suggested first filters:

```text
query_coverage >= 0.70
identity_range roughly 0.20-0.90
length_range_aa roughly 250-450 unless a retained fusion is declared
required_motifs include RT catalytic DD/YADD-like region, retron X NAXXH-like
motif, and retron Y VTG-like motif
exclude obvious group II intron RTs, DGR RTs, fragments, and unresolved long
fusions unless explicitly retained
```

### Implemented Follow-On Contract

`conservation-source-contract-v1` is recorded at:

```text
docs/studies/eco1_rt_repack/workbench/provenance/conservation-sources.yaml
```

The human reproduction method is recorded at:

```text
docs/studies/eco1_rt_repack/contexts/msa-method.md
```

The conservation-profile materializer now exists as study-owned code, but it
requires explicit aligned FASTA inputs for `broad_retron_rt` and
`eco1_like_retron_rt`. Phase 1 must continue to fail on
`conservation_profile_not_materialized` until a real
`conservation_profile.parquet` is generated from those declared sources. It
must not let review-figure alignments, prose, missing provider rows, or the
mismatched NCBI Eco1 target row imply designability.

### External Source Links

- Tao et al. 2026:
  <https://www.nature.com/articles/s41587-026-03149-6>
- Redesigned prime-editor RT code:
  <https://github.com/Allentaoyz/Redesigned_prime_editor_RTs>
- Mestre et al. 2020:
  <https://academic.oup.com/nar/article/48/22/12632/6020195>
- Simon et al. 2019:
  <https://academic.oup.com/nar/article/47/21/11007/5584520>
- Khan et al. 2025:
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC11911249/>
- BV-BRC API:
  <https://github.com/BV-BRC/BV-BRC-API>

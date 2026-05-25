## V1 Construct Subject Scope

- Last verified: 2026-05-25
- Owner: dnadesign-maintainers

The v1 construct-subject universe is sequence-authority gated. Every promoted
row must carry explicit `construct_subject__lnrna_sequence` and
`construct_subject__rt_cds_sequence` before Construct realizes the normalized
dual-cassette context.

| Construct subject family | Status | Rule |
| --- | --- | --- |
| GenBank-authorized retrons | active | Promote the 36 checked-in GenBank records with explicit lnRNA and RT CDS authority. |
| RT-CDS in silico DMS | active | Generate exhaustive sense-codon amino-acid substitutions through the public Permuter API using top E. coli codons. |
| Crawford Eco1 lnRNA/MSD source sequences | active | Promote the union of source design-reference and abundance-observation lnRNA sequences with fixed WT Eco1 RT when they pass Eco1 forward k-mer orientation QC. |
| Khan cross-retron rows | blocked until RT CDS authority | Promote only rows with explicit source RT CDS DNA sequence; do not treat RT accessions or RT-DNA product sequences as RT CDS. |
| Future source datasets | planned | Add a source-owned resolver that emits the same construct-subject fields and issues before Construct execution. |

The inclusion rule is:

```text
include only if it helps test or diversify programmable multicopy ssDNA production in the synthetic dual-cassette context
```

`lab-anchor` here is source-history language. Construct projection still binds
construct subject rows into the `lnrna` and `rt_cds` slots.

Crawford inclusion is sequence-authority preserving rather than flank-anchor
gated. Many abundance-bearing Crawford variants can legitimately alter the
declared MSD or nearby short flanks, so missing exact MSD substrings or local
source flank anchors are recorded as QC annotations, not automatic exclusion.
The promotion gate instead requires a valid DNA4 lnRNA sequence, Eco1-like
forward k-mer similarity, and no stronger reverse-complement match. Promoted
rows are annotated as projections into the dnadesign synthetic context rather
than exact Crawford expression context recreations, and their source A1/A2
geometry is not assumed to match the dnadesign A1/A2=20 convention.

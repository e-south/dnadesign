## V1 Construct Subject Scope

- Last verified: 2026-05-26
- Owner: dnadesign-maintainers

The v1 construct-subject universe is sequence-authority gated. Every promoted
row must carry explicit `construct_subject__lnrna_sequence` and
`construct_subject__rt_cds_sequence` before Construct realizes the normalized
dual-cassette context.

| Construct subject family | Status | Rule |
| --- | --- | --- |
| GenBank-authorized retrons | active | Promote the 36 checked-in GenBank records as one first-class source cohort with explicit lnRNA and RT CDS authority. |
| RT-CDS in silico DMS | active | Generate exhaustive sense-codon amino-acid substitutions through the public Permuter API using top E. coli codons. |
| Crawford Eco1 lnRNA/MSD source sequences | active | Promote abundance-affiliated source lnRNA sequences with fixed WT Eco1 RT when they pass Eco1 forward k-mer orientation QC. |
| Khan cross-retron rows | active with representability issues | Promote rows with explicit source ncRNA sequence, explicit source RT CDS DNA sequence, translation-exact RT CDS validation, an affiliated RT-DNA abundance prior, and fit inside the current 2,000 bp Construct lane. |
| Compiler-generated MSD lnRNA variants | active fixture pool | Compile the YIU-compatible Snapback cap x scar-nick stem-base primitive pool, reverse-complement the MSD product into the template lnRNA, and pair with fixed WT Eco1 RT. |
| Future source datasets | planned | Add a source-owned resolver that emits the same construct-subject fields and issues before Construct execution. |

The inclusion rule is:

```text
include only if it helps test or diversify programmable multicopy ssDNA production in the synthetic dual-cassette context
```

`lab-anchor` here is source-history language. Construct projection still binds
construct subject rows into the `lnrna` and `rt_cds` slots.

Crawford inclusion is abundance-affiliated and sequence-authority preserving
rather than flank-anchor gated. Many abundance-bearing Crawford variants can
legitimately alter the declared MSD or nearby short flanks, so missing exact MSD
substrings or local source flank anchors are recorded as QC annotations, not
automatic exclusion. The promotion gate requires an abundance-observation row, a
valid DNA4 lnRNA sequence, Eco1-like forward k-mer similarity, and no stronger
reverse-complement match. Design-reference-only sequences remain source
provenance and issue records. Promoted rows are annotated as projections into
the dnadesign synthetic context rather than exact Crawford expression context
recreations, and their source A1/A2 geometry is not assumed to match the
dnadesign A1/A2=20 convention.

Compiler-generated MSD rows are study-owned sequence/design references. They
enter the same Construct subject table as Crawford/Khan/GenBank rows, but they
do not carry abundance data and they do not request pre-Infer concatenation. The
YIU-compatible pool composes five DE033 Snapback cap primitive ranks with
sixteen scar-nick TetO stem-base primitive ranks. The gate is exact: compile a
bounded MSD unit from primitive provenance, require the lnRNA insert to equal
`reverse_complement(msd_product_sequence_5to3)`, match one template MSD span
plus the declared 5-prime and 3-prime flanks, and fail on any orientation,
duplicate, or Construct window violation.

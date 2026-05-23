## V1 Candidate Scope

- Last verified: 2026-05-22
- Owner: dnadesign-maintainers

The first candidate universe is deliberately narrow:

| Candidate family | Status | Rule |
| --- | --- | --- |
| Eco1 WT RT plus retron26-derived lnRNA | planned lab-anchor source row | Include after exact RT and lnRNA sequence authorities are resolved. |
| Eco1 WT RT plus retron43-derived lnRNA | planned lab-anchor source row | Include after exact RT and lnRNA sequence authorities are resolved. |
| Weak/rescue lab-anchor sources | planned | Include only when source-backed records already exist. |
| Compiler variants | planned | Include finite cloning-feasible rows from checked-in retron hairpin compiler primitives. |
| Catalytic-dead RT control | deferred | Include only after the exact RT CDS edit and codon policy are known. |
| Khan high producers | deferred | Keep overlay-only until construct-compatible RT plus lnRNA pairings are explicit. |

The inclusion rule is:

```text
include only if it helps test or diversify programmable multicopy ssDNA production in the synthetic dual-cassette context
```

Do not broaden v1 into a generic retron atlas, broad RT DMS, or broad payload
search.
`lab-anchor` here is source-history language. Construct projection still binds
candidate rows into the `lnrna` and `rt_cds` slots.

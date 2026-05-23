---
doc_id: study-rt-lnrna-sponging-construct-triage-construct-contract
surface: study-context
study_id: rt_lnrna_sponging_construct_triage
owner: dnadesign-maintainers
last_verified: 2026-05-23
---

## Construct Contract

The construct contract is `dual_cassette_rt_lnrna_expression_v1`.

One candidate row means one lnRNA/msr-msd-payload cassette followed by one RT
CDS cassette in a fixed synthetic plasmid regional context:

```text
[plasmid prefix context]
[promoter_lnrna][lnRNA/msr-msd-payload][terminator_lnrna]
[interstitial region]
[promoter_RT][RBS_RT][RT_CDS][terminator_RT]
[plasmid suffix context]
```

### Phase 0 Decision

The v1 path is Construct's public multi-slot assembler. The lnRNA cassette and
RT CDS are separate named slots in one template realization job:

- `lnrna`: lnRNA/msr-msd-payload cassette slot.
- `rt_cds`: Eco1 WT or future RT CDS slot.

Construct owns placement, guard checks, emitted sequence, orientation-aware slot
spans, and `realized_context` sequence views. The study owns candidate-row
semantics, source authority, payload program, overlay linkage, and view names.
Do not precompose lnRNA and RT into one hidden anchor.
When the study needs slot-pooled Infer features, it maps a named Construct slot
into the sequence-view anchor bounds through `output_variants[].anchor_part`;
`lnrna` and `rt_cds` remain slots, not package-level Construct product kinds.

Use Construct projection for the study-owned candidate-to-slot mapping,
Construct realization for the runtime sequence-emission step, and construct
context view materialization for the USR sequence-view rows written after
realization.
Candidate pairing ids remain study identity. USR base row ids remain canonical
sequence ids; candidate ids are carried through study overlays and labels when
temporary candidate rows are passed to Construct.

### Required Before Projection

- Use `../workbench/provenance/genbank-feature-offset-audit.md` as the source
  for plasmid constants, cassette boundaries, Eco1 WT RT CDS, and
  retron26/retron43 lnRNA spans.
- Use
  `../operations/contract/fixtures/construct/construct-projection-manifest.yaml`
  as the study-side projection manifest for named slots and expected spans.
- Materialize construct context views from audited offsets and slot spans, not
  arbitrary padding.
- The 1,600 bp context is the dedicated `1600bp-region.gb` target region,
  contained in pES retron-26 at zero-based half-open vector coordinates
  `[56,1656)`, not the first 1,600 bp of the circular pES-retron-26 record.
- Region-relative retron26 control spans are `lnrna: [130,303)` and
  `rt_cds: [468,1431)`. Longer lnRNA candidates keep the interstitial constant
  and shift the fixed 1,600 bp window by the lnRNA center delta, symmetrically
  trimming the prefix and suffix flanks. Retron43 therefore emits
  `lnrna: [123,310)` and `rt_cds: [475,1438)`.
- Candidates whose required slots cannot fit inside the 1,600 bp view fail
  instead of being arbitrarily truncated.

Construct must emit `realized_context` sequence views. Study role tags,
source-family semantics, candidate roles, and abundance-overlay regimes belong
in view semantics or study fixtures, not in new Construct product kinds.

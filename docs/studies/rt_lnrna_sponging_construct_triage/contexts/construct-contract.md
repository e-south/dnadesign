---
doc_id: study-rt-lnrna-sponging-construct-triage-construct-contract
surface: study-context
study_id: rt_lnrna_sponging_construct_triage
owner: dnadesign-maintainers
last_verified: 2026-05-22
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

### Required Before Projection

- Use `../workbench/provenance/genbank-feature-offset-audit.md` as the source
  for plasmid constants, cassette boundaries, Eco1 WT RT CDS, and
  retron26/retron43 lnRNA spans.
- Use
  `../operations/contract/fixtures/construct/construct-projection-manifest.yaml`
  as the study-side projection manifest for named slots and expected spans.
- Materialize construct context views from audited offsets and slot spans, not
  arbitrary padding.
- Failure behavior for candidates exceeding the 1,600 bp view without
  truncation.

Construct must emit `realized_context` sequence views. Study role tags,
source-family semantics, candidate anchor roles, and abundance-overlay regimes
belong in view semantics or study fixtures, not in new Construct product kinds.

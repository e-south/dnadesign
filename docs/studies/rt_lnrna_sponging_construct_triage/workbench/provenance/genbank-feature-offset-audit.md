## GenBank Feature Offset Audit

- Study: `rt_lnrna_sponging_construct_triage`
- Verified: 2026-05-22
- Parser: `dnadesign.usr.BiopythonGenBankParser`
- Registry: `genbank-source-authority.yaml`
- Coordinate convention: 1-based inclusive spans copied from parsed GenBank
  feature intervals.

### Source Records

| Source id | File | Locus | Length | Topology | SHA-256 |
| --- | --- | --- | ---: | --- | --- |
| `pes_retron_26_vector` | `../pes-retron-26.gb` | `pES-retron-26` | 4956 | circular | `eb76e50add545bdba31b0ad2a6dacf9e85de22862f52ca5d61960e65f4ab5c39` |
| `pes_retron_26_lnrna_a1_a2` | `../pes-retron-26-a1-a2.gb` | `pES-retron-26-a1-a2` | 173 | linear | `f25a3ab588f058562903899164d2b56715d21a6a28d718104e68d2f07cf6e73b` |
| `retron_eco1_rt` | `../retron-eco1-rt.gb` | `retron-Eco1-rt` | 963 | linear | `8b6f9219fffb43af3134c637bf753320811a90974a4952f4693cde1c322437af` |
| `pes_retron_43_vector` | `../pes-retron-43.gb` | `pES-retron-43` | 4970 | circular | `a438d7cb8ef1e846be3ad2a84e5ecc948358b535585757a0487457c6ef7c4a18` |
| `retron_179_orientation_reference` | `../retron-179-a1-a2.gb` | `retron-179-a1-a2` | 178 | linear | `763e9a37481206c6559d60c602eff857d2e056b44518944263d03e3f61db1263` |

### Retron-26 Working Anchor

| Feature | Vector span | a1-a2 span | Strand | Sequence or note |
| --- | ---: | ---: | ---: | --- |
| `a1(20)` | 187-206 | 1-20 | + | `ATTCCGTATGCGCACCCTTA` |
| `msr` | 195-273 | 9-87 | + | Same sequence in vector and subcomponent. |
| `Branched G` | 207 | 21 | + | `G` |
| `P3 Loop` | 237-255 | 51-69 | + | `GGATGTTGGTTCGGCATCC` |
| `RT recognition motif (GUU)` | 245-247 | 59-61 | + | `GTT` |
| `3' Flanking` | 265-281 | 79-95 | + | Last 4 bases derive right base `TCTG`. |
| `msd[tetO]` | 265-338 | 79-152 | - | Contains tetO primary, WT loop, and tetO complement. |
| `tet operator` primary | 282-300 | 96-114 | + | `TCCCTATCAGTGATAGAGA` |
| `WT loop` | 301-304 | 115-118 | + | `GCCT` |
| `tet operator` complement | 305-323 | 119-137 | - | `TCTCTATCACTGATAGGGA` |
| `5' Flanking` | 324-338 | 138-152 | - | First 4 bases derive left base `CCCG`. |
| `a2` | 340-351 | 154-165 | + | `TAAGGGTGCGCA` |
| `a2(20)` | 340-359 | 154-173 | + | `TAAGGGTGCGCATACGGAAT` |

### Retron-43 Failed Anchor

| Feature | Vector span | Strand | Sequence or note |
| --- | ---: | ---: | --- |
| `a1(20)` | 187-206 | + | `ATTCCGTATGCGCACCCTTA` |
| `msr` | 195-273 | + | Matches retron-26 msr. |
| `Branched G` | 207 | + | `G` |
| `P3 Loop` | 237-255 | + | `GGATGTTGGTTCGGCATCC` |
| `RT recognition motif (GUU)` | 245-247 | + | `GTT` |
| `3' Flanking` | 265-281 | + | Last 4 bases derive right base `TCGA`. |
| `G-T mismatch` | 279 | + | First annotated mismatch. |
| `msd[tetO]` | 265-352 | - | Extended relative to retron-26. |
| `tet operator` primary | 282-300 | + | `TCCCTATCAGTGATAGAGA` |
| `loop` | 308-311 | + | `CGGG` |
| `tet operator` complement | 319-337 | - | `TCTCTATCACTGATAGGGA` |
| `5' Flanking` | 338-352 | - | First 4 bases derive left base `CTTG`. |
| `G-T mismatch` | 340 | + | Second annotated mismatch. |
| `a2` | 354-365 | + | `TAAGGGTGCGCA` |
| `a2(20)` | 354-373 | + | `TAAGGGTGCGCATACGGAAT` |

### RT CDS Identity

| Source | RT feature span | CDS span | Translation length | Identity status |
| --- | ---: | ---: | ---: | --- |
| `pes_retron_26_vector` | 525-1487 | 525-1487 | 321 aa | Matches `retron_eco1_rt`. |
| `pes_retron_43_vector` | 539-1501 | 539-1501 | 321 aa | Matches `retron_eco1_rt`. |
| `retron_eco1_rt` | 1-963 | 1-963 | 321 aa | Canonical Eco1 WT RT reference for fixtures. |

The RT CDS sequence and CDS translation are identical across the two vectors and
the standalone `retron-eco1-rt.gb` record.

### Snapback Orientation Reference

`retron-179-a1-a2.gb` is orientation evidence only for Phase 1. It is not a
candidate source in this study slice.

| Feature | Span | Strand | Sequence | Role |
| --- | ---: | ---: | --- | --- |
| `Right Base` | 92-95 | + | `ATTG` | Explicit right stem base annotation. |
| `msd[teto] complement` | 96-114 | + | `TCCCTATCAGTGATAGAGA` | Payload complement placement. |
| `Foldback return` | 115-117 | + | `GAG` | Snapback return. |
| `Foldback` | 115-123 | + | `GAGTCTCTC` | Full foldback geometry. |
| `Cap` | 118-120 | + | `TCT` | Snapback cap. |
| `Foldback stem` | 121-123 | - | `CTC` | Retained stem complement. |
| `msd[teto]` | 124-142 | - | `TCTCTATCACTGATAGGGA` | Payload primary annotation on reverse strand. |
| `Left Base` | 143-146 | - | `CACT` | Explicit left stem base annotation. |

### Construct Projection Gate

Source authority is resolved for the two lab-anchor-derived candidate rows. The
Construct placement ontology is still multi-slot: `lnrna` and `rt_cds` are
separate required slots in one template realization. The public strategy is
`construct_multi_slot_assembly_v1`, recorded in
`operations/contract/fixtures/construct/construct-projection-manifest.yaml`.
The remaining gate is runtime materialization of Construct context views from
those audited offsets, without arbitrary padding or truncation.

## Source Handoff Ledger

- Last verified: 2026-05-25
- Owner: dnadesign-maintainers

### Live Inventories

| Source | dnadesign-data source ID | Path | Rows | Use |
| --- | --- | --- | ---: | --- |
| Khan abundance prior overlay | `khan_2024_retron_census_abundance_prior_overlay_tsv` | `../dnadesign-data/sources/literature/Khan_et_al_2024_retron_census/processed/overlays/abundance_prior_overlay.tsv` | 99 | Numeric RT-DNA abundance prior. |
| Khan RT/ncRNA references | `khan_2024_retron_census_rt_lnrna_references_tsv` | `../dnadesign-data/sources/literature/Khan_et_al_2024_retron_census/processed/references/rt_lnrna_references.tsv` | 171 | Cross-retron provenance bridge. |
| Khan RT/ncRNA sequence authority | `khan_2024_retron_census_rt_lnrna_sequence_authority_tsv` | `../dnadesign-data/sources/literature/Khan_et_al_2024_retron_census/processed/references/rt_lnrna_sequence_authority.tsv` | 171 | Terminal-keyed ncRNA plus explicit RT CDS authority contract. |
| Crawford abundance observations | `crawford_2025_retron_ncrna_ml_eco1_ncrna_abundance_observations_tsv` | `../dnadesign-data/sources/literature/Crawford_et_al_2025_retron_ncRNA_ML/processed/overlays/eco1_ncrna_abundance_observations.tsv` | 4174 | Eco1-local msDNA abundance prior. |
| Crawford lnRNA/MSD references | `crawford_2025_retron_ncrna_ml_eco1_lnrna_msd_designs_tsv` | `../dnadesign-data/sources/literature/Crawford_et_al_2025_retron_ncRNA_ML/processed/references/eco1_lnrna_msd_designs.tsv` | 2578 | Eco1-local sequence/design references. |

### Study-Owned References

| Source | Path | Use |
| --- | --- | --- |
| MSD design registry | `docs/studies/retron_hairpin_design/compiler/catalog/msd_design_registry.yaml` | Retron MSD source labels and cap provenance, including C26/C43 examples. |
| MSD cap source lookup | `docs/studies/retron_hairpin_design/compiler/catalog/msd_cap_sources.yaml` | Retron MSD cap sequences and source labels, including C26/C43 examples. |
| Scar-nick profile panel | `docs/studies/retron_hairpin_design/workbench/design_sets/scar_nick_profile_panel_v1.yaml` | Finite engineered variant rationale and MsdDesignSpec provenance. |
| Compiler MSD lnRNA pool fixture | `docs/studies/rt_lnrna_sponging_construct_triage/operations/contract/fixtures/source-promotions/msd-compiler-pool.yaml` | YIU-compatible study-owned MSD primitive pool that emits 5 x 16 compiler-generated lnRNA variants with reverse-complement insertion into the retron26 template lnRNA. |
| RT-lnRNA variant GenBank metadata | `docs/studies/rt_lnrna_sponging_construct_triage/workbench/provenance/retron-variant-genbank-metadata.yaml` | User-supplied variant comments, Benchling links, and expected RT/lnRNA class. |
| RT-lnRNA variant GenBank catalog | `docs/studies/rt_lnrna_sponging_construct_triage/workbench/provenance/retron-variant-genbank-catalog.yaml` | Parsed lnRNA and RT slot source authority for 36 Construct-representable variants: 35 whole-plasmid GenBank sources plus the BL21 lnRNA-only source paired with Eco1 WT RT. |

### GenBank Source Authorities

| Source | Path | Record | Length | Use |
| --- | --- | --- | ---: | --- |
| pES retron-26 expression vector | `genbank/pes-retron-26.gb` | `pES-retron-26` | 4956 | Working-anchor vector constants, lnRNA offsets, and RT in vector context. |
| Dual-cassette 2,000 bp region | `genbank/2000bp-region.gb` | `2000bp-region` | 2000 | Target context authority; maps to pES retron-26 `[56,2056)` in zero-based half-open coordinates. |
| pES retron-26 a1-a2 | `genbank/pes-retron-26-a1-a2.gb` | `pES-retron-26-a1-a2` | 173 | Working-anchor lnRNA/a1-a2 subcomponent offsets. |
| Eco1 WT RT | `genbank/retron-eco1-rt.gb` | `retron-Eco1-rt` | 963 | Canonical Eco1 WT RT CDS and translation reference. |
| pES retron-43 expression vector | `genbank/pes-retron-43.gb` | `pES-retron-43` | 4970 | Failed-anchor vector constants, lnRNA offsets, and RT in vector context. |
| Retron-179 orientation reference | `genbank/retron-179-a1-a2.gb` | `retron-179-a1-a2` | 178 | Orientation-only left/right base, foldback, and snapback cap geometry reference. |

Parsed offsets and SHA-256 values are pinned in
`genbank-feature-offset-audit.md`. The machine-readable registry is
`genbank-source-authority.yaml`.

### Resolved Authorities

- Exact Eco1 WT RT CDS sequence id:
  `genbank:retron-eco1-rt.gb#ECD_00831`.
- Exact Eco1 WT RT CDS translation provenance:
  `genbank:retron-eco1-rt.gb#CDS`.
- Exact retron26 lnRNA/a1-a2 source:
  `genbank:pes-retron-26-a1-a2.gb#a1-a2`.
- Exact retron43 lnRNA cassette source:
  `genbank:pes-retron-43.gb#a1-a2`.
- Exact pES retron-26 and pES retron-43 vector constants in source records.
- Exact 2,000 bp target context:
  `genbank:2000bp-region.gb#record`, contained in pES retron-26 at `[56,2056)`.
- 36 GenBank-authorized RT-lnRNA construct subjects resolve to explicit lnRNA
  and RT CDS slot sequences.
- 4,148 abundance-affiliated Crawford Eco1-local source lnRNA sequences pass
  DNA4 validation, Eco1 forward k-mer orientation QC, and reverse-complement
  rejection, then project with fixed WT Eco1 RT. The 18 design-reference-only
  sequences are retained as source provenance and issue records. Exact MSD and
  short-flank matches are retained as QC annotations because source variants
  can intentionally alter those regions. These rows are annotated as
  dnadesign-context projections, not exact Crawford expression context
  recreations.
- 129 Khan terminal-keyed RT-lnRNA rows pass explicit source ncRNA, explicit RT
  CDS DNA, translation-exact RT CDS validation, and the current
  lnRNA-centered 2,000 bp construct-window preflight.
- 80 compiler-generated MSD lnRNA fixture rows compile from the YIU-compatible
  5 x 16 Snapback cap and scar-nick stem-base primitive pool, insert the
  reverse complement of the 5-prime-to-3-prime MSD product into the retron26
  lnRNA template after exact flank checks, and pair with fixed Eco1 WT RT.
- 6,080 RT-CDS in silico DMS construct subjects are generated through the
  public `dnadesign.permuter` coding-DNA DMS API.

### Remaining Blockers

- Khan source rows outside the current Construct window. The sequence-authority
  table resolves 169 RT CDS sequences, but 40 otherwise sequence-authorized rows
  exceed the current lnRNA-centered 2,000 bp geometry.
  Under the same placement policy, a 2.1 kb context would still block 33 rows,
  2.2 kb would block 15, 2.3 kb would block 5, 2.5 kb would block 1, and
  2.626 kb would block 0. The checked-in lane remains fixed at 2,000 bp.
- Evo2 Infer sidecars for the six declared Construct sequence views.
- OPAL-ready fixed-size feature table with real sponging labels.

The live consolidated Construct workspace now materializes 10,473 construct
subjects into 20,946 realized 2,000 bp contexts with 62,838 sequence-view
declarations.

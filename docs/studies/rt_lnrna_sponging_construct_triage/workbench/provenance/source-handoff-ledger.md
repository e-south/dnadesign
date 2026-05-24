## Source Handoff Ledger

- Last verified: 2026-05-24
- Owner: dnadesign-maintainers

### Live Inventories

| Source | Path | Rows | Use |
| --- | --- | ---: | --- |
| Khan abundance prior overlay | `../dnadesign-data/sources/literature/Khan_et_al_2024_retron_census/processed/overlays/abundance_prior_overlay.tsv` | 99 | Numeric RT-DNA abundance prior. |
| Khan RT/ncRNA references | `../dnadesign-data/sources/literature/Khan_et_al_2024_retron_census/processed/references/rt_lnrna_references.tsv` | 171 | Cross-retron provenance bridge. |
| Crawford abundance observations | `../dnadesign-data/sources/literature/Crawford_et_al_2025_retron_ncRNA_ML/processed/overlays/eco1_ncrna_abundance_observations.tsv` | 4174 | Eco1-local msDNA abundance prior. |
| Crawford lnRNA/MSD references | `../dnadesign-data/sources/literature/Crawford_et_al_2025_retron_ncRNA_ML/processed/references/eco1_lnrna_msd_designs.tsv` | 2578 | Eco1-local sequence/design references. |

### Study-Owned References

| Source | Path | Use |
| --- | --- | --- |
| MSD design registry | `docs/studies/retron_hairpin_design/compiler/catalog/msd_design_registry.yaml` | C26/C43 source labels and cap provenance. |
| MSD cap source lookup | `docs/studies/retron_hairpin_design/compiler/catalog/msd_cap_sources.yaml` | C26/C43 cap sequences and source labels. |
| Scar-nick profile panel | `docs/studies/retron_hairpin_design/workbench/design_sets/scar_nick_profile_panel_v1.yaml` | Finite engineered variant rationale and MsdDesignSpec provenance. |
| RT-lnRNA variant GenBank metadata | `docs/studies/rt_lnrna_sponging_construct_triage/workbench/provenance/retron-variant-genbank-metadata.yaml` | User-supplied variant comments, Benchling links, and expected RT/lnRNA class. |
| RT-lnRNA variant GenBank catalog | `docs/studies/rt_lnrna_sponging_construct_triage/workbench/provenance/retron-variant-genbank-catalog.yaml` | Parsed lnRNA and RT slot source authority for 36 Construct-representable variants: 35 whole-plasmid GenBank sources plus the BL21 lnRNA-only source paired with Eco1 WT RT. |

### GenBank Source Authorities

| Source | Path | Record | Length | Use |
| --- | --- | --- | ---: | --- |
| pES retron-26 expression vector | `genbank/pes-retron-26.gb` | `pES-retron-26` | 4956 | Working-anchor vector constants, lnRNA offsets, and RT in vector context. |
| Dual-cassette 1,600 bp region | `genbank/1600bp-region.gb` | `1600bp-region` | 1600 | Target context authority; maps to pES retron-26 `[56,1656)` in zero-based half-open coordinates. |
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
- Exact 1,600 bp target context:
  `genbank:1600bp-region.gb#record`, contained in pES retron-26 at `[56,1656)`.

### Remaining Blockers

- Construct context view materialization from the audited offsets and the
  checked multi-slot projection manifest.
- OPAL-ready fixed-size feature table with real sponging labels.

The two lab-anchor-derived candidate rows are source-authority resolved and
construct-representable under `construct_multi_slot_assembly_v1`, but
materialized context views have not been written yet.

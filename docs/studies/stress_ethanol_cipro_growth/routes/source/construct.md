---
doc_id: study-stress-ethanol-cipro-growth-route-source-construct
surface: study-route-detail
study_id: stress_ethanol_cipro_growth
owner: dnadesign-maintainers
last_verified: 2026-05-18
parent_route: ../README.md
type: route
plane: data-plane
owner_boundary: usr-plus-construct
surface_role: consolidation
current_state: complete
entry_artifact: densegen_and_reference_source_records
exit_artifact: anchor_core60_and_context_datasets
---

## Construct Anchor/Context Refresh Route Detail

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-18

Parent router: [README.md](../README.md).

### Construct anchor/context refresh

- Type: `route`
- Plane: `data-plane`
- Surface role: `consolidation`
- Owner-boundary: `usr` plus `construct`
- Current state: `complete`
- Entry artifact: `densegen_prom_eth_cip_source`,
  `usr_promoter_references`, and `usr_sfxi_pdual10_densegen_promoters`
- Exit artifact: `construct_prom_eth_cip_reference_core60`,
  `construct_prom_eth_cip_reference_contexts`, `usr_prom_eth_cip_anchor`,
  and `construct_prom_eth_cip_context`
- Primary workspace: `src/dnadesign/construct/workspaces/study_stress_ethanol_cipro_pdual10`
- First command: `uv run construct workspace run-project --workspace src/dnadesign/construct/workspaces/study_stress_ethanol_cipro_pdual10 --project reference_core60 --dry-run --format json`
- Route note: Construct owns the reference-view and context-refresh lineage.
  For native regulator audit details, use the checked-in status note and
  LatentDNA route detail instead of extending this map.

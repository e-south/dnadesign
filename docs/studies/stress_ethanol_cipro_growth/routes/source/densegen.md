## DenseGen EDA Route Detail

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-18

Parent router: [README.md](../README.md).

- Type: `route`
- Plane: `data-plane`
- Surface role: `producer`
- Owner-boundary: `densegen`
- Current state: `attention`; source rows are ready, but operator-visible plot
  artifacts are stale in the current snapshot
- Entry artifact: `densegen_prom_eth_cip_source`
- Exit artifact: `evidence.analysis_surfaces.densegen` plus
  `outputs/plots/current_inventory.json`
- Primary doc/workspace: `src/dnadesign/densegen/workspaces/study_stress_ethanol_cipro/README.md`
- First command: `uv run dense plot -c src/dnadesign/densegen/workspaces/study_stress_ethanol_cipro/config.yaml`
- Route note: DenseGen owns the producer-side plot taxonomy, current inventory,
  freshness, and notebook visibility contract for this surface.

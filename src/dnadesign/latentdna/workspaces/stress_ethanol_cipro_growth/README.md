# stress_ethanol_cipro_growth latentdna workspace

This workspace holds the LatentDNA comparison surfaces for the active stress / ethanol / ciprofloxacin promoter study. It compares candidate Evo2 spaces to help choose a downstream `X` before assays.

- Workspace id: `stress_ethanol_cipro_growth`
- Study binding: `docs/studies/stress_ethanol_cipro_growth/latentdna_binding.yaml`
- Snapshot artifact: `outputs/status/workspace_snapshot.json`
- Gate: `representation_health_summary`
- Primary review path: `dataset_overview`, `design_structure_summary`, `sigma35_ordinal_audit`, `context_robustness_summary`, `candidate_decision_frontier`
- Companion visuals: `sigma35_stress_margin_gallery`, `context_pair_summary`
- Appendix support: `sigma35_centroid_distance_gallery`
- Appendix surfaces: `design_centroid_margin_gallery`, `reference_alignment_summary`, `representation_scree_diagnostic`, `appendix_umap_gallery`
- UMAP role: appendix orientation only

Common commands:

1. `uv run latentdna workspace snapshot --workspace stress_ethanol_cipro_growth --json`
2. `uv run latentdna validate workspace --workspace stress_ethanol_cipro_growth --deep`
3. `uv run latentdna deliverable status representation_health_summary --workspace stress_ethanol_cipro_growth`
4. `uv run latentdna notebook generate latent_geometry_browser --workspace stress_ethanol_cipro_growth`

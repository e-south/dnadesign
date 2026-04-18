# stress_ethanol_cipro_growth latentdna workspace

This workspace is the study-bound LatentDNA scaffold for the active promoter study. Study semantics enter through `latentdna_binding.yaml` and the published `workspace_snapshot.json`. The workspace stays comparison-first and reference-first.

- Workspace id: `stress_ethanol_cipro_growth`
- Study binding: `docs/studies/stress_ethanol_cipro_growth/latentdna_binding.yaml`
- Snapshot artifact: `outputs/status/workspace_snapshot.json`
- Primary question: compare candidate pooled Evo2 representations and surface evidence for downstream human choice of `X`
- Browser role: plot-first review notebook with secondary geometry and comparison audit tabs
- UMAP role: appendix context only

Operator path:

1. `uv run latentdna workspace snapshot --workspace stress_ethanol_cipro_growth --json`
2. `uv run latentdna validate workspace --workspace stress_ethanol_cipro_growth --deep`
3. `uv run latentdna deliverable status reference_margin_analysis --workspace stress_ethanol_cipro_growth`
4. `uv run latentdna notebook generate latent_geometry_browser --workspace stress_ethanol_cipro_growth`

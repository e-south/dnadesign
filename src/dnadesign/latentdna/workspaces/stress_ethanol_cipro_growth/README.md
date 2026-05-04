# stress_ethanol_cipro_growth latentdna workspace

This workspace holds the LatentDNA comparison surfaces for the active stress / ethanol / ciprofloxacin promoter study. It compares candidate Evo2 spaces to help choose a downstream `X` before assays.

- Workspace id: `stress_ethanol_cipro_growth`
- Study binding: `docs/studies/stress_ethanol_cipro_growth/latentdna_binding.yaml`
- Snapshot artifact: `outputs/status/workspace_snapshot.json`
- Gate: `representation_health_summary`
- Active feature source contract: canonical Infer feature sidecars joined to USR sequence-view and view-semantics sidecars
- Available active geometry: 7B construct-insert `seq_mean` and 7B forward 1 kb context `anchor_mean`
- Available diagnostic reference geometry: 7B reference core60 `analysis_window`
  plus 7B reference forward/reverse-complement context `seq_mean` and
  `anchor_mean`
- Planned main geometry: forward 1 kb context `seq_mean` and
  reverse-complement 1 kb context `seq_mean` and `anchor_mean` after the main
  merged handoff Infer batches complete
- Reference metadata sources: `usr_promoter_references`, `construct_prom_eth_cip_reference_core60`, and `construct_prom_eth_cip_reference_contexts`
- Sigma-35 inventory: source-backed from DenseGen plan tokens, DenseGen fixed-element details, USR `seq_annot` `-35` features, or Construct retained-feature bounds; missing embedding rows mean missing Infer vectors, not filtered Sigma-35 categories
- Primary review path: `dataset_overview`, `design_structure_summary`, `sigma35_ordinal_audit`, `context_robustness_summary`, `candidate_decision_frontier`
- Companion visuals: `sigma35_stress_margin_gallery`, `context_pair_summary`
- Appendix support: `sigma35_centroid_distance_gallery`
- Appendix surfaces: `design_centroid_margin_gallery`, `reference_alignment_summary`, `representation_scree_diagnostic`, `appendix_umap_gallery`
- Current regenerated plots: primary, companion, and appendix surfaces,
  including `appendix_umap_gallery`, are materialized from current code/config.
- Reference-only probe: `outputs/plots/reference_strength_probe/` contains
  exploratory PCA scatter plots and Spearman summaries for the completed
  reference core60/context sidecars.
- UMAP role: appendix orientation only
- Browser geometry default: candidate grid with mixed-length anchor-source
  insert means and 1 kb context anchor-mean panels

Common commands:

1. `uv run latentdna workspace snapshot --workspace stress_ethanol_cipro_growth --json`
2. `uv run latentdna validate workspace --workspace stress_ethanol_cipro_growth --deep`
3. `uv run latentdna inspect source anchor_7b_seq_mean_features --workspace stress_ethanol_cipro_growth --json`
4. `uv run latentdna inspect source full_context_7b_forward_anchor_mean_features --workspace stress_ethanol_cipro_growth --json`
5. `uv run latentdna inspect source reference_context_7b_forward_anchor_mean_features --workspace stress_ethanol_cipro_growth --json`
6. `uv run latentdna deliverable status representation_health_summary --workspace stress_ethanol_cipro_growth`
7. `uv run latentdna deliverable status appendix_geometry_review --workspace stress_ethanol_cipro_growth`
8. `uv run latentdna notebook generate latent_geometry_browser --workspace stress_ethanol_cipro_growth`

Existing materialized view/plot artifacts may predate the sidecar-backed source
contract. Treat source-contract drift in deep validation as a required refresh
signal before using regenerated plots as current evidence.

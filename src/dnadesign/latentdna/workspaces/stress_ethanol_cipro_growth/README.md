# stress_ethanol_cipro_growth latentdna workspace

This workspace is the study-bound `latentdna` scaffold for `docs/studies/stress_ethanol_cipro_growth`.

- Sources are already bound to the checked-in promoter handoff datasets.
- `study_binding` points back to the study record so validation and status surfaces can reconcile readiness.
- The canonical LatentDNA artifact root is `outputs/`.
- `outputs/latentdna/` is a rejected legacy layout. Use `latentdna workspace refresh --target legacy` to remove it without touching upstream `usr/datasets`.
- The checked-in atlas path now preserves `promoter_wt_core` inside the sampled projection scopes so required control labels survive end-to-end atlas rendering.
- Full-view PCA and aligned export lanes can require `--allow-memory-overage` on a 16 GiB workstation.
- The checked-in cluster-correspondence path now runs on aligned reduced views plus explicit neighbor graphs; raw aligned Leiden fits on the 157k-row 20B matrices are intentionally not the operator path.
- The saved outputs currently support pooled representation analysis across
  `z20_60`, `z20_1k_anchor`, `z20_1k_seq`, pooled logits, and scalar QC; they
  do not support token-level hidden-state or per-position logit claims inside
  the 1 kb construct.

Primary next steps:

1. `uv run latentdna workspace refresh --workspace src/dnadesign/latentdna/workspaces/stress_ethanol_cipro_growth --target legacy --dry-run`
2. `uv run latentdna validate workspace --workspace src/dnadesign/latentdna/workspaces/stress_ethanol_cipro_growth --deep`
3. Open `src/dnadesign/latentdna/docs/workflows/promoter-study-latent-atlas.md` for the current interpretation boundary and planned figure pack.
4. Review `src/dnadesign/latentdna/workspaces/stress_ethanol_cipro_growth/outputs/notebooks/browser/controls.json` to confirm which pooled views, scalar tables, and likelihood columns are already joinable in the saved browser state.
5. Treat `z20_60` and `z20_1k_seq` as the primary comparison, keep pooled logits as benchmarks, and use `z20_1k_anchor` or `drag20` only as QC until the reference-alignment and grouped-benchmark plots are added.
6. `uv run latentdna notebook smoke --workspace src/dnadesign/latentdna/workspaces/stress_ethanol_cipro_growth --json`
7. `uv run marimo run src/dnadesign/latentdna/workspaces/stress_ethanol_cipro_growth/outputs/notebooks/browser/notebook.py`

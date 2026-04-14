# Promoter-Study Latent Atlas

**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-13

**Type:** workflow
**Plane:** downstream-tool
**Owner-boundary:** latentdna
**Registry-id:** latentdna.promoter-study.latent-atlas

`latentdna` owns downstream latent analysis artifacts after vector-bearing USR
surfaces already exist. It does not run infer and it does not train OPAL
models.

The active promoter-study workspace is already checked in at
`src/dnadesign/latentdna/workspaces/stress_ethanol_cipro_growth`. Treat this
workflow as the study-bound handoff after the anchor and construct-context USR
datasets already carry the feature columns you want to analyze.

### First tracer-bullet path

```bash
# Reuse one explicit path for every study-bound latentdna command.
export LATENTDNA_WORKSPACE=src/dnadesign/latentdna/workspaces/stress_ethanol_cipro_growth

# Validate the checked-in workspace plus the live study-bound source schema.
uv run latentdna validate workspace \
  --workspace "$LATENTDNA_WORKSPACE" \
  --deep

# Materialize the primary 20B anchor view directly from the checked-in workspace.
uv run latentdna view materialize z20_60 \
  --workspace "$LATENTDNA_WORKSPACE"

# Build the deterministic sampled support used by the atlas recipe.
# The checked-in workspace derives `design_family` from promoter metadata.
uv run latentdna sample build atlas_anchor_sample \
  --workspace "$LATENTDNA_WORKSPACE" \
  --view z20_60 \
  --strategy stratified \
  --group-column design_family \
  --n 20000 \
  --seed 17

# Fit the primary persisted projection on the sampled scope.
uv run latentdna projection fit z20_60 \
  --workspace "$LATENTDNA_WORKSPACE" \
  --sample atlas_anchor_sample \
  --run-id umap_z20_60

# Ask whether the checked-in atlas deliverable is ready before building it.
uv run latentdna deliverable status atlas_2x2_intermediate_main \
  --workspace "$LATENTDNA_WORKSPACE"

# Materialize the study's primary atlas deliverable and browser notebook.
uv run latentdna deliverable run atlas_2x2_intermediate_main \
  --workspace "$LATENTDNA_WORKSPACE"
```

### Current deliverable slices

```bash
# Refresh the primary context-shift deliverable once the paired 20B views are ready.
uv run latentdna deliverable status context_shift_primary \
  --workspace "$LATENTDNA_WORKSPACE"
# Materialize the same context-shift deliverable after status confirms the gap.
uv run latentdna deliverable run context_shift_primary \
  --workspace "$LATENTDNA_WORKSPACE"

# Refresh the WT-neighborhood enrichment surface.
uv run latentdna deliverable status control_neighborhood_enrichment \
  --workspace "$LATENTDNA_WORKSPACE"
# Materialize the enrichment surface after checking deliverable status.
uv run latentdna deliverable run control_neighborhood_enrichment \
  --workspace "$LATENTDNA_WORKSPACE"

# Refresh the current 7B-versus-20B agreement deliverable.
uv run latentdna deliverable status agreement_7b_vs_20b \
  --workspace "$LATENTDNA_WORKSPACE"
# Materialize the agreement deliverable once the status surface is explicit.
uv run latentdna deliverable run agreement_7b_vs_20b \
  --workspace "$LATENTDNA_WORKSPACE"
```

### Structural agreement and Leiden cluster slice

```bash
# Compile reusable neighborhood structure on the aligned 20B support.
uv run latentdna neighbors fit z20_60_knn \
  --workspace "$LATENTDNA_WORKSPACE" \
  --view z20_60 \
  --alignment anchor_ctx_20b \
  --k 15 \
  --backend approximate

# Build the paired challenger neighborhood artifact on the same aligned support.
uv run latentdna neighbors fit z20_1k_anchor_knn \
  --workspace "$LATENTDNA_WORKSPACE" \
  --view z20_1k_anchor \
  --alignment anchor_ctx_20b \
  --k 15 \
  --backend approximate

# Fit the current aligned Leiden cluster sets expected by the study contract.
uv run latentdna cluster fit leiden_z20_60 \
  --workspace "$LATENTDNA_WORKSPACE" \
  --view z20_60 \
  --alignment anchor_ctx_20b \
  --method leiden \
  --resolution 0.6

# Mirror the same Leiden fit for the context-aware 20B view.
uv run latentdna cluster fit leiden_z20_1k_anchor \
  --workspace "$LATENTDNA_WORKSPACE" \
  --view z20_1k_anchor \
  --alignment anchor_ctx_20b \
  --method leiden \
  --resolution 1.0

# Render the checked-in cluster correspondence deliverable from the Leiden outputs.
uv run latentdna deliverable status cluster_correspondence_primary \
  --workspace "$LATENTDNA_WORKSPACE"
# Materialize the correspondence plot after the Leiden inputs are ready.
uv run latentdna deliverable run cluster_correspondence_primary \
  --workspace "$LATENTDNA_WORKSPACE"
```

### Export handoff slice

```bash
# Export the current primary aligned multiblock handoff.
uv run latentdna export matrix x2_primary_20b \
  --workspace "$LATENTDNA_WORKSPACE"

# Export the current 7B challenger ablation bundle.
uv run latentdna export matrix x3_ablation_7b \
  --workspace "$LATENTDNA_WORKSPACE"

# Emit the aligned primary bundle as a flat table when a downstream consumer prefers tabular handoff.
uv run latentdna export table x2_primary_20b \
  --workspace "$LATENTDNA_WORKSPACE"
```

### Operational inventory slice

```bash
# Inspect the persisted artifact inventory and current materialization state.
uv run latentdna inspect artifacts \
  --workspace "$LATENTDNA_WORKSPACE"

# Inspect declared views and explicit cross-view support without recomputing anything.
uv run latentdna inspect views \
  --workspace "$LATENTDNA_WORKSPACE"

# Inspect one compiled alignment artifact and its persisted support summary.
uv run latentdna inspect alignment anchor_ctx_20b \
  --workspace "$LATENTDNA_WORKSPACE"

# Generate the browser notebook when atlas plots already exist under
# `outputs/latentdna/plots`.
uv run latentdna notebook generate browser \
  --workspace "$LATENTDNA_WORKSPACE"

# Open the generated marimo browser from the checked-in workspace output root.
uv run marimo run \
  src/dnadesign/latentdna/workspaces/stress_ethanol_cipro_growth/outputs/latentdna/notebooks/browser.py

# Enumerate persisted artifacts in machine-friendly form.
uv run latentdna runs list \
  --workspace "$LATENTDNA_WORKSPACE" \
  --json
```

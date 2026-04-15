# Promoter-Study Latent Atlas

**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-14

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

All LatentDNA artifacts for this workspace live under `outputs/`.
`outputs/latentdna/` is a rejected legacy layout and should be removed before
running the workflow.

### View taxonomy for the active study

- Primary intermediate views:
  - `z20_60`
  - `z20_1k_anchor`
- Committee intermediate views:
  - `z7_60`
  - `z7_1k_anchor`
- Challenger/QC views:
  - `logits7_60`
  - `logits7_1k_anchor`
  - `z20_1k_seq`
  - `logits20_60`
  - `logits20_1k_anchor`
- `output_layer_mean__*` columns render as pooled logits in LatentDNA docs, UI,
  and plot labels. They are not described as hidden-state embeddings.
- The browser atlas viewer now supports single-view, side-by-side, `2 x 2`
  intermediate, and `2 x 3` model-by-family layouts over persisted projections.

### First tracer-bullet path

```bash
# Reuse one explicit path for every study-bound latentdna command.
export LATENTDNA_WORKSPACE=src/dnadesign/latentdna/workspaces/stress_ethanol_cipro_growth

# Clear only workspace-local LatentDNA outputs when you need to remove stale
# artifacts or the rejected legacy tree. This does not touch upstream
# `usr/datasets`.
uv run latentdna workspace refresh \
  --workspace "$LATENTDNA_WORKSPACE" \
  --target legacy \
  --dry-run

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
  --reference-set promoter_wt_core \
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
# On a 16 GiB workstation this lane may require `--allow-memory-overage`
# because it reduces the full 20B anchor and delta views.
uv run latentdna deliverable status context_shift_primary \
  --workspace "$LATENTDNA_WORKSPACE"
# Materialize the same context-shift deliverable after status confirms the gap.
uv run latentdna deliverable run context_shift_primary \
  --workspace "$LATENTDNA_WORKSPACE" \
  --allow-memory-overage

# Refresh the current PCA scree surface for the primary anchor-only view.
# On a 16 GiB workstation this reduction may require `--allow-memory-overage`.
uv run latentdna deliverable status control_pca_explained_variance_curve \
  --workspace "$LATENTDNA_WORKSPACE"
# Materialize the PCA explained-variance surface after checking deliverable status.
uv run latentdna deliverable run control_pca_explained_variance_curve \
  --workspace "$LATENTDNA_WORKSPACE" \
  --allow-memory-overage

# Refresh the current 7B-versus-20B agreement deliverable.
uv run latentdna deliverable status agreement_7b_vs_20b \
  --workspace "$LATENTDNA_WORKSPACE"
# Materialize the agreement deliverable once the status surface is explicit.
uv run latentdna deliverable run agreement_7b_vs_20b \
  --workspace "$LATENTDNA_WORKSPACE"
```

### Context-audit and browser control-plane slice

```bash
# Materialize the challenger and committee views that the browser atlas viewer can compare.
uv run latentdna view materialize logits7_60 \
  --workspace "$LATENTDNA_WORKSPACE"

uv run latentdna view materialize z20_1k_seq \
  --workspace "$LATENTDNA_WORKSPACE"

uv run latentdna view materialize logits7_1k_anchor \
  --workspace "$LATENTDNA_WORKSPACE"

uv run latentdna view materialize logits20_60 \
  --workspace "$LATENTDNA_WORKSPACE"

uv run latentdna view materialize logits20_1k_anchor \
  --workspace "$LATENTDNA_WORKSPACE"

# Ask whether the geometry switchboard deliverable is missing before building it.
uv run latentdna deliverable status geometry_switchboard_20b \
  --workspace "$LATENTDNA_WORKSPACE"

# Materialize the browser control plane, the persisted multiview projection
# inventory, and the `2 x 3` atlas over 7B and 20B intermediate/logits lanes.
uv run latentdna deliverable run geometry_switchboard_20b \
  --workspace "$LATENTDNA_WORKSPACE"

# Ask whether the explicit 20B context audit has enough artifacts to evaluate delta20.
uv run latentdna deliverable status context_audit_primary_20b \
  --workspace "$LATENTDNA_WORKSPACE"

# Materialize the delta-versus-drag audit surfaces and refresh the browser notebook controls.
uv run latentdna deliverable run context_audit_primary_20b \
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

# Reduce the aligned 20B anchor-only support before structural clustering.
uv run latentdna view reduce z20_60 \
  --workspace "$LATENTDNA_WORKSPACE" \
  --run-id z20_60_anchor_ctx_pca \
  --alignment anchor_ctx_20b \
  --dims 32 \
  --reduced-view-id z20_60_anchor_ctx_pc32

# Mirror the same aligned reduction for the context-aware 20B support.
uv run latentdna view reduce z20_1k_anchor \
  --workspace "$LATENTDNA_WORKSPACE" \
  --run-id z20_1k_anchor_anchor_ctx_pca \
  --alignment anchor_ctx_20b \
  --dims 32 \
  --reduced-view-id z20_1k_anchor_anchor_ctx_pc32

# Build reusable neighbor graphs on the reduced aligned supports.
uv run latentdna neighbors fit leiden_z20_60_knn \
  --workspace "$LATENTDNA_WORKSPACE" \
  --reduced-view z20_60_anchor_ctx_pc32 \
  --k 15 \
  --backend approximate

uv run latentdna neighbors fit leiden_z20_1k_anchor_knn \
  --workspace "$LATENTDNA_WORKSPACE" \
  --reduced-view z20_1k_anchor_anchor_ctx_pc32 \
  --k 15 \
  --backend approximate

# Fit the current aligned Leiden cluster sets from those reduced neighbor graphs.
uv run latentdna cluster fit leiden_z20_60 \
  --workspace "$LATENTDNA_WORKSPACE" \
  --reduced-view z20_60_anchor_ctx_pc32 \
  --method leiden \
  --neighbor-set leiden_z20_60_knn \
  --k 15 \
  --resolution 0.6

uv run latentdna cluster fit leiden_z20_1k_anchor \
  --workspace "$LATENTDNA_WORKSPACE" \
  --reduced-view z20_1k_anchor_anchor_ctx_pc32 \
  --method leiden \
  --neighbor-set leiden_z20_1k_anchor_knn \
  --k 15 \
  --resolution 1.0

# Render the checked-in cluster correspondence deliverable from the reduced-view Leiden outputs.
uv run latentdna deliverable status cluster_correspondence_primary \
  --workspace "$LATENTDNA_WORKSPACE"
# Materialize the correspondence plot after the reduced-view cluster inputs are ready.
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
# The aligned export lane may also need `--allow-memory-overage` on a 16 GiB host.
uv run latentdna export table x2_primary_20b \
  --workspace "$LATENTDNA_WORKSPACE" \
  --allow-memory-overage
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
# `outputs/plots`.
uv run latentdna notebook generate browser \
  --workspace "$LATENTDNA_WORKSPACE"

# The generated notebook now carries a control-plane payload used by the
# atlas viewer and compare tab. `notebook generate` may report `attention` when the
# default deliverable plot is missing, but it still writes the notebook and the
# explicit degraded-state warnings.
cat src/dnadesign/latentdna/workspaces/stress_ethanol_cipro_growth/outputs/notebooks/browser/controls.json

# Fail fast if the generated notebook or its persisted assets are unhealthy.
uv run latentdna notebook smoke \
  --workspace "$LATENTDNA_WORKSPACE" \
  --json

# Open the generated marimo browser from the checked-in workspace output root.
uv run marimo run \
  src/dnadesign/latentdna/workspaces/stress_ethanol_cipro_growth/outputs/notebooks/browser/notebook.py

# Enumerate persisted artifacts in machine-friendly form.
uv run latentdna runs list \
  --workspace "$LATENTDNA_WORKSPACE" \
  --json
```

`latentdna notebook smoke` remains the gate. It now checks that the notebook
file exists, the stored health contract loads, the plot catalog loads, the
default deliverable can render from persisted assets, and the browser
control-plane artifact at `outputs/notebooks/browser/controls.json` is present
and readable.

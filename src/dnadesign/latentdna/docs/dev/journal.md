# Development Journal

**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-17

## Active refactor

LatentDNA now treats the promoter-study workspace as a pre-assay representation-triage surface rather than a reference-first benchmark proxy.

Current checked-in direction:

- publish a sanctioned `workspace_snapshot.json` contract for study-facing status
- keep latent-analysis primitives study-neutral and inject promoter-study semantics through `contexts/latentdna/binding.yaml`
- restrict the active workspace to the canonical eight geometry spaces across `intermediate_embedding` and `output_layer_mean`
- treat UMAP as appendix-only geometry context
- keep plot semantics explicit and sidecar-backed

## Maintainer notes

- Pressure-path evidence should be recorded against the current decision ladder:
  1. `dataset_overview`
  2. `representation_health_summary`
  3. `design_structure_summary`
  4. `sigma35_ordinal_audit`
  5. `context_robustness_summary`
  6. `appendix_umap_gallery`
- When updating workspace artifacts, regenerate from the checked-in workspace config rather than editing outputs by hand.
- Study-family tools must consume the study binding plus workspace snapshot only; they must not import `dnadesign.latentdna.src.*`.

## Latest evidence

- Status/control-plane paths now load workspace config without eagerly validating every plot-semantics sidecar; explicit workspace validation still owns that fail-fast contract.
- `WorkspaceContext` now resolves and stores the canonical `output_root` once instead of re-normalizing it on each access.
- On the April 17, 2026 live `stress_ethanol_cipro_growth` workspace, warmed `workspace_snapshot()` measured 0.18s mean across three runs and warmed `deliverable_status(representation_health_summary)` measured 0.13s mean across three runs after the loader/control-plane split.
- Independent fresh-process `workspace_snapshot()` launches measured 1.75s on the first cold launch and about 1.00s on subsequent launches, so cold-cache variance remains material.

## Next checks

- keep boundary checks covering study-to-tool imports
- keep docs/tests free of removed artifact IDs and forbidden output-root fallbacks
- keep the browser defaults aligned with the canonical geometry inventory and reduced preferred hue set

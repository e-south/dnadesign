# latentdna CLI Contracts

**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-14

### Common flags

- All commands accept:
  - `--workspace <id|path>` where the command needs a workspace
  - `--format text|json|yaml`
  - `--json`
  - `--quiet`
- Mutating commands also accept:
  - `--force`
  - `--dry-run`
- Stochastic mutating commands additionally accept:
  - `--seed`

`--quiet` keeps text mode to one concise success line. `--dry-run` validates the declared target and reports the expected output paths without writing artifacts.

Workspace-specific flags:

- `latentdna workspace init --template <template-id>`
- `latentdna workspace init --from-study-dir <path>`
- `latentdna workspace refresh --target <artifact-dir|legacy|catalog|runs|logs>`
- `latentdna validate workspace --deep`

### Primitive command groups

- `latentdna workspace init`
- `latentdna workspace where`
- `latentdna workspace list`
- `latentdna workspace show`
- `latentdna workspace refresh`
- `latentdna validate workspace`
- `latentdna inspect source`
- `latentdna inspect views`
- `latentdna inspect alignment`
- `latentdna inspect landmarks`
- `latentdna inspect missingness`
- `latentdna inspect artifacts`
- `latentdna snapshot build`
- `latentdna alignment build`
- `latentdna view materialize`
- `latentdna view derive`
- `latentdna view reduce`
- `latentdna view stats`
- `latentdna scalar derive`
- `latentdna sample build`
- `latentdna neighbors fit`
- `latentdna cluster fit`
- `latentdna projection fit`
- `latentdna distance score`
- `latentdna enrich score`
- `latentdna agreement compare`
- `latentdna export matrix`
- `latentdna export table`
- `latentdna plot render`
- `latentdna notebook generate`
- `latentdna notebook smoke`
- `latentdna recipe validate`
- `latentdna recipe run`
- `latentdna deliverable list`
- `latentdna deliverable status`
- `latentdna deliverable run`
- `latentdna runs list`
- `latentdna runs show`
- `latentdna runs prune`
- `latentdna inspect notebook-health`

### Machine output contracts

- Mutating commands emit `latentdna.command_result.v1`.
- Artifact directories carry `latentdna.manifest.v1` manifests.
- `latentdna deliverable status` emits `latentdna.deliverable_status.v1`.
- `latentdna neighbors fit` and `latentdna cluster fit` require exactly one of `--view` or `--reduced-view`; reduced views are already scope-fixed and cannot be combined with `--sample` or `--alignment`.
- `latentdna workspace init --json` emits `latentdna.command_result.v1` with `artifact_kind=workspace`.
- `latentdna notebook generate` may emit `status=attention` when the notebook artifact exists but the default deliverable plot has not been rendered yet.
- `latentdna notebook smoke` exits non-zero when notebook health is `error`.
- `latentdna inspect notebook-health` exits non-zero when stored notebook health is `error`.
- Text mode stays concise; JSON mode is the stable automation surface.

### Implemented artifact families

- `snapshot`: persisted key row ledgers plus `metadata.parquet` companions over declared sources without reloading vector columns
- `view`: source-backed matrices from `column` and `bundle_matrix` sources plus derived `vector_difference`, `normalize`, `aggregate_by_key`, `apply_reducer`, and `concatenate` views
- `alignment_set`: persisted key-support ledgers plus row-index mappings
- `reducer`: persisted PCA state plus fit summaries
- `reduced_view`: low-rank transformed matrices with explicit row support
- `scalar_table`: `vector_norm`, safe `column_expression`, `select_columns`, `rename_columns`, and explicit `join_tables` outputs
- `sample_set`: deterministic plotting scopes, explicit-id selections, and set-algebra unions/intersections over persisted sample artifacts
- `sample build` may also preserve a declared `reference_set` while sampling from a view, so required control or reference rows do not disappear from downstream atlas plots.
- `neighbor_set`: exact or approximate kNN results over explicit scopes
- `cluster_set`: persisted k-means or Leiden assignments over explicit view/sample/alignment scopes plus recorded cluster provenance
- `projection`: UMAP coordinates over explicit scopes
- `distance_set`: landmark distance tables over existing view artifacts
- `enrichment_set`: landmark-neighborhood enrichment tables over configured cohort columns
- `agreement_set`: persisted kNN-overlap rows plus optional cluster-agreement and landmark-neighborhood summary rows
- `export_bundle`: deterministic numeric matrices or aligned tables plus row and feature ledgers, including alignment-backed block projection when the workspace declares it
- `plot`: artifact-driven `projection_scatter`, `projection_grid`, `heatmap`, `distance_scatter`, `xy_scatter`, `distribution`, `curve`, `correspondence_heatmap`, and `agreement_summary` outputs
- `notebook`: generated read-only marimo artifact review apps over persisted outputs, including inline plot-file viewing, a workspace-wide plot browser over `outputs/plots`, and a persisted browser control plane in `outputs/notebooks/<id>/controls.json`

### Plot render modes

- `latentdna plot render <plot-id>` supports two explicit modes.
- Named mode: resolve `plot-id` from `plots.<plot-id>` in the workspace config and render from that declared recipe.
- Inline mode: provide `--kind` plus the required artifact flags such as `--projection`, `--distance`, `--scalar`, `--enrichment`, or `--agreement`.
- Projection plots may also declare `color_column`, `panel_titles`, and optional `label_column` plus `label_values` to keep multi-panel atlases visually comparable.
- Scalar plots now also include `xy_scatter` for joined scalar tables, `curve` for reducer/enrichment summaries, and `correspondence_heatmap` for paired categorical cluster structure.
- Mixing named and inline plot specs in one invocation is rejected.

### Real-study pressure path

- Use `latentdna workspace init --from-study-dir docs/studies/stress_ethanol_cipro_growth` to hydrate the committee template from the checked-in promoter-study record.
- Use `latentdna workspace refresh` to clear workspace-local LatentDNA artifacts or the rejected legacy tree without touching upstream `usr/datasets`.
- Use `latentdna validate workspace --deep` to confirm the declared source keys, view vector columns, cohort columns, landmark selector columns, and study-binding files against the live study data without materializing embedding matrices.
- Use `latentdna deliverable status <deliverable-id>` after recipe or deliverable runs to surface freshness drift from recorded input-path digests rather than only presence/absence checks.
- Canonical runtime artifacts live under `outputs/`. `outputs/latentdna/` is a rejected legacy layout and must be removed rather than shimmed.

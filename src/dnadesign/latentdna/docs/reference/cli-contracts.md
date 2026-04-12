# latentdna CLI Contracts

**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-12

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
- `latentdna validate workspace --deep`

### Primitive command groups

- `latentdna workspace init`
- `latentdna workspace where`
- `latentdna workspace list`
- `latentdna workspace show`
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
- `latentdna recipe validate`
- `latentdna recipe run`
- `latentdna deliverable list`
- `latentdna deliverable status`
- `latentdna deliverable run`
- `latentdna runs list`
- `latentdna runs show`
- `latentdna runs prune`

### Machine output contracts

- Mutating commands emit `latentdna.command_result.v1`.
- Artifact directories carry `latentdna.manifest.v1` manifests.
- `latentdna deliverable status` emits `latentdna.deliverable_status.v1`.
- `latentdna workspace init --json` emits `latentdna.command_result.v1` with `artifact_kind=workspace`.
- Text mode stays concise; JSON mode is the stable automation surface.

### Implemented artifact families

- `snapshot`: persisted key row ledgers plus `metadata.parquet` companions over declared sources without reloading vector columns
- `view`: source-backed matrices from `column` and `bundle_matrix` sources plus derived `vector_difference`, `normalize`, `aggregate_by_key`, `apply_reducer`, and `concatenate` views
- `alignment_set`: persisted key-support ledgers plus row-index mappings
- `reducer`: persisted PCA state plus fit summaries
- `reduced_view`: low-rank transformed matrices with explicit row support
- `scalar_table`: `vector_norm`, safe `column_expression`, `select_columns`, `rename_columns`, and explicit `join_tables` outputs
- `sample_set`: deterministic plotting scopes, explicit-id selections, and set-algebra unions/intersections over persisted sample artifacts
- `neighbor_set`: exact or approximate kNN results over explicit scopes
- `cluster_set`: persisted k-means assignments over explicit view/sample/alignment scopes
- `projection`: UMAP coordinates over explicit scopes
- `distance_set`: landmark distance tables over existing view artifacts
- `enrichment_set`: landmark-neighborhood enrichment tables over configured cohort columns
- `agreement_set`: persisted kNN-overlap rows plus optional cluster-agreement and landmark-neighborhood summary rows
- `export_bundle`: deterministic numeric matrices or aligned tables plus row and feature ledgers, including alignment-backed block projection when the workspace declares it
- `plot`: artifact-driven `projection_scatter`, `projection_grid`, `heatmap`, `distance_scatter`, `distribution`, and `agreement_summary` outputs
- `notebook`: generated read-only marimo artifact review apps over persisted outputs, including inline plot-file viewing plus a workspace-wide plot browser over `outputs/latentdna/plots`

### Plot render modes

- `latentdna plot render <plot-id>` supports two explicit modes.
- Named mode: resolve `plot-id` from `plots.<plot-id>` in the workspace config and render from that declared recipe.
- Inline mode: provide `--kind` plus the required artifact flags such as `--projection`, `--distance`, `--scalar`, `--enrichment`, or `--agreement`.
- Mixing named and inline plot specs in one invocation is rejected.

### Real-study pressure path

- Use `latentdna workspace init --from-study-dir docs/studies/stress_ethanol_cipro_growth` to hydrate the committee template from the checked-in promoter-study record.
- Use `latentdna validate workspace --deep` to confirm the declared source keys, view vector columns, cohort columns, landmark selector columns, and study-binding files against the live study data without materializing embedding matrices.
- Use `latentdna deliverable status <deliverable-id>` after recipe or deliverable runs to surface freshness drift from recorded input-path digests rather than only presence/absence checks.

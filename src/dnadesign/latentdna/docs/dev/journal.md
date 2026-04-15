## latentdna Development Journal

**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-12

This journal tracks `dnadesign.latentdna` implementation progress against the clean-slate build spec.

## 2026-04-09 - Phase 1 Tracer Bullet

### Objective

Prove the first artifact-first downstream loop:

- scaffold a workspace from a packaged template
- validate `latentdna.workspace.v1`
- inspect a source without hidden orchestration
- materialize a source-backed view into canonical matrix form
- build a deterministic sample set
- fit a UMAP projection on the sampled scope
- render a plot from persisted artifacts only

### Implemented in this slice

- New `dnadesign.latentdna` public package surface with `api.py`, `cli.py`, and `contracts.py`
- Internal workspace/config loader with coordinate-space validation for declared vector differences
- Source inspection and read-only adapters for `usr`, `parquet`, and `matrix_bundle`
- Persisted `view`, `sample_set`, `projection`, and `plot` artifact families with `manifest.json`
- Structured mutating command results plus audit records under `outputs/logs/audit/`
- Docs skeleton, workspace templates, and targeted tests for layout, docs routing, CLI, config validation, and the first USR-backed integration path

### Constraints and deliberate deferrals

- This phase does not yet implement alignment artifacts, landmark workflows, neighbors, distances, enrichments, agreement, exports, recipes, or deliverables.
- Plot rendering currently covers `projection_scatter` and `projection_grid` only.
- Source-backed view materialization is implemented; derived-view execution is deferred even though static config validation already checks coordinate legality for declared vector differences.

### Validation notes

- Red gate: latentdna tests initially failed at import/collection because the package surface did not exist.
- Green target: package layout, workspace init/validate, and the USR-backed `inspect -> materialize -> sample -> projection -> plot` path.

### Next steps

1. Implement `alignment build` as a persisted artifact with explicit support semantics and aggregation rules.
2. Add landmark, distance, scalar, and enrichment primitives so the control-neighborhood and control-distance workflows move out of notebooks.
3. Add `view derive` and `view reduce` for `Δcontext` and low-rank export preparation.
4. Add `export matrix`, `recipe run`, and `deliverable status|run`.
5. Expand docs and benchmarks so the promoter-study template covers the full atlas and OPAL handoff path.

## 2026-04-09 - Alignment, Delta, and Landmark Slice

### Objective

Move the first multiview and landmark workflows out of notebooks by adding:

- explicit persisted `alignment_set` artifacts
- derived `vector_difference` views
- reusable scalar tables
- landmark distance scoring over source-backed views

### Implemented in this slice

- Typed workspace config support for `alignments`, `scalars`, and `landmarks`
- `latentdna alignment build` with persisted `rows.parquet` and `mapping.parquet`
- `latentdna view derive` for `vector_difference` artifacts with alignment-backed row support
- `latentdna scalar derive` for `vector_norm` and safe arithmetic `column_expression`
- `latentdna distance score` for centroid, medoid, and per-member row landmark representations
- Materialization fix so package-level metadata columns are projected only when they exist on a given source
- Promoter-study starter template extended with context view, delta view, landmark declarations, and first scalar definitions
- Integration coverage for `materialize -> alignment -> derive -> distance -> scalar`

### Constraints and deliberate deferrals

- `view derive` still implements `vector_difference` only.
- Landmark distance scoring currently targets source-backed views; more general derived-view/reference flows remain for later work.
- Recipes, deliverables, reducers, exports, neighbors, enrichments, agreements, and notebook scaffolds are still pending.

### Validation notes

- Green target:
  - `uv run ruff check src/dnadesign/latentdna`
  - `uv run ruff format --check src/dnadesign/latentdna`
  - `uv run pytest -q src/dnadesign/latentdna/tests`
- The latentdna pytest slice was green at the end of the alignment/delta/landmark slice.

### Next steps

1. Add `view reduce` plus reducer state artifacts so exports stop depending on full-width matrices.
2. Implement `export matrix` with a stable feature ledger for OPAL handoff.
3. Add `neighbors fit`, `enrich score`, and `agreement compare` so the remaining required workflow families are artifact-owned.
4. Layer `recipe validate|run` and `deliverable status|run` on top of the primitive commands without introducing hidden orchestration.
5. Add benchmark coverage for alignment, delta build, landmark distance scoring, and first export bundles.

## 2026-04-10 - Reducer and Export Slice

### Objective

Add the first explicit low-rank handoff path:

- persisted PCA reducer artifacts
- reduced-view artifacts for downstream reuse
- deterministic export bundles with a stable feature ledger

### Implemented in this slice

- `latentdna view reduce` for PCA reducers with persisted `state.npz` and `summary.json`
- optional `reduced_view` materialization from the fitted reducer
- export config support for named matrix bundles in workspace config
- `latentdna export matrix` for reducer-backed and table-backed block concatenation
- `features.parquet` ledgers with stable `feature_name`, `block_order`, and `feature_order`
- integration coverage for `derive -> sample -> reduce -> export`
- promoter-study starter template extended with first reducer/export bundle definitions

### Constraints and deliberate deferrals

- `view reduce` currently supports PCA only.
- `export matrix` currently supports `reduced_view` and `table_columns` blocks only.
- Export bundles still assume all blocks already share the same ordered row basis; more general cross-artifact alignment inside exports remains deferred.
- Recipes, deliverables, neighbors, enrichments, agreements, notebook scaffolds, and benchmark harnesses are still pending.

### Validation notes

- Green target:
  - `uv run ruff check src/dnadesign/latentdna`
  - `uv run ruff format --check src/dnadesign/latentdna`
  - `uv run pytest -q src/dnadesign/latentdna/tests`
  - `uv run python -m dnadesign.devtools.docs_checks`
- The latentdna pytest slice is green after the reducer/export integration path was added.

### Next steps

1. Add exact/approximate `neighbors fit` and `enrich score` so the control-neighborhood workflow becomes artifact-owned.
2. Add `agreement compare` and supporting cluster/neighbor summaries for cross-view structural comparisons.
3. Expand export support to more multiblock row-basis patterns needed for `x2_primary_20b` and `x3_ablation_7b`.
4. Layer `recipe validate|run` and `deliverable status|run` on top of the primitive commands without introducing hidden orchestration.
5. Add benchmark coverage and package-boundary tests for reducers, exports, and remaining runtime seams.

## 2026-04-10 - Neighbor and Agreement Slice

### Objective

Add the first persisted structural-comparison path after reduction/export:

- exact and approximate `neighbor_set` artifacts over explicit scopes
- agreement artifacts that compare cross-view local structure without raw coordinate mixing
- docs and template updates that keep the current handoff seam visible

### Implemented in this slice

- `latentdna neighbors fit` with persisted `indices.npy`, `distances.npy`, and `rows.parquet`
- exact and approximate neighbor backends with explicit backend recording in manifests
- scoped neighbor fitting over full-view, `sample_set`, or `alignment_set` row support
- `latentdna agreement compare` for kNN-overlap summaries over matching neighbor scopes
- `agreement_set` artifacts with `table.parquet` and `summary.json`
- public API wiring, CLI contract updates, and promoter-study workflow docs for the new structural comparison path
- end-to-end integration coverage for `materialize -> alignment -> neighbors -> agreement`

### Constraints and deliberate deferrals

- `agreement compare` currently implements kNN-overlap only; cluster agreement and landmark-neighborhood overlap remain for later work.
- `neighbors fit` currently supports `euclidean` and `cosine` metrics only.
- Enrichment, heatmap rendering for neighborhood stats, recipes, deliverables, notebook scaffolds, and benchmark harnesses are still pending.

### Validation notes

- Red gate: the new phase-4 integration test initially failed because `neighbors` and `agreement` command groups did not exist.
- Green target:
  - `uv run ruff check src/dnadesign/latentdna`
  - `uv run ruff format --check src/dnadesign/latentdna`
  - `uv run pytest -q src/dnadesign/latentdna/tests`
  - `uv run python -m dnadesign.devtools.docs_checks`
- The latentdna test slice is green after the neighbor/agreement path was added. The existing UMAP warning from the phase-1 projection test remains non-failing.

### Next steps

1. Add `enrich score` plus cohort contracts and `heatmap` rendering so the control-neighborhood workflow becomes fully artifact-owned.
2. Extend `agreement compare` with cluster- and landmark-neighborhood summaries where the workspace declares those prerequisite artifacts.
3. Expand export support to more multiblock row-basis patterns needed for `x2_primary_20b` and `x3_ablation_7b`.
4. Layer `recipe validate|run` and `deliverable status|run` on top of the primitive commands without introducing hidden orchestration.
5. Add benchmark coverage and package-boundary tests for neighbors, agreements, reducers, and export reuse paths.

## 2026-04-10 - Enrichment and Heatmap Slice

### Objective

Finish the control-neighborhood workflow as an artifact-owned path by adding:

- typed cohort declarations in `latentdna.workspace.v1`
- persisted neighborhood enrichment artifacts over configured landmarks and cohorts
- read-only heatmap rendering from those enrichment artifacts

### Implemented in this slice

- Typed `cohorts` support for `kind: column` declarations with workspace validation for referenced sources
- `latentdna enrich score` with persisted `table.parquet` and `summary.json` outputs under `outputs/enrichments/`
- Landmark-neighborhood cohort summaries over existing `neighbor_set` artifacts, including deterministic `neighbor_fraction`, `background_fraction`, `enrichment_delta`, and `enrichment_ratio` columns
- `heatmap` support in `latentdna plot render` via `--enrichment` and `--value-column`
- Promoter-study template, workflow docs, CLI contracts, and docs tests updated to route the new control-neighborhood slice
- End-to-end integration coverage for `materialize -> neighbors -> enrich -> heatmap`

### Constraints and deliberate deferrals

- `enrich score` currently supports `kind: column` cohorts only.
- The current enrichment summary is deterministic and artifact-first, but it does not yet implement hypothesis tests or multiple-testing correction.
- Landmark-neighborhood enrichment currently expects the neighbor row basis to carry the cohort and landmark selector columns directly; richer alignment-backed joins remain for later work.
- Recipes, deliverables, notebook scaffolds, and benchmark harnesses are still pending.

### Validation notes

- Red gate:
  - cohort config validation initially did not reject unknown sources
  - the new integration path initially failed because the `enrich` command group did not exist
- Green target:
  - `uv run pytest -q src/dnadesign/latentdna/tests/contracts/test_workspace_config.py src/dnadesign/latentdna/tests/integrations/test_phase5_enrichment_heatmap_workflow.py`
  - `uv run ruff check src/dnadesign/latentdna`
  - `uv run ruff format --check src/dnadesign/latentdna`
  - `uv run pytest -q src/dnadesign/latentdna/tests`
  - `uv run python -m dnadesign.devtools.docs_checks`
- The latentdna slice is green after the enrichment/heatmap path was added. The existing UMAP warning from the phase-1 projection test remains non-failing.

### Next steps

1. Extend `agreement compare` with cluster- and landmark-neighborhood summaries where the workspace declares those prerequisite artifacts.
2. Expand export support to more multiblock row-basis patterns needed for `x2_primary_20b` and `x3_ablation_7b`.
3. Layer `recipe validate|run` and `deliverable status|run` on top of the primitive commands without introducing hidden orchestration.
4. Add benchmark coverage and package-boundary tests for neighbors, enrichments, agreements, reducers, and export reuse paths.
5. Decide whether enrichment should grow explicit statistical testing or remain a deterministic summary primitive with tests layered elsewhere.

## 2026-04-10 - Recipe and Deliverable Slice

### Objective

Add the thin orchestration layer promised by the build spec without turning `latentdna` into a hidden workflow engine:

- typed `recipes` and `deliverables` in `latentdna.workspace.v1`
- `recipe validate|run` over existing primitive services only
- `deliverable list|status|run` with explicit readiness reporting

### Implemented in this slice

- Typed workspace support for `recipes.steps[*]` and `deliverables` plus static validation for unsupported ops, unknown references, and cyclic step graphs
- `latentdna recipe validate` with deterministic step ordering
- `latentdna recipe run` as a thin orchestrator over the existing primitive service layer, including safe skip behavior for already-materialized valid artifacts
- `latentdna deliverable list`, `latentdna deliverable status`, and `latentdna deliverable run`
- `latentdna.deliverable_status.v1` for user-facing readiness checks over declared prerequisites and outputs
- Template, workflow docs, and CLI reference updates for the first checked-in orchestration path: `control_neighborhood_enrichment`
- Plot fast-fail fix so missing projection inputs raise a typed missing-artifact error without leaving behind a poisoned empty plot directory
- End-to-end integration coverage for `deliverable status -> deliverable run -> recipe rerun -> partial deliverable attention`

### Constraints and deliberate deferrals

- Recipes currently orchestrate the existing primitive surface only; cluster, notebook, and benchmark steps are still outside the implemented registry.
- `deliverable status` reports presence/readiness, but it does not yet compute stale-input digests or freshness mismatches.
- `recipe run` skips fully materialized valid artifacts, but partial step outputs still require an explicit `--force` rebuild rather than silent repair.
- Notebook scaffolds, benchmark harnesses, cluster-backed agreement summaries, and the remaining export-bundle expansions are still pending.

### Validation notes

- Red gate:
  - workspace loading initially accepted cyclic recipe graphs and deliverables that referenced missing recipes
  - the missing `recipe`/`deliverable` CLI groups blocked the new phase-6 orchestration path
  - `plot render` was leaking a raw missing-file path instead of a typed missing-artifact error
- Green target:
  - `uv run pytest -q src/dnadesign/latentdna/tests/contracts/test_workspace_config.py src/dnadesign/latentdna/tests/integrations/test_phase6_recipe_deliverable_workflow.py`
  - `uv run ruff check src/dnadesign/latentdna`
  - `uv run ruff format --check src/dnadesign/latentdna`
  - `uv run pytest -q src/dnadesign/latentdna/tests`
  - `uv run python -m dnadesign.devtools.docs_checks`
- The targeted phase-6 slice is green. The existing UMAP warning from the phase-1 projection stack remains non-failing.

### Next steps

1. Extend `agreement compare` with cluster- and landmark-neighborhood summaries where the workspace declares those prerequisite artifacts.
2. Expand export support to more multiblock row-basis patterns needed for `x2_primary_20b` and `x3_ablation_7b`.
3. Add notebook scaffolds that load persisted artifacts and hook them into the recipe/deliverable surface without re-owning computation.
4. Add benchmark coverage and package-boundary tests for recipes, deliverables, neighbors, enrichments, agreements, reducers, and export reuse paths.
5. Decide whether enrichment should grow explicit statistical testing or remain a deterministic summary primitive with tests layered elsewhere.

## 2026-04-10 - Notebook Scaffold Slice

### Objective

Add the first notebook surface promised by the build spec without letting notebooks retake ownership of latent-analysis computation:

- typed `notebooks` in `latentdna.workspace.v1`
- `latentdna notebook generate` for read-only artifact review scaffolds
- recipe/deliverable support for notebook artifacts generated from already-materialized outputs

### Implemented in this slice

- Typed workspace support for `notebooks.<id>` with `artifact_review` scaffold declarations and explicit artifact references
- Static validation for unsupported notebook artifact kinds and duplicate notebook artifact aliases
- `latentdna notebook generate` that fails fast on missing prerequisite artifacts and writes one immutable notebook artifact under the workspace output root
- Generated `notebook.py` scaffolds with portable workspace-relative artifact paths plus loader helpers for manifests, tables, coordinates, and matrices
- `recipe run` support for `notebook.generate`, so notebooks can remain a thin downstream step in declared deliverables
- Template, workflow docs, CLI reference docs, and package-layout tests updated for the first checked-in notebook path: `control_plan_review`
- End-to-end integration coverage for direct notebook fast-fail plus `deliverable run -> notebook artifact -> recipe rerun skip`

### Constraints and deliberate deferrals

- The current notebook surface emits Python review scaffolds only; richer notebook templates and richer narrative/report scaffolds are still deferred.
- Notebook configs currently validate artifact kinds and aliases, but they do not yet statically prove that non-config-backed artifact ids will be produced by a matching recipe.
- Notebook generation reads persisted artifacts only; it does not repair or recompute missing upstream views, samples, projections, or plots.
- Benchmark harnesses, richer agreement summaries, and the remaining export-family expansions are still pending.

### Validation notes

- Red gate:
  - workspace loading initially ignored `notebooks:` entirely, so notebook declarations were silently dropped from the contract surface
  - the `notebook` CLI group and `notebook.generate` recipe op did not exist, so the end-to-end scaffold path was unreachable
  - workflow/reference docs did not advertise the new notebook surface
- Green target:
  - `uv run pytest -q src/dnadesign/latentdna/tests/contracts/test_workspace_config.py src/dnadesign/latentdna/tests/integrations/test_phase7_notebook_workflow.py src/dnadesign/latentdna/tests/test_docs_contract.py`
  - `uv run ruff check src/dnadesign/latentdna`
  - `uv run ruff format --check src/dnadesign/latentdna`
  - `uv run pytest -q src/dnadesign/latentdna/tests`
  - `uv run python -m dnadesign.devtools.docs_checks`
- The targeted phase-7 slice is green. The existing UMAP warning from the projection stack remains non-failing.

### Next steps

1. Extend `agreement compare` with cluster- and landmark-neighborhood summaries where the workspace declares those prerequisite artifacts.
2. Expand export support to more multiblock row-basis patterns needed for `x2_primary_20b` and `x3_ablation_7b`.
3. Add benchmark coverage and stronger package-boundary tests for notebook scaffolds, recipes, deliverables, neighbors, enrichments, agreements, reducers, and export reuse paths.
4. Decide whether notebook configs should grow explicit template variants or stay on one `artifact_review` scaffold until benchmark and agreement hardening land.
5. Decide whether enrichment should grow explicit statistical testing or remain a deterministic summary primitive with tests layered elsewhere.

## 2026-04-10 - Clustered Agreement Slice

### Objective

Extend the structural-agreement surface beyond row-level kNN overlap while keeping artifacts explicit and backwards compatible:

- add one minimal `cluster fit` primitive so cluster agreement is a real artifact workflow rather than a placeholder
- extend `agreement compare` to summarize cluster agreement and landmark-neighborhood overlap alongside the existing kNN overlap rows
- preserve the existing kNN-only artifact and summary shape so earlier slices and consumers do not regress

### Implemented in this slice

- `latentdna cluster fit` with deterministic `kmeans` over view-backed full-view, sample, or alignment scopes
- shared scoped-view helpers so `neighbors fit` and `cluster fit` reuse the same alignment/sample row-basis resolution path
- `cluster_set` artifacts with `assignments.parquet`, `summary.json`, and `manifest.json`
- recipe/API/CLI wiring for `cluster.fit`, including package exports and CLI registration
- richer `latentdna agreement compare` inputs:
  - optional `--left-clusters` and `--right-clusters`
  - optional repeated `--landmark`
- mixed-method `agreement_set` artifacts that now persist:
  - row-level `knn_overlap` rows
  - global `cluster_agreement` metric rows for ARI and NMI
  - per-landmark `landmark_neighbor_overlap` rows with Jaccard and side-specific overlap fractions
- alignment-aware landmark selection for agreement summaries, so landmark predicates can still resolve when alignment-scoped neighbor rows no longer carry the original metadata selector columns
- backward-compatible kNN summary fields at the top level of `summary.json` so the earlier phase-4 consumer shape still passes unchanged
- phase-8 integration coverage for `cluster fit + rich agreement compare` and package-layout coverage for the new internal `clusters/` module

### Constraints and deliberate deferrals

- `cluster fit` currently clusters view-backed matrices only; projection-backed clustering is still deferred.
- Agreement summaries currently emit global cluster metrics only; there is no persisted confusion matrix or per-cluster overlap table yet.
- Landmark-neighborhood overlap currently relies on either selector columns already present in the neighbor rows or an alignment scope that exposes reusable key columns.
- The checked-in workspace template still does not declare a cluster-backed recipe or deliverable by default; the new primitive surface is available, but template adoption is a later step.
- Benchmark harnesses and the remaining export-family expansions are still pending.

### Validation notes

- Red gate:
  - the new phase-8 integration test initially failed because the `cluster` CLI group and `cluster fit` service did not exist
  - the first mixed-method agreement artifact dropped method-specific columns when writing `table.parquet`, which broke cluster metric rows
  - the first richer agreement summary shape regressed the old top-level kNN summary keys expected by the phase-4 workflow test
  - docs checks failed once on a missing explanatory comment in the workflow shell block after the new cluster commands were added
- Green target:
  - `uv run pytest -q src/dnadesign/latentdna/tests/integrations/test_phase4_neighbors_agreement_workflow.py src/dnadesign/latentdna/tests/integrations/test_phase8_cluster_agreement_workflow.py`
  - `uv run ruff check src/dnadesign/latentdna`
  - `uv run ruff format --check src/dnadesign/latentdna`
  - `uv run pytest -q src/dnadesign/latentdna/tests`
  - `uv run python -m dnadesign.devtools.docs_checks`
- The targeted phase-8 slice is green and the full latentdna test slice is green. The existing UMAP warning from the projection stack remains non-failing; this slice did not add new warnings.

### Next steps

1. Expand export support to more multiblock row-basis patterns needed for `x2_primary_20b` and `x3_ablation_7b`.
2. Add benchmark coverage and stronger package-boundary tests for clusters, notebook scaffolds, recipes, deliverables, neighbors, enrichments, agreements, reducers, and export reuse paths.
3. Decide whether notebook configs should grow explicit template variants or stay on one `artifact_review` scaffold until benchmark and export hardening land.
4. Decide whether enrichment should grow explicit statistical testing or remain a deterministic summary primitive with tests layered elsewhere.
5. Decide whether `cluster fit` should grow projection-backed inputs and checked-in cluster-backed recipe/deliverable templates now that `agreement compare` can consume cluster artifacts.

## 2026-04-10 - Alignment-Backed Export Slice

### Objective

Finish the next practical export seam after clustered agreement by unblocking mixed-basis export bundles:

- allow `export matrix` to use an explicit `alignment_set` as the row basis
- let `reduced_view` and `table_columns` blocks project onto that aligned support without hidden row-order assumptions
- fill in the checked-in committee template/docs so `x2_primary_20b` and `x3_ablation_7b` stop being placeholder next steps

### Implemented in this slice

- Typed export-block support for optional `alignment` and `alignment_aggregation` fields in `latentdna.workspace.v1`
- Static workspace validation that now rejects export blocks referencing unknown alignments
- `export matrix` row-basis resolution extended to explicit `alignment_set` and `sample_set` ledgers
- Alignment-backed block projection in `src/dnadesign/latentdna/src/exports/matrix.py`:
  - `reduced_view` blocks can now reorder and aggregate onto alignment support
  - `table_columns` blocks can now do the same for scalar/distance tables
- Export manifests now record the alignment dependency for projected blocks instead of silently depending only on source files
- Checked-in `landmark_atlas_committee` template expanded with:
  - 7B paired views plus `delta7`
  - `anchor_ctx_7b`
  - `x2_primary_20b`
  - `x3_ablation_7b`
- Workflow/reference docs updated so the multiblock export path is advertised as implemented rather than deferred
- Phase-9 integration coverage for mixed-basis exports that concatenate:
  - anchor-space reduced views projected onto aligned support
  - aligned distance columns
  - alignment-native delta reduced views/scalars

### Constraints and deliberate deferrals

- Export alignment is still explicit-only: blocks project onto aligned support only when the workspace declares `alignment`; there is still no implicit key-based join fallback.
- `export matrix` still supports `reduced_view` and `table_columns` blocks only.
- More complex export joins that are not naturally expressible as one aligned support are still deferred; this slice only covers explicit alignment-backed projection onto a single row basis.
- The checked-in template now declares `x2` and `x3`, but there are still no checked-in recipes or deliverables that materialize those bundles automatically.

### Validation notes

- Red gate:
  - the export builder initially still assumed identical row ledgers and could not use `alignment_set` rows as a bundle basis
  - export blocks could reference undeclared alignments without static validation
- Green target:
  - `uv run pytest -q src/dnadesign/latentdna/tests/contracts/test_workspace_config.py src/dnadesign/latentdna/tests/integrations/test_phase9_export_alignment_workflow.py`
  - `uv run ruff check src/dnadesign/latentdna`
  - `uv run ruff format --check src/dnadesign/latentdna`
  - `uv run pytest -q src/dnadesign/latentdna/tests`
  - `uv run python -m dnadesign.devtools.docs_checks`
- The targeted phase-9 slice is green and the full latentdna test slice is green. The existing UMAP warning from the projection stack remains non-failing; this slice did not add new warnings.

### Next steps

1. Add benchmark coverage and stronger package-boundary tests for clusters, notebook scaffolds, recipes, deliverables, neighbors, enrichments, agreements, reducers, and export reuse paths.
2. Decide whether notebook configs should grow explicit template variants or stay on one `artifact_review` scaffold until benchmark hardening lands.
3. Decide whether enrichment should grow explicit statistical testing or remain a deterministic summary primitive with tests layered elsewhere.
4. Decide whether `cluster fit` should grow projection-backed inputs and checked-in cluster-backed recipe/deliverable templates now that `agreement compare` can consume cluster artifacts.
5. Decide whether export should grow anything beyond explicit alignment-backed projection or keep more complex row-basis joins outside the primitive `export matrix` surface.

## 2026-04-10 - Tabular Export, Snapshot, and Artifact Inventory Slice

### Objective

Close the next spec-level public-surface gaps after alignment-backed exports:

- add `export table` so the same configured bundle can emit an aligned tabular handoff instead of numeric matrix-only output
- add `snapshot build` so workspaces can freeze source metadata ledgers without materializing vectors
- add explicit artifact inventory/reporting surfaces (`inspect artifacts`, `runs list|show|prune`) so persisted state is inspectable and maintainable from the CLI

### Implemented in this slice

- Shared export-block resolution in `src/dnadesign/latentdna/src/exports/matrix.py`, so matrix and table exports reuse the same alignment-aware block projection logic and feature-ledger ordering
- New `src/dnadesign/latentdna/src/exports/table.py` builder plus `latentdna export table`
- `export table` artifacts now write:
  - `rows.parquet`
  - `table.parquet`
  - `features.parquet`
  - `manifest.json`
- New artifact inventory service in `src/dnadesign/latentdna/src/services/run_service.py` plus:
  - `latentdna runs list`
  - `latentdna runs show`
  - `latentdna runs prune`
- Deliverable-safe pruning guardrails:
  - `runs prune` now refuses to delete artifacts still referenced by live deliverables unless `--force` is explicit
- Richer inspect surfaces in `src/dnadesign/latentdna/src/services/inspection_service.py` plus:
  - `latentdna inspect views`
  - `latentdna inspect alignment`
  - `latentdna inspect landmarks`
  - `latentdna inspect missingness`
  - `latentdna inspect artifacts`
- New `snapshot build` primitive with workspace-owned `snapshot` artifacts under `outputs/snapshots/`
- Recipe support extended so the new `export.table` and `snapshot.build` primitives can participate in thin orchestration instead of staying ad hoc only
- Phase-10 and phase-11 integration coverage for:
  - alignment-backed tabular exports
  - artifact inventory/show/prune flows
  - snapshot metadata row ledgers that exclude vector payload columns

### Constraints and deliberate deferrals

- `snapshot build` currently persists `rows.parquet` only; there is still no separate `metadata.parquet` companion artifact.
- Artifact inventory is manifest-backed and workspace-local; there is still no stale-input digest analysis layered onto `runs list` or `inspect artifacts`.
- `runs prune` currently removes one explicit artifact per command; there is still no broader garbage-collection mode for pruning whole unreferenced classes of artifacts.
- `export table` reuses the existing export block kinds only; there is still no new export-only block family beyond `reduced_view` and `table_columns`.
- The broader benchmark harness and stronger package-boundary tests are still pending.

### Validation notes

- Red gate:
  - the phase-10 workflow failed first because `latentdna export table` did not exist
  - the same phase-10 workflow also failed because `runs` and the richer `inspect` subcommands were missing
  - the phase-11 workflow failed because `latentdna snapshot build` did not exist
- Green target:
  - `uv run pytest -q src/dnadesign/latentdna/tests/integrations/test_phase10_export_table_runs_workflow.py`
  - `uv run pytest -q src/dnadesign/latentdna/tests/integrations/test_phase11_snapshot_workflow.py`
- The targeted new slices are green.

### Next steps

1. Add benchmark coverage and stronger package-boundary tests for clusters, notebook scaffolds, recipes, deliverables, neighbors, enrichments, agreements, reducers, export reuse paths, and the new snapshot/inventory surfaces.
2. Decide whether `snapshot build` should also emit `metadata.parquet` and whether recipes/deliverables should start depending on snapshots by default rather than live sources.
3. Decide whether `runs prune` should grow batch pruning modes for unreferenced artifact classes or stay explicit-per-artifact.
4. Decide whether notebook configs should grow explicit template variants or stay on one `artifact_review` scaffold until benchmark hardening lands.
5. Decide whether `cluster fit` should grow projection-backed inputs and checked-in cluster-backed recipe/deliverable templates now that `agreement compare` can consume cluster artifacts.

## 2026-04-10 - Snapshot Metadata Companion Slice

### Objective

Tighten the new snapshot public surface so the artifact separates stable row support from copied metadata:

- keep `rows.parquet` as the key ledger for downstream row-basis reuse
- add `metadata.parquet` as the richer non-vector companion
- advertise the split in manifests and docs so future snapshot-backed flows have a clearer seam

### Implemented in this slice

- `snapshot build` now writes both:
  - `rows.parquet` with the declared `record_key`, `subject_key`, and optional `context_key`
  - `metadata.parquet` with the row-basis columns plus configured metadata columns
- Snapshot manifests now record both output files and expose separate `row_columns` and `metadata_columns` params
- Source snapshots still exclude vector payload columns from both files
- Phase-11 snapshot coverage was tightened around the key-ledger contract, and a new phase-12 workflow test now covers the `metadata.parquet` companion

### Constraints and deliberate deferrals

- Recipes and deliverables still consume live sources unless a workspace explicitly chooses to route through snapshots; this slice improves the snapshot artifact contract but does not change orchestration defaults.
- `metadata.parquet` currently mirrors only configured/copied source metadata columns; there is still no extra provenance sidecar beyond the manifest and audit log.
- The broader benchmark harness and stronger package-boundary tests are still pending.

### Validation notes

- Red gate:
  - the existing snapshot builder still wrote one combined `rows.parquet` ledger, so the tightened phase-11 test failed immediately
  - the new phase-12 workflow failed because `metadata.parquet` did not exist and the manifest only advertised one output
- Green target:
  - `uv run pytest -q src/dnadesign/latentdna/tests/integrations/test_phase11_snapshot_workflow.py src/dnadesign/latentdna/tests/integrations/test_phase12_snapshot_metadata_workflow.py`
- The targeted snapshot companion slice is green.

### Next steps

1. Add benchmark coverage and stronger package-boundary tests for clusters, notebook scaffolds, recipes, deliverables, neighbors, enrichments, agreements, reducers, export reuse paths, and the new snapshot/inventory surfaces.
2. Decide whether recipes and deliverables should start preferring snapshot-backed inputs by default or keep snapshots as an explicit opt-in durability layer.
3. Decide whether `runs prune` should grow batch pruning modes for unreferenced artifact classes or stay explicit-per-artifact.
4. Decide whether notebook configs should grow explicit template variants or stay on one `artifact_review` scaffold until benchmark hardening lands.
5. Decide whether `cluster fit` should grow projection-backed inputs and checked-in cluster-backed recipe/deliverable templates now that `agreement compare` can consume cluster artifacts.

## 2026-04-10 - Real Promoter-Study Pressure Slice

### Objective

Move the checked-in promoter-study committee path from schema-only confidence to real-study pressure:

- bind the committee workspace to the active `stress_ethanol_cipro_growth` study record
- reconcile the checked-in template with the live source schema
- push one real artifact path past `validate workspace --deep`
- record the next true blocker instead of stopping at static review findings

### Implemented in this slice

- The checked-in `landmark_atlas_committee` template now matches the active study's live committee columns:
  - 7B intermediate views now reference `block26_mlp_out`
  - the plan cohort now uses `densegen__plan`
  - the checked-in control-neighborhood recipe no longer hard-codes `backend: exact`; it now uses `backend: auto`
- `view materialize` no longer converts vector columns through `to_pylist()`. It now:
  - validates numeric Arrow list columns directly
  - streams source batches through the USR/parquet scan seam
  - writes `matrix.npy` incrementally with an on-disk memmap
  - streams `rows.parquet` with a `ParquetWriter`
  - cleans up partial view directories if materialization fails
- Workspace command coverage gained a realish promoter-study fixture that exercises:
  - `workspace init --from-study-dir`
  - `validate workspace --deep`
  - `view materialize`
  - `sample build`
  against USR overlay columns shaped like the active study committee data

### Constraints and deliberate deferrals

- Real full-scale view materialization is still blocked upstream of latentdna's old full-table assembly path. The active-study `z7_60` run now fails inside the USR DuckDB overlay query layer with an out-of-memory error before the first batch reaches latentdna's writer loop.
- `alignment build` still requires pre-materialized view artifacts, so the real promoter-study alignment path cannot advance until at least the paired source-backed views are materialized.
- The broader benchmark harness, additional deliverable families, richer derived-view breadth, and the remaining spec-level plot kinds are still pending.

### Validation notes

- Green targeted checks:
  - `uv run ruff check src/dnadesign/latentdna/src/sources/resolver.py src/dnadesign/latentdna/src/views/materialize.py src/dnadesign/latentdna/src/services/view_service.py src/dnadesign/latentdna/tests/cli/test_workspace_command.py`
  - `uv run pytest -q src/dnadesign/latentdna/tests/cli/test_workspace_command.py src/dnadesign/latentdna/tests/integrations/test_phase1_workflow.py`
- Real promoter-study commands:
  - `uv run latentdna workspace init --workspace /tmp/latentdna_stress_ethanol_cipro_pressure_v2 --template landmark_atlas_committee --from-study-dir docs/studies/stress_ethanol_cipro_growth --json`
  - `uv run latentdna validate workspace --workspace /private/tmp/latentdna_stress_ethanol_cipro_pressure_v2 --deep --json`
- The real study workspace now deep-validates cleanly against the active USR planes, including the live 7B `block26` columns and the live `densegen__plan` cohort field.
- Real `alignment build anchor_ctx_20b` and `alignment build anchor_ctx_7b` still fail with missing-prerequisite errors because the paired view artifacts are not materialized yet. That is current behavior, not a new regression.
- Real `view materialize z7_60` now reaches the live scan path and fails later with:
  - `Out of Memory Error: failed to allocate data of size 16.0 MiB (12.7 GiB/12.7 GiB used)`
  This is a more useful failure than the earlier template/schema mismatch because it isolates the remaining bottleneck to the USR overlay query/runtime seam.

### Next steps

1. Reduce memory pressure in the USR overlay scan path for wide vector columns so the active-study `z7_60` and `z20_60` views can materialize at 157k-row scale.
2. After that reader-side fix lands, rerun the real sequence `view materialize -> sample build -> projection fit` on the active study and record artifact sizes/timings.
3. Decide whether `alignment build` should grow a source/snapshot-backed mode so explicit aligned support can be compiled before full vector materialization.
4. Continue the remaining spec gaps in priority order: benchmark harness, missing plot kinds, missing derived-view kinds, and the broader promoter-study deliverable bundle set.

## 2026-04-10 - Swarm Audit Follow-Up

### Objective

Run a bounded multi-agent audit after the real-study pressure slice to rank the remaining latentdna gaps, misalignments, and blockers more precisely.

### Findings carried forward

- The live promoter-study `view materialize -> sample -> projection` path is still blocked at 157k-row scale by the upstream USR DuckDB scan seam. Latentdna now reaches that seam cleanly, but the active `z7_60` run still OOMs before the first batch is written.
- Artifact freshness is still under-modeled for USR sources. View and snapshot manifests currently hash only `records.parquet`, even though the active study depends on `_derived/infer` and `_derived/construct` overlay files. That means overlay changes can evade stale-input detection.
- Several spec-required breadth items are still open:
  - derived-view kinds beyond `vector_difference`
  - sample strategies beyond `all`, `random`, and `stratified`
  - plot kinds beyond `projection_scatter`, `projection_grid`, and `heatmap`
  - richer readiness/freshness semantics for deliverables and run inventory
  - the documented common CLI flag surface (`--quiet`, `--dry-run`) is still not implemented package-wide
- The docs/tests/template surface is still thinner than the spec and still has some study-scale drift:
  - one workflow example still used `--backend exact` on a structural-agreement path even though full-view exact neighbors are rejected above 5000 rows
  - there is still no automated regression that exercises the real study-scale materialization blocker
  - the required top-level test tree from the build spec is still only partially present

### Validation notes

- Swarm mode: `platform_subagent`, star topology, `max_workers=3`, `max_depth=1`, read-only.
- Reviewer coverage split across:
  - runtime/performance and real-study pressure seams
  - contract/spec drift
  - docs/tests/templates/journal alignment

### Next steps

1. Fix the USR overlay scan memory seam first; it is still the dominant blocker for the live study path.
2. Extend input digest/freshness tracking so USR overlay files participate in stale-input detection for views, snapshots, deliverables, and run inventory.
3. Finish the next highest-value spec gaps in breadth order: derived views, sample strategies, and missing plot kinds.
4. Tighten docs/tests so the checked-in workflow examples and regression coverage reflect what is actually safe at active promoter-study scale.

## 2026-04-10 - Missing Plot Kinds Slice

### Objective

Close the remaining spec-required read-only plot gaps without reintroducing notebook-owned logic:

- add `distance_scatter` over persisted `distance_set` tables
- add `distribution` over one explicit persisted table-backed artifact at a time
- add `agreement_summary` over persisted `agreement_set` summaries

### Implemented in this slice

- `latentdna plot render` now supports:
  - `distance_scatter` from `--distance`
  - `distribution` from one explicit `--scalar`, `--distance`, `--enrichment`, or `--agreement`
  - `agreement_summary` from `--agreement`
- Plot service and recipe wiring now carry the new artifact ids plus optional `--x-column` and `--y-column` parameters through to manifests and audit records.
- `defaults.plot_formats` is now behaviorally active: plot artifacts write the configured output formats instead of hard-coding `svg` and `png`.
- Workflow/reference docs now advertise the new artifact-driven plot surface, including promoter-study examples for distance and agreement diagnostics.
- New phase-13 integration coverage exercises the three new renderer kinds from persisted `distance_set` and `agreement_set` artifacts.

### Constraints and deliberate deferrals

- `distribution` currently renders one explicit artifact table at a time; it does not yet join or facet across multiple artifact families in one plot.
- `agreement_summary` currently renders aggregate metrics from `summary.json`; it does not yet emit per-cluster or per-landmark drill-down panels.
- This slice does not touch the live-study USR overlay-scan blocker, the remaining CLI `--quiet` / `--dry-run` harmonization work, or the broader derived-view breadth gap.

### Validation notes

- Green targeted checks:
  - `uv run ruff check src/dnadesign/latentdna/src/plots/render.py src/dnadesign/latentdna/src/services/plot_service.py src/dnadesign/latentdna/src/cli/commands/plot.py src/dnadesign/latentdna/src/services/recipe_service.py src/dnadesign/latentdna/tests/integrations/test_phase13_plot_diagnostics_workflow.py src/dnadesign/latentdna/tests/test_docs_contract.py`
  - `uv run pytest -q src/dnadesign/latentdna/tests/integrations/test_phase1_workflow.py src/dnadesign/latentdna/tests/integrations/test_phase5_enrichment_heatmap_workflow.py src/dnadesign/latentdna/tests/integrations/test_phase6_recipe_deliverable_workflow.py src/dnadesign/latentdna/tests/integrations/test_phase13_plot_diagnostics_workflow.py src/dnadesign/latentdna/tests/test_docs_contract.py`
  - `uv run python -m dnadesign.devtools.docs_checks`
- Regression notes:
  - Existing `projection_scatter`, `heatmap`, and recipe-driven plot flows remained green after the new renderer kinds were added.
  - The existing UMAP warning from the projection stack remains non-failing and unchanged by this slice.

### Next steps

1. Return to the active-study pressure path and keep working the upstream USR overlay scan seam until `view materialize z7_60` completes without OOM.
2. Harmonize the documented CLI common-flag surface, especially `--quiet` and `--dry-run`, across the remaining mutating command families.
3. Continue narrowing the remaining contract drift in `latentdna.workspace.v1`, especially the still-suspect reserved-vs-active fields outside the plotting surface.
4. Close the next breadth gaps after the live-study blocker: additional derived-view kinds and any still-missing sample/analysis surface required by the revised spec.

## 2026-04-10 - USR Overlay Projection and Provenance Slice

### Objective

Keep pushing on the real-study materialization blocker without widening the visible `latentdna` surface:

- narrow the upstream USR DuckDB overlay scan so projected scans request only the overlay columns the caller actually needs
- make `latentdna` source manifests more truthful by recording concrete overlay-part provenance instead of only coarse overlay roots

### Implemented in this slice

- USR overlay query planning now passes the requested derived-column subset through to the overlay-view builder instead of always materializing every overlay column in a namespace.
- `create_overlay_view(...)` now projects only the requested key-plus-derived columns from overlay parquet parts, including the multi-part dedupe path that ranks overlay rows by creation time.
- Added a USR regression in `src/dnadesign/usr/tests/test_dataset_scan_projection.py` that proves `_duckdb_query(...)` forwards the projected overlay column list to `_create_overlay_view(...)`.
- `latentdna` USR source provenance now records:
  - the overlay root path for conservative freshness on added/removed parts
  - one `overlay_part` entry per contributing parquet part for more truthful physical provenance in manifests
- Added a `latentdna` freshness regression that proves a materialized view manifest records `overlay_part` provenance for the exact requested `infer` and `densegen` overlay columns.

### Constraints and deliberate deferrals

- This slice reduces avoidable overlay scan width, but it does not yet prove the full live-study `z7_60` materialization path end-to-end at 157k-row scale.
- The active-study OOM may still need more work in the USR/DuckDB seam beyond projected overlay columns, especially if DuckDB is still retaining a large intermediate during the join/arg-max path.
- The package-wide CLI `--quiet` / `--dry-run` harmonization and the broader derived-view breadth gap remain untouched here.

### Validation notes

- Green targeted checks:
  - `uv run pytest -q src/dnadesign/usr/tests/test_dataset_scan_projection.py`
  - `uv run pytest -q src/dnadesign/latentdna/tests/integrations/test_freshness_contracts.py -k overlay_part_provenance`
  - `uv run pytest -q src/dnadesign/usr/tests/test_dataset_scan_projection.py src/dnadesign/latentdna/tests/integrations/test_freshness_contracts.py src/dnadesign/latentdna/tests/cli/test_workspace_command.py -k 'overlay or materialize or freshness'`
  - `uv run ruff check src/dnadesign/usr/src/dataset.py src/dnadesign/usr/src/dataset_overlay_query.py src/dnadesign/usr/src/dataset_query.py src/dnadesign/usr/tests/test_dataset_scan_projection.py src/dnadesign/latentdna/src/sources/usr_source.py src/dnadesign/latentdna/tests/integrations/test_freshness_contracts.py`
- Red-green note:
  - the new `test_view_manifest_records_overlay_part_provenance` check failed first because view manifests only recorded overlay directories, not concrete `overlay_part` entries
  - it went green after `usr_source.source_provenance(...)` started emitting per-part provenance while retaining overlay-root entries for conservative freshness

### Next steps

1. Re-run the real active-study `view materialize z7_60` pressure path against this narrower overlay scan to see whether the OOM moved or cleared.
2. If the live-study run still fails, instrument the remaining DuckDB seam more directly so time-to-first-batch and peak-RSS can be measured around the overlay join/arg-max path.
3. After the live-study scan seam is genuinely stable, finish the remaining freshness truthfulness work in runs/deliverables and then return to package-wide CLI `--quiet` / `--dry-run` harmonization.

## 2026-04-10 - Freshness-Aware Recipe Rebuild and Live z7 Pressure Follow-Up

### Objective

Close the highest-signal latentdna-native gap exposed by the swarm audit while immediately pressure-testing the live promoter-study path again:

- stop `recipe run` from silently skipping stale artifacts that already exist on disk
- tighten deliverable config validation where the referenced output is statically knowable from workspace config
- re-run the active-study `z7_60` path far enough to replace the old OOM note with real current evidence

### Implemented in this slice

- `recipe run` now evaluates expected step artifacts for freshness before deciding to skip:
  - if every expected output exists and is fresh, the step is still skipped
  - if every expected output exists but any output is stale/attention/error, the step is rebuilt with step-local `force=True` instead of being silently skipped
  - recipe metrics now record `rebuilt_steps`, and per-step summaries record `status: rebuilt` plus the recorded rebuild reasons
- Workspace loading now validates config-backed deliverable outputs the same way it already validated config-backed `requires` references. This catches cases like `outputs.views: [missing_view]` at load time instead of failing later during status/run flows.
- Added regressions for both behaviors:
  - `test_recipe_run_rebuilds_stale_view_after_overlay_change`
  - `test_load_workspace_config_rejects_unknown_config_backed_deliverable_output`
- The live promoter-study `z7_60` pressure path advanced materially:
  - `view materialize z7_60` succeeded on the active study workspace instead of reproducing the earlier OOM
  - the resulting view artifact contains `157164` rows and `4096` dims
  - the view artifact footprint is currently about `2.4G` on disk (`matrix.npy` about `2.4G`, `rows.parquet` about `6.3M`, `manifest.json` about `529K`)
  - `sample build z7_60_sample20k --strategy random --n 20000 --seed 17` succeeded on the persisted live view artifact
  - `projection fit z7_60 --sample z7_60_sample20k --run-id umap_z7_60_sample20k` also succeeded, producing a projection artifact of about `1.1M`

### Constraints and deliberate deferrals

- This slice improves freshness-aware orchestration for the existing recipe surface, but the broader freshness story is still incomplete for snapshots, run inventory, and deliverable-level stale-input reporting beyond the already-implemented checks.
- Deliverable output validation is only tightened for categories that are statically config-backed. Runtime-only output categories such as `samples`, `projections`, and `plots` still cannot be fully validated at load time.
- The live pressure evidence is currently strongest for the 7B anchor-only primary view. The corresponding real `z20_60`, `z7_1k_anchor`, `z20_1k_anchor`, and downstream alignment/delta/export paths were not re-run in this slice.
- High-cardinality `overlay_part` provenance is now truthful but large. The real `z7_60` manifest is already sizable because it records every contributing overlay part explicitly.

### Validation notes

- Green targeted checks:
  - `uv run ruff check src/dnadesign/latentdna/src/services/recipe_service.py src/dnadesign/latentdna/src/workspaces/loader.py src/dnadesign/latentdna/tests/integrations/test_freshness_contracts.py src/dnadesign/latentdna/tests/contracts/test_workspace_config.py`
  - `MPLCONFIGDIR=/tmp/latentdna_mpl uv run pytest -q src/dnadesign/latentdna/tests/integrations/test_phase6_recipe_deliverable_workflow.py src/dnadesign/latentdna/tests/integrations/test_freshness_contracts.py src/dnadesign/latentdna/tests/contracts/test_workspace_config.py`
- Red-green notes:
  - `test_recipe_run_rebuilds_stale_view_after_overlay_change` failed first because `recipe run materialize_only` still reported `executed_steps == 0` and skipped the stale view artifact
  - `test_load_workspace_config_rejects_unknown_config_backed_deliverable_output` failed first because loader validation only checked `requires`, not config-backed `outputs`
- Live promoter-study commands:
  - `MPLCONFIGDIR=/tmp/latentdna_mpl uv run latentdna workspace init --workspace /tmp/latentdna_stress_ethanol_cipro_pressure_v3 --template landmark_atlas_committee --from-study-dir docs/studies/stress_ethanol_cipro_growth --json`
  - `MPLCONFIGDIR=/tmp/latentdna_mpl uv run latentdna validate workspace --workspace /tmp/latentdna_stress_ethanol_cipro_pressure_v3 --deep --json`
  - `MPLCONFIGDIR=/tmp/latentdna_mpl /usr/bin/time -l uv run latentdna view materialize z7_60 --workspace /tmp/latentdna_stress_ethanol_cipro_pressure_v3 --json`
  - `MPLCONFIGDIR=/tmp/latentdna_mpl uv run latentdna sample build z7_60_sample20k --workspace /tmp/latentdna_stress_ethanol_cipro_pressure_v3 --view z7_60 --strategy random --n 20000 --seed 17 --json`
  - `MPLCONFIGDIR=/tmp/latentdna_mpl /usr/bin/time -l uv run latentdna projection fit z7_60 --workspace /tmp/latentdna_stress_ethanol_cipro_pressure_v3 --sample z7_60_sample20k --run-id umap_z7_60_sample20k --metric cosine --seed 17 --json`
- Timing notes:
  - the live `z7_60` materialization completed in about `55.13` seconds wall time
  - the 20k sampled UMAP projection completed in about `31.97` seconds wall time
  - `/usr/bin/time -l` still ends with a non-zero status in this sandbox because its final `sysctl kern.clockrate` probe is not permitted, but the wrapped latentdna commands themselves completed successfully and emitted success JSON before that timing-wrapper error

### Next steps

1. Re-run the corresponding live `z20_60` path (`view materialize -> sample build -> projection fit`) so the primary 20B lane has the same real pressure evidence as `z7_60`.
2. Materialize the paired context views and re-test `alignment build anchor_ctx_7b` / `anchor_ctx_20b` on real study data now that the first live primary view path is no longer blocked at materialization.
3. Extend freshness-aware rebuild/readiness beyond recipe skip logic into snapshots, run inventory, and deliverable reporting so overlay-backed stale inputs are handled consistently across the whole workspace surface.
4. Decide whether per-part overlay provenance should stay fully expanded in manifests or gain a summarized companion artifact once the live study path is regularly producing 500KB+ manifests.
5. After the live primary/context paths are pressure-tested, return to the remaining breadth gaps: CLI common-flag harmonization, additional derived-view kinds, and any benchmark harness slices still missing from the build spec.

## 2026-04-11 - Matrix Bundle, Multiview Breadth, and Package Gates Slice

### Objective

Close the next spec-parity gaps exposed by the swarm audit without weakening any contracts:

- make `matrix_bundle` a real source kind in the main `view materialize` path
- widen the runtime surface to the next required derived-view, scalar, and sample-set operations
- promote the checked-in committee template and docs surface from one narrow path to the broader deliverable set
- add package boundary and benchmark-harness gates instead of leaving those slices as deferred placeholders

### Implemented in this slice

- `matrix_bundle` is now usable in the core loop:
  - source-backed views may declare `vector.kind: bundle_matrix`
  - materialization validates `rows.parquet` plus `matrix.npy` or `matrix.npz`
  - the bundle is canonicalized into workspace-owned `views/<id>/matrix.npy` and `rows.parquet`
- Derived-view breadth now includes:
  - `normalize`
  - `aggregate_by_key`
  - `apply_reducer`
  - `concatenate`
- Scalar breadth now includes:
  - `select_columns`
  - `rename_columns`
- Sample breadth now includes:
  - `union`
  - `intersection`
- Deliverable validation is stricter:
  - declared `outputs` must now be produced by the linked recipe, including config-backed outputs such as `views`, `scalars`, `reducers`, `reduced_views`, and `exports`
- Freshness reporting is more truthful for aligned and export-heavy flows:
  - alignment manifests now record concrete input paths
  - export manifests now record row-basis, block, and alignment paths alongside digests
  - deliverable status can therefore report `ok` for the full synthetic aligned export lane instead of degrading to `attention` because provenance was incomplete
- The checked-in promoter-study committee template now exposes the broader deliverable surface:
  - `atlas_2x2_intermediate`
  - `control_neighborhood_enrichment`
  - `control_distance_margins`
  - `context_shift_primary`
  - `drag_qc`
  - `agreement_7b_vs_20b`
  - `x0_primary_20b`
  - `x1_primary_20b`
  - `x2_primary_20b`
  - `x3_ablation_7b`
- The docs router/reference surface now includes workflow and contract pages for context shift, agreement, exports, source/view/alignment/scalar/deliverable contracts, manifests, and performance budgets.
- Added new hardening gates:
  - package import-boundary test coverage
  - a fixture-scale benchmark harness that emits wall time, throughput, peak RSS, artifact size, and correctness summaries
  - a synthetic end-to-end primary export deliverable test that exercises materialize -> alignment -> delta -> scalar -> distance -> reduce -> export

### Constraints and deliberate deferrals

- The benchmark harness is fixture-scale smoke coverage, not a live-study performance claim.
- The real promoter-study primary/context reruns beyond the earlier `z7_60` pressure evidence were not repeated in this slice.
- Package-wide CLI `--quiet` / `--dry-run` harmonization still remains open.

### Validation notes

- Green targeted checks:
  - `uv run pytest -q src/dnadesign/latentdna/tests/contracts/test_workspace_config.py`
  - `uv run pytest -q src/dnadesign/latentdna/tests/package/test_source_tree_contracts.py src/dnadesign/latentdna/tests/package/test_dependency_boundaries.py src/dnadesign/latentdna/tests/integrations/test_phase15_primary_export_workflow.py src/dnadesign/latentdna/tests/perf/test_benchmark_harness.py`
- Red-green notes:
  - the new multiview breadth tests failed first because the workspace schema and runtime only supported `vector_difference`, narrow scalar ops, and no sample-set unions/intersections
  - the new primary export deliverable test failed first because alignment and export manifests recorded digests without path-backed provenance, causing deliverable freshness to degrade to `attention`

### Next steps

1. Re-run the real primary/context promoter-study lanes so the widened deliverable surface has matching live pressure evidence.
2. Decide whether more artifact input kinds should be freshness-resolvable by semantic kind as well as by path-backed digests.
3. Return to the remaining advisory CLI surface gap: package-wide `--quiet` / `--dry-run`.

## 2026-04-11 - CLI Surface, View Stats, and Scalar Join Slice

### Objective

Close the next spec-parity gaps that were still visible after the primary export slice:

- harmonize the documented CLI common flags across the implemented command surface
- add the missing `view stats` primitive instead of leaving it as a docs-only contract
- finish scalar-table breadth with explicit inner joins over persisted row-basis tables

### Implemented in this slice

- The mutating CLI surface now supports real `--dry-run` previews with structured `latentdna.command_result.v1` output and `dry_run: true` markers for:
  - `workspace init`
  - `snapshot build`
  - `alignment build`
  - `view materialize|derive|reduce`
  - `scalar derive`
  - `sample build`
  - `neighbors fit`
  - `cluster fit`
  - `projection fit`
  - `distance score`
  - `enrich score`
  - `agreement compare`
  - `plot render`
  - `export matrix|table`
  - `notebook generate`
  - `recipe run`
  - `deliverable run`
  - `runs prune`
- Text-mode `--quiet` is now wired across the CLI command tree, and command-result payloads collapse to a one-line status summary instead of dumping the full key/value body.
- Added the missing `latentdna view stats` command plus runtime support for:
  - persisted `view` artifacts
  - persisted `reduced_view` artifacts
  - norm and missing-value summaries
  - reducer-backed `explained_variance_ratio` reporting when a reducer summary is available
- `scalar derive` now supports `join_tables`:
  - joins two or more persisted scalar/distance tables
  - requires explicit key columns
  - enforces unique keys per source
  - preserves deterministic row order from the first source
  - fails on duplicate non-key column reuse instead of silently clobbering columns
- Reference docs were updated so the checked-in CLI/workspace/scalar contracts match the implemented surface again, including:
  - `src/dnadesign/latentdna/docs/reference/cli-contracts.md`
  - `src/dnadesign/latentdna/docs/reference/scalar-contract.md`
  - `src/dnadesign/latentdna/docs/reference/workspace-schema.md`

### Constraints and deliberate deferrals

- The current dry-run surface is intentionally shallow: it validates command targets and reports expected output paths, but it does not try to execute heavy runtime prerequisites in a side-effect-free shadow mode.
- `view stats` currently reports reducer-backed explained variance when the view or reduced-view manifest points at a reducer summary, but it does not yet compute deeper descriptive diagnostics such as per-dimension variance tables.
- This slice does not re-run the real promoter-study 20B/context lanes; it closes the remaining local contract gaps so those operational reruns are no longer blocked by CLI/docs drift.

### Validation notes

- Red gates:
  - the new workspace contract test initially failed because `join_tables` was still missing from `latentdna.workspace.v1`
  - the new phase-16 integration test initially failed because `view materialize --dry-run` was not a recognized option and `view stats` did not exist
  - scalar join then failed again because both the derive runtime and service manifest path still assumed every non-`vector_norm` scalar had a single `source`
- Green targeted checks:
  - `uv run ruff check src/dnadesign/latentdna`
  - `uv run pytest -q src/dnadesign/latentdna/tests/cli/test_workspace_command.py src/dnadesign/latentdna/tests/contracts/test_workspace_config.py src/dnadesign/latentdna/tests/integrations/test_phase6_recipe_deliverable_workflow.py src/dnadesign/latentdna/tests/integrations/test_phase7_notebook_workflow.py src/dnadesign/latentdna/tests/integrations/test_phase13_plot_diagnostics_workflow.py src/dnadesign/latentdna/tests/integrations/test_phase15_primary_export_workflow.py src/dnadesign/latentdna/tests/integrations/test_phase16_stats_cli_surface_workflow.py src/dnadesign/latentdna/tests/test_docs_contract.py`
- Verification caveat:
  - `uv run ruff format --check src/dnadesign/latentdna` still reports pre-existing formatting drift in unrelated latentdna files outside this slice, so this slice stayed scoped instead of widening into unrelated formatting churn.

### Next steps

1. Re-run the real promoter-study primary/context lanes so the now-complete CLI surface has matching live pressure evidence, not just fixture coverage.
2. Decide whether dry-run should remain a shallow contract preview or grow deeper prerequisite validation for selected heavy commands.
3. Continue tightening freshness/readiness truthfulness for snapshots, run inventory, and deliverable reporting where overlay-backed inputs still degrade into advisory warnings.

## 2026-04-11 - Downstream Freshness Truthfulness Slice

### Objective

Tighten the post-phase-16 readiness surface so downstream artifacts stop degrading into advisory `attention` states when their upstream inputs are actually fresh:

- table-derived scalar artifacts should report freshness through their real upstream artifact kinds
- `agreement_summary` plots should report freshness through the persisted `agreement_set` they summarize
- `runs list|show` and `deliverable status|run` should therefore stay truthful for these downstream outputs

### Implemented in this slice

- Added phase-17 integration coverage for a synthetic downstream bundle that exercises:
  - a `column_expression` scalar derived from a persisted distance table
  - a `distribution` plot over that scalar
  - an `agreement_summary` plot over a persisted agreement artifact
  - `deliverable run`, `deliverable status`, and `runs list` readiness checks over the resulting outputs
- `scalar derive` manifest inputs now resolve table-backed dependencies to their canonical artifact kinds:
  - `scalar_table` when the source is another scalar artifact
  - `distance_set` when the source is a persisted distance artifact
- Those scalar manifest inputs now also record concrete `table.parquet` paths, so freshness can be evaluated by both path-backed digest checks and upstream artifact recursion.
- `plot render` manifest inputs now record canonical artifact kinds and path-backed provenance for:
  - `heatmap`
  - `distance_scatter`
  - `distribution`
  - `agreement_summary`
  - projection-backed scatter/grid plots
- `agreement_summary` plots now point at the owning `agreement_set` rather than a one-off pseudo-kind, so downstream readiness no longer falls back to advisory unknowns.

### Constraints and deliberate deferrals

- This slice improves downstream truthfulness for scalar- and plot-backed diagnostics, but it does not yet widen deliverable categories to include snapshots or other new runtime-only artifact families.
- It also does not change the shallow `--dry-run` contract or add new live promoter-study reruns; this is a freshness/readiness hardening pass only.
- Landmark-only agreement freshness still depends on the existing source-provenance surface; this slice focused on the table/plot downstream seams that phase 16 left advisory.

### Validation notes

- Red gate:
  - the new phase-17 integration test initially failed because `deliverable run downstream_freshness_bundle` returned `status: attention` even though every upstream artifact had been freshly materialized
  - the root causes were:
    - `scalar derive` recorded upstream tables as generic `table` inputs, which freshness could not recurse through
    - `plot render --kind agreement_summary` recorded a non-canonical `agreement_summary` input kind instead of the owning `agreement_set`
- Green targeted checks:
  - `uv run ruff check src/dnadesign/latentdna/src/services/scalar_service.py src/dnadesign/latentdna/src/services/plot_service.py src/dnadesign/latentdna/tests/integrations/test_phase17_downstream_freshness_workflow.py`
  - `uv run pytest -q src/dnadesign/latentdna/tests/integrations/test_phase6_recipe_deliverable_workflow.py src/dnadesign/latentdna/tests/integrations/test_freshness_contracts.py src/dnadesign/latentdna/tests/integrations/test_phase13_plot_diagnostics_workflow.py src/dnadesign/latentdna/tests/integrations/test_phase16_stats_cli_surface_workflow.py src/dnadesign/latentdna/tests/integrations/test_phase17_downstream_freshness_workflow.py`

### Next steps

1. Re-run the real promoter-study primary/context lanes so the broadened freshness/readiness surface has matching live pressure evidence, not just synthetic workflow coverage.
2. Decide whether snapshot artifacts should become first-class deliverable outputs now that their freshness surface is implemented independently of config-backed declarations.
3. Continue tightening readiness truthfulness anywhere manifests still rely on advisory-only provenance instead of canonical artifact kinds plus path-backed digests.
4. Decide whether `--dry-run` should remain a shallow contract preview or grow deeper prerequisite validation for selected heavy commands.

## 2026-04-11 - Real Downstream Pressure Follow-Up

### Objective

Pressure test the new downstream freshness/readiness slice on the active promoter-study workspace instead of stopping at synthetic fixtures:

- rerun the live 20B primary view path that phase 16 had not yet exercised
- prove the real downstream scalar and agreement deliverables can land in `ok`
- record the next true blockers or drift after the freshness truthfulness work met live data

### Implemented in this slice

- Reduced memory pressure in the USR multi-part overlay scan path used by `latentdna view materialize`:
  - the overlay builder no longer materializes a wide temporary staging table before the per-key collapse
  - unbounded overlay scans now force `threads=1` and `preserve_insertion_order=false` to match DuckDB's lower-memory guidance for wide full-table reads
- Added USR regressions for both runtime changes:
  - multi-part overlay staging stays lazy instead of writing a temp table
  - unbounded overlay scans apply the low-memory DuckDB settings
- Fixed live promoter-study template drift uncovered by the real run:
  - `spy_p` now selects `usr_label__primary == spyp` in the checked-in `landmark_atlas_committee` template
  - the checked-in `agreement_7b_vs_20b` recipe no longer asks for landmark-neighborhood overlap on a sampled support that can legitimately omit the single control rows in the live study
- Extended freshness resolution so notebook artifacts can recurse through canonical upstream `agreement_set` and `export_bundle` inputs instead of falling back to advisory unknowns
- Tightened phase-17 synthetic coverage so the downstream freshness test now includes a notebook review artifact over an `agreement_set` plus `agreement_summary` plot

### Validation notes

- Red gates from the live pressure pass:
  - `deliverable run control_distance_margins` initially failed with a DuckDB OOM while materializing `z20_60`
  - after the 20B view started materializing, the same deliverable exposed live label drift: the active control row is `spyp`, not `spyP`
  - `deliverable run agreement_7b_vs_20b` then exposed a template/workflow mismatch: sampled agreement support did not reliably contain the single live control rows needed for landmark-neighborhood overlap
  - after the agreement recipe was trimmed, the remaining `attention` came from notebook freshness not recognizing `agreement_set` as a canonical upstream kind
- Green targeted checks:
  - `uv run ruff check src/dnadesign/usr/src/dataset_query.py src/dnadesign/usr/src/dataset_overlay_query.py src/dnadesign/usr/tests/test_dataset_scan_projection.py src/dnadesign/usr/tests/test_overlays.py src/dnadesign/latentdna/src/services/freshness_service.py src/dnadesign/latentdna/tests/integrations/test_phase17_downstream_freshness_workflow.py`
  - `uv run ruff format --check src/dnadesign/usr/src/dataset_query.py src/dnadesign/usr/src/dataset_overlay_query.py src/dnadesign/usr/tests/test_dataset_scan_projection.py src/dnadesign/usr/tests/test_overlays.py src/dnadesign/latentdna/src/services/freshness_service.py src/dnadesign/latentdna/tests/integrations/test_phase17_downstream_freshness_workflow.py`
  - `uv run pytest -q src/dnadesign/usr/tests/test_dataset_scan_projection.py src/dnadesign/usr/tests/test_overlays.py -k 'low_memory_settings or materialized_staging_table'`
  - `uv run pytest -q src/dnadesign/latentdna/tests/integrations/test_phase7_notebook_workflow.py src/dnadesign/latentdna/tests/integrations/test_phase17_downstream_freshness_workflow.py`
- Live promoter-study commands and outcomes:
  - `MPLCONFIGDIR=/tmp/latentdna_mpl uv run latentdna workspace init --workspace /tmp/latentdna_phase17_pressure_20260411 --template landmark_atlas_committee --from-study-dir docs/studies/stress_ethanol_cipro_growth --json`
  - `MPLCONFIGDIR=/tmp/latentdna_mpl uv run latentdna validate workspace --workspace /tmp/latentdna_phase17_pressure_20260411 --deep --json`
  - `MPLCONFIGDIR=/tmp/latentdna_mpl /usr/bin/time -p uv run latentdna view materialize z20_60 --workspace /tmp/latentdna_phase17_pressure_20260411 --json`
    - succeeded after the USR scan fixes with `157164` rows, `8192` dims, and about `89.42s` wall time
  - `MPLCONFIGDIR=/tmp/latentdna_mpl /usr/bin/time -p uv run latentdna deliverable run control_distance_margins --workspace /tmp/latentdna_phase17_pressure_20260411 --json`
    - succeeded with `status: ok` in about `64.98s`
  - `MPLCONFIGDIR=/tmp/latentdna_mpl /usr/bin/time -p uv run latentdna deliverable run agreement_7b_vs_20b --workspace /tmp/latentdna_phase17_pressure_20260411 --json`
    - after the template adjustments, produced the sampled agreement artifacts plus review notebook
  - `MPLCONFIGDIR=/tmp/latentdna_mpl uv run latentdna deliverable status control_distance_margins --workspace /tmp/latentdna_phase17_pressure_20260411 --json`
    - returned `status: ok`
  - `MPLCONFIGDIR=/tmp/latentdna_mpl uv run latentdna deliverable status agreement_7b_vs_20b --workspace /tmp/latentdna_phase17_pressure_20260411 --json`
    - returned `status: ok`
  - `MPLCONFIGDIR=/tmp/latentdna_mpl uv run latentdna runs list --workspace /tmp/latentdna_phase17_pressure_20260411 --json`
    - now reports the live `agreement_7b_vs_20b_review` notebook, `agreement_7b_vs_20b_summary` plot, `agreement_7b_vs_20b` artifact, and the control-distance scalar/plot artifacts all as `ok`

### Next steps

1. Re-run the remaining live context/export lanes (`z20_1k_anchor`, `z7_1k_anchor`, alignments, delta views, and `x0/x1/x2/x3`) now that the 20B primary path and downstream readiness surface are no longer blocked.
2. Decide whether landmark-neighborhood overlap on agreement lanes should return with a guaranteed-control sampling strategy or remain out of the checked-in sampled agreement deliverable.
3. Reduce the cost of full live freshness recomputation for `deliverable status` and `runs list`; the large real overlay provenance surface is now truthful but noticeably expensive to walk.
4. Decide whether notebook manifests should also record path-backed upstream artifact inputs, even though canonical-kind recursion is now sufficient for correctness.

## 2026-04-11 - Freshness Cache and Readiness Fan-Out Slice

### Objective

Reduce the cost of the now-truthful readiness surface without weakening freshness correctness:

- stop `deliverable status` from re-hashing the same path-backed provenance repeatedly within one status call
- stop `runs list` from re-walking and re-hashing shared upstream artifact trees for every listed artifact
- keep `recipe run` on the same cache path so freshness-aware skip/rebuild checks do not regress as the real workspace gets larger

### Implemented in this slice

- Added a per-call `FreshnessCache` in `src/dnadesign/latentdna/src/services/freshness_service.py` that memoizes:
  - path digest lookups
  - recursive artifact freshness results
- `evaluate_artifact_freshness(...)` now reuses cached results for already-checked artifact ids instead of recursively recomputing the same dependency chain.
- Path-backed freshness checks now resolve each concrete path at most once per cache-bearing call, including both:
  - `source_provenance`
  - manifest `inputs[*].path`
- Threaded one shared freshness cache through:
  - `latentdna deliverable status`
  - `latentdna runs list`
  - `recipe run` freshness-aware skip/rebuild checks
- Added integration regressions proving the optimization on a real artifact chain (`view -> sample -> projection`):
  - `test_deliverable_status_hashes_shared_freshness_paths_once_per_call`
  - `test_runs_list_hashes_shared_freshness_paths_once_per_call`

### Validation notes

- Red gate:
  - the new caching regressions failed first because one `deliverable status` or `runs list` call could hash the same shared provenance paths up to eight times while walking `view`, `sample`, and `projection` outputs
- Green targeted checks:
  - `uv run ruff check src/dnadesign/latentdna/src/services/freshness_service.py src/dnadesign/latentdna/src/services/deliverable_service.py src/dnadesign/latentdna/src/services/run_service.py src/dnadesign/latentdna/src/services/recipe_service.py src/dnadesign/latentdna/tests/integrations/test_freshness_contracts.py`
  - `uv run ruff format --check src/dnadesign/latentdna/src/services/freshness_service.py src/dnadesign/latentdna/src/services/deliverable_service.py src/dnadesign/latentdna/src/services/run_service.py src/dnadesign/latentdna/src/services/recipe_service.py src/dnadesign/latentdna/tests/integrations/test_freshness_contracts.py`
  - `uv run pytest -q src/dnadesign/latentdna/tests/integrations/test_freshness_contracts.py src/dnadesign/latentdna/tests/integrations/test_phase6_recipe_deliverable_workflow.py src/dnadesign/latentdna/tests/integrations/test_phase17_downstream_freshness_workflow.py`
- Verification caveat:
  - the existing UMAP warning from the small projection-based workflow fixtures remains non-failing and unchanged by this slice

### Next steps

1. Re-run the remaining live context/export lanes (`z20_1k_anchor`, `z7_1k_anchor`, alignments, delta views, and `x0/x1/x2/x3`) now that the heavy readiness surfaces are cheaper to query between steps.
2. Decide whether landmark-neighborhood overlap on sampled agreement lanes should return with a guaranteed-control sampling strategy or remain out of the checked-in sampled agreement deliverable.
3. Measure the real active-study improvement explicitly by timing `deliverable status` and `runs list` before/after the cache-bearing surface on the live promoter-study workspace.
4. Decide whether notebook manifests should also record path-backed upstream artifact inputs, even though canonical-kind recursion is already sufficient for correctness.

## 2026-04-11 - Interactive Marimo Notebook Viewer Slice

### Objective

Upgrade the notebook review surface from a passive Python scaffold to an interactive marimo app so persisted plot artifacts are actually viewable inline:

- emit real marimo notebooks from `latentdna notebook generate`
- keep the notebook surface read-only over persisted artifacts
- render plot files inline while still exposing manifests, table previews, and array summaries
- pressure test the generated app on a real fixture notebook instead of stopping at text generation

### Implemented in this slice

- `src/dnadesign/latentdna/src/notebooks/scaffold.py` now emits a marimo app instead of a plain Python script:
  - `__generated_with`
  - `marimo.App(width="full")`
  - reactive artifact picker UI
  - inventory table over the declared notebook artifacts
  - manifest, file, parquet-preview, and array-summary tabs
  - inline rendering for plot-backed `svg`, `png`, `jpg`, `jpeg`, `webp`, and `html` outputs
- The generated notebook still keeps the same read-only contract:
  - workspace-relative artifact resolution only
  - no hidden recomputation of views, samples, projections, or plots
  - helper `load_artifact(...)` remains available inside the notebook
- `src/dnadesign/latentdna/src/services/notebook_service.py` now records notebook runtime metadata as `runtime: marimo` in the notebook manifest params.
- Phase-7 integration expectations now assert the generated notebook contains the marimo runtime surface, plot renderer helper, and artifact picker wiring.
- Workflow/reference docs now describe the notebook artifact as an interactive marimo review app and include the `uv run marimo run .../notebook.py` launch path.

### Validation notes

- Red gates:
  - `uvx marimo check` first failed on `multiple-definitions` because the generated notebook leaked repeated top-level cell globals such as `artifact`
  - the first browser pressure pass in Chrome DevTools exposed a runtime/UX bug: the artifact dropdown used label-mapped options but initialized with the alias value, so marimo raised `The option name 'z20_60' is not a valid option`
- Green targeted checks:
  - `uv run ruff check src/dnadesign/latentdna/src/notebooks/scaffold.py src/dnadesign/latentdna/src/services/notebook_service.py src/dnadesign/latentdna/tests/integrations/test_phase7_notebook_workflow.py src/dnadesign/latentdna/tests/test_docs_contract.py`
  - `uv run ruff format --check src/dnadesign/latentdna/src/notebooks/scaffold.py src/dnadesign/latentdna/src/services/notebook_service.py src/dnadesign/latentdna/tests/integrations/test_phase7_notebook_workflow.py src/dnadesign/latentdna/tests/test_docs_contract.py`
  - `uv run pytest -q src/dnadesign/latentdna/tests/integrations/test_phase7_notebook_workflow.py src/dnadesign/latentdna/tests/test_docs_contract.py`
  - `uv run python -m dnadesign.devtools.docs_checks`
  - `uvx marimo check /tmp/latentdna_marimo_ui.MyW53d/workspace/outputs/notebooks/atlas_review/notebook.py`
- Browser/runtime evidence:
  - generated a real fixture notebook via `uv run latentdna deliverable run atlas_review_bundle --workspace /tmp/latentdna_marimo_ui.MyW53d/workspace --json`
  - launched it with `MPLCONFIGDIR=/tmp/latentdna_mpl uv run marimo run /tmp/latentdna_marimo_ui.MyW53d/workspace/outputs/notebooks/atlas_review/notebook.py --headless --host 127.0.0.1 --port 27189`
  - the first Chrome DevTools pass caught the invalid dropdown initialization bug above
  - after switching the picker to alias-valued options, regenerating the notebook, and re-running `marimo check`, the notebook launched cleanly with no follow-on server-side execution errors observed during the fixed app run

### Next steps

1. Run the same marimo review flow against the checked-in committee notebooks (`atlas_committee_review`, `control_plan_review`, `agreement_7b_vs_20b_review`, `x2_primary_20b_review`) once the remaining live context/export lanes are materialized.
2. Add one automated notebook smoke gate beyond `marimo check`, so a rendered fixture notebook catches UI-contract regressions like invalid selector initialization before a manual browser pass.
3. Resume the remaining live context/export reruns (`z20_1k_anchor`, `z7_1k_anchor`, alignments, deltas, and `x0/x1/x2/x3`) now that downstream plot artifacts have a usable interactive review surface.
4. Decide whether notebook manifests should grow explicit path-backed upstream inputs for audit/debug readability, even though canonical artifact-kind recursion is already sufficient for freshness correctness.

## 2026-04-12 - Workspace Plot Browser Follow-On Slice

### Objective

Finish the first marimo review surface so it can act as a durable plot browser instead of only replaying the notebook-declared artifact list:

- keep the declared-artifact review path unchanged
- make every persisted `plot` artifact under `outputs/plots` viewable from the notebook UI
- confirm the updated marimo UX in a real browser pass rather than relying on static text generation alone

### Implemented in this slice

- Extended `src/dnadesign/latentdna/src/notebooks/scaffold.py` so generated notebooks now:
  - scan `outputs/plots` at runtime for plot artifacts with `manifest.json`
  - expose a dedicated `Workspace plots` browser tab alongside the existing declared-artifact tab
  - render discovered plot artifacts with the same manifest/file/preview surface used for declared artifacts
  - keep the scan read-only, so newly rendered plot artifacts appear without regenerating the notebook
- Tightened the phase-7 regression in `src/dnadesign/latentdna/tests/integrations/test_phase7_notebook_workflow.py`:
  - the fixture notebook no longer declares the plot artifact directly
  - the generated notebook contract is now expected to include the workspace-plot discovery UI and scan path instead
- Updated the reference/workflow docs to advertise the runtime plot-browser behavior in:
  - `docs/reference/cli-contracts.md`
  - `docs/reference/workspace-schema.md`
  - `docs/workflows/promoter-study-latent-atlas.md`

### Validation notes

- Green targeted checks:
  - `uv run ruff check src/dnadesign/latentdna/src/notebooks/scaffold.py src/dnadesign/latentdna/tests/integrations/test_phase7_notebook_workflow.py src/dnadesign/latentdna/tests/test_docs_contract.py`
  - `uv run ruff format --check src/dnadesign/latentdna/src/notebooks/scaffold.py src/dnadesign/latentdna/tests/integrations/test_phase7_notebook_workflow.py src/dnadesign/latentdna/tests/test_docs_contract.py`
  - `uv run pytest -q src/dnadesign/latentdna/tests/integrations/test_phase7_notebook_workflow.py src/dnadesign/latentdna/tests/test_docs_contract.py`
  - `uv run python -m dnadesign.devtools.docs_checks`
  - `uv run marimo check /tmp/latentdna_marimo_workspace.BQpUW0/workspace/outputs/notebooks/atlas_review/notebook.py`
- Browser/runtime evidence:
  - generated a fixture notebook whose declared artifacts omit the plot artifact via:
    - `MPLCONFIGDIR=/tmp/latentdna_mpl uv run latentdna deliverable run atlas_review_bundle --workspace /tmp/latentdna_marimo_workspace.BQpUW0/workspace --json`
  - rendered an additional plot artifact into the same workspace via:
    - `MPLCONFIGDIR=/tmp/latentdna_mpl uv run latentdna plot render atlas_secondary_plot --workspace /tmp/latentdna_marimo_workspace.BQpUW0/workspace --kind projection_scatter --projection umap_z20_60 --json`
  - launched the regenerated notebook in app mode via:
    - `MPLCONFIGDIR=/tmp/latentdna_mpl uv run marimo run /tmp/latentdna_marimo_workspace.BQpUW0/workspace/outputs/notebooks/atlas_review/notebook.py --headless --host 127.0.0.1 --port 27189`
  - the first Chrome DevTools pass caught a real marimo runtime error:
    - `RuntimeError: Accessing the value of a UIElement in the cell that created it is not allowed`
    - fixed by splitting the workspace-plot picker creation and value consumption into separate cells
  - after regeneration, Chrome DevTools confirmed:
    - the `Workspace plots` browser copy is present
    - both `atlas_demo_plot` and `atlas_secondary_plot` are visible in the runtime plot inventory
    - rendered plot files (`plot.svg` and `plot.png`) are present in the notebook UI surface
    - the prior notebook-internal error notifications disappeared; remaining console noise was limited to marimo runtime warnings/issues outside the latentdna notebook logic

### Next steps

1. Add one automated notebook smoke gate beyond `marimo check` that exercises the generated `Workspace plots` tab against a real rendered plot artifact.
2. Run the same browser-validated plot-browsing flow against the checked-in committee notebooks once the remaining live context/export lanes are materialized.
3. Resume the remaining live context/export reruns (`z20_1k_anchor`, `z7_1k_anchor`, alignments, deltas, and `x0/x1/x2/x3`) now that the notebook can browse all persisted plot artifacts in one place.
4. Decide whether notebook manifests should grow explicit path-backed upstream inputs for audit/debug readability, even though runtime plot discovery is intentionally broader than the declared notebook artifact list.

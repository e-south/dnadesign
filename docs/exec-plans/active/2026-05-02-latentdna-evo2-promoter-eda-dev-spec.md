## Exec plan: LatentDNA Evo2 promoter EDA dev spec

**Status:** active
**Owner:** Shockwing / Codex handoff
**Created:** 2026-05-02
**Last updated:** 2026-05-02

### Purpose / Big Picture

LatentDNA needs to become the study-agnostic exploration layer for Evo2-derived
sequence representations without baking promoter biology, current study names,
or temporary notebook behavior into package primitives. The immediate product
outcome is a small, high-signal Marimo browser and artifact set for two current
promoter studies:

- `stress_ethanol_cipro_growth`, the large pre-assay representation-triage study
  with DenseGen anchors, SFXI pDual-10 rows, MG1655 native references, Anderson
  iGEM standards, W collection standards, 60 bp anchors, 1 kb contexts, reverse-
  complement contexts, intermediate embeddings, output-layer means, and log-
  likelihood scalar diagnostics at varying completion states.
- `regulondb_native_promoter_panel`, the smaller native/core60 study with 3,182
  native source records, 3,181 canonical core60 rows, completed Evo2 7B native
  and core60 vector/scalar sidecars, and categorical RegulonDB metadata such as
  sigma factor sets.

Worth-doing preflight: the best-case result is not "more plots." The best case
is a review surface that makes it hard to choose a collapsed or misleading
candidate X, shows when context, length, or pooling changes move records onto
different geometry, and leaves an executor with explicit contracts for extending
the same workflow to non-promoter sequences or non-Evo2 models. If that cannot
be achieved without promoter-specific runtime branches, the work should stop
and narrow the package contract first.

### Progress

- [x] (2026-05-02 13:33Z) Drafted handoff spec from current study records, LatentDNA workspace configs, notebook scaffold, plot semantics, and package contracts.
- [x] (2026-05-02 13:33Z) Executor begins baseline inventory and validates current workspace/materialization state before code changes.
- [x] (2026-05-02 13:33Z) Slice 1: tighten notebook/control-plane contract with canonical `geometry_browser`, no legacy surface aliases, output-layer labels, and candidate representation metadata.
- [x] (2026-05-02 13:33Z) Slice 2: make `representation_health_summary` report configured planned/unavailable vector candidates as `NA` rows with status metadata instead of silently omitting or ranking them.
- [x] (2026-05-02 14:24Z) Slice 3: expand RegulonDB from UMAP-only review to a compact review path with representation health, native/core60 paired shift, sigma-factor cohort structure, appendix UMAP orientation, and regenerated Marimo notebook outputs.
- [x] (2026-05-02 14:24Z) Removed old appendix geometry naming instead of carrying aliases: `appendix_geometry_review` is the canonical stress appendix deliverable id.
- [x] (2026-05-02 14:24Z) Validation evidence recorded: both live workspaces deep-validate; RegulonDB `regulondb_review_recipe` ran with 29 executed steps; RegulonDB notebook smoke passed; both generated Marimo notebooks pass `marimo check`.
- [x] (2026-05-02 19:12Z) Slice 4: generalized ordinal-axis scoring from a Sigma-35-only builder into `ordinal_axis_audit`, with Sigma-35 now declared in workspace/template config and numeric-strength axes supported through `axis.rank_column`.
- [x] (2026-05-02 19:47Z) Slice 4 validation: all LatentDNA tests pass; both live workspaces deep-validate; stress `sigma35_ordinal_audit` and `dataset_overview` deliverables refreshed; generated Marimo notebooks pass `marimo check`; RegulonDB notebook smoke passes; stress notebook smoke has only the costly UMAP/reference appendix freshness gap.
- [x] (2026-05-02 16:01Z) Slice 5: generalized stress reference-collapse scoring from broad metadata group columns into config-declared `reference_sets`, with explicit `reference_set_status` rows for absent, incomplete, too-small, and missing-column collections.
- [x] (2026-05-02 16:01Z) Slice 5 validation: stress `reference_alignment_summary_metrics` rebuilt with 141 rows across 9 configured reference sets; stress `appendix_umap_gallery` refreshed with 8 executed and 72 skipped recipe steps; stress notebook generation, notebook smoke, Marimo check, deep workspace validation, and workspace snapshot all pass with every stress deliverable at `freshness: ok`.
- [ ] (2026-05-02 16:01Z) Later slices remain for any OPAL export-facing decisions that require maintainer weighting policy or phenotype-backed active-learning promotion rules.

### Surprises & Discoveries

Observation: the stress workspace is already much closer to the target shape
than the raw request suggests. It has a primary review path with
`dataset_overview`, `representation_health_summary`,
`design_structure_summary`, `sigma35_ordinal_audit`,
`context_robustness_summary`, and `candidate_decision_frontier`, plus appendix
reference and UMAP surfaces.

Evidence: `src/dnadesign/latentdna/workspaces/stress_ethanol_cipro_growth/config.yaml`
defines the current candidate sets, plot semantics, deliverables, and notebook
control defaults; `docs/studies/stress_ethanol_cipro_growth/status.md` says
these primary surfaces are current except for the costly UMAP appendix.

Observation: RegulonDB has complete local 7B sidecars for native/full and
derived core60 lanes, including output-layer means and log-likelihood scalars,
but the checked-in LatentDNA deliverable is still mostly a two-plot sigma UMAP
panel over intermediate embeddings.

Evidence: `docs/studies/regulondb_native_promoter_panel/status.md` reports
`local_infer_complete_7b`; `src/dnadesign/latentdna/workspaces/regulondb_native_promoter_panel/config.yaml`
declares native/core60 intermediate and output-layer views but only renders two
intermediate sigma UMAP plots.

Observation: the generated notebook scaffold already has two useful top-level
surfaces, not three: a plot-review surface and a geometry browser. The review
surface itself has internal `Grid` and `Explore` tabs with progressive
disclosure through `mo.accordion`.

Evidence: `src/dnadesign/latentdna/src/notebooks/scaffold_panels.py` builds
top-level `Review` and `Geometry browser` tabs, and plot cards expose
`At a glance`, `Study notes`, `Guardrails`, `Caption`, `Preprocessing`, `Math`,
`Why this helps choose X`, `Limits`, `Failure modes`, and `Plot details`
accordion sections from plot semantics and study deliverable markdown.

Observation: the stress reference-collapse plot now needs selector-backed
reference-set semantics rather than broad metadata group columns. Broad columns
such as `source_family`, `selection_basis`, and
`promoter_standard__collection_id` can still be useful metadata audits, but
they do not express which named collection was expected, which rows were
matched, or why a collection is absent in a candidate view.

Evidence: `src/dnadesign/latentdna/src/scalars/preassay.py` resolves
`reference_alignment_summary` `reference_sets` through the generic
reference-set contract, and
`src/dnadesign/latentdna/workspaces/stress_ethanol_cipro_growth/config.yaml`
now declares spyP/sulAp, SFXI archive, native MG1655, Anderson iGEM, W
collection, and core60 variants through selector ids.

Observation: the prior stress notebook smoke blocker was artifact freshness,
not notebook source validity. A non-forced
`appendix_umap_gallery` deliverable refresh reused most upstream work and
rebuilt the stale appendix projections/plots plus the notebook output.

Evidence: `latentdna deliverable run appendix_umap_gallery --workspace
stress_ethanol_cipro_growth --json` completed with 8 executed steps and 72
skipped steps; subsequent notebook generation reported 17 plots, 8
deliverables, 10 geometries, and no missing ordered plots.

### Decision Log

Decision: keep LatentDNA runtime primitives study-neutral. Promoter-specific
terms may appear in workspace configs, plot semantics, checked-in study docs,
and metadata columns, but not as package-level control-flow branches.

Rationale: `src/dnadesign/latentdna/AGENTS.md` explicitly says promoter
workspaces are dogfood fixtures, not internal tool semantics. This is the main
boundary that keeps future non-promoter and non-Evo2 sequence work possible.

Date/Author: 2026-05-02 / Codex

Decision: treat log-likelihood outputs as scalar diagnostics, not default
geometry, unless a later supervised/active-learning export explicitly chooses
one-dimensional scalar blocks.

Rationale: LatentDNA can visualize and export scalar diagnostics, but UMAP,
PCA-rank, centroid, and pairwise geometry primitives should operate on vector
representations. This avoids pretending a scalar likelihood is the same
artifact type as mean-pooled intermediate embeddings or output-layer mean
vectors.

Date/Author: 2026-05-02 / Codex

Decision: output-layer mean vectors are legitimate vector candidates only when
their canonical sidecar sources are materialized and validation reports them as
current. Until then they must remain visible as planned or diagnostic, not
silently omitted or promoted.

Rationale: the user wants all collected data visible, but the stress study
record still reports incomplete main output-layer, reverse-complement, and
scalar sidecars. The workspace contract already supports planned roles and
zero-row planned sources; the implementation should make those states explicit.

Date/Author: 2026-05-02 / Codex

Decision: keep the top-level notebook at two tabs and name the second tab
`Geometry browser`. Do not reintroduce a third tab
unless it has a distinct user job and its own acceptance criteria.

Rationale: the existing two-tab shape maps cleanly to the requested workflow:
review the curated plot shortlist first, then customize geometry/hue/reference
overlays. A third surface risks becoming a dumping ground for stale diagnostics.

Date/Author: 2026-05-02 / Codex

Decision: do not add backward-compatibility shims, legacy aliases, or dual-name
acceptance for notebook/control-plane APIs changed in this work. If a name is
wrong, rename it at the contract and update callers, configs, tests, and docs.

Rationale: LatentDNA is still being shaped as the stable study-agnostic
analysis package. Carrying legacy names such as old notebook surface aliases
would create confusing ontology and hidden maintenance cost before there is a
published external API to preserve.

Date/Author: 2026-05-02 / Shockwing + Codex

Decision: rank candidate X spaces with high-dimensional scalar summaries first,
and use UMAP as an orientation/appendix surface only.

Rationale: the current LatentDNA workflow doc already warns not to choose
representations by UMAP aesthetics. UMAP is useful for qualitative inspection
and annotation overlays, but collapse, context stability, ordinal alignment,
reference separation, and metadata enrichment should be measured upstream of
projection artifacts.

Date/Author: 2026-05-02 / Codex

Decision: package-level ordinal scoring is now `ordinal_axis_audit`; study
semantics such as Sigma-35, Anderson strength, or W collection strength belong
in workspace config through `axis.column`, `axis.order_path` or
`axis.rank_column`, and optional metric-id mapping.

Rationale: this preserves the current Sigma-35 deliverable while making the
primitive reusable for non-promoter or non-Sigma ordinal metadata without a
runtime branch keyed to a study-specific axis.

Date/Author: 2026-05-02 / Codex

Decision: reference-collapse summaries should prefer config-declared
`reference_sets` over broad metadata grouping when the study question concerns
named controls, standards, or landmarks.

Rationale: named reference sets encode the expected rows, selector predicates,
labels, and missing-data posture explicitly. That lets LatentDNA report
collection-specific collapse and absence without baking MG1655, Anderson, W
collection, SFXI, or spyP/sulAp into package control flow.

Date/Author: 2026-05-02 / Codex

### Outcomes & Retrospective

Slices 1-5 have shipped as the current implementation pass. The completed
outcome is:

- a narrower, clearer LatentDNA representation ontology;
- a canonical two-surface Marimo notebook contract using `Review` and
  `Geometry browser`;
- explicit planned/unavailable rows in representation-health summaries;
- a richer but still minimal RegulonDB review path beyond UMAP;
- Marimo notebooks generated from config-backed controls with no hard-coded
  study branches;
- explicit planned/missing/current states for every vector and scalar surface;
- tests and docs that keep the package future-proof instead of promoter-bound.
- a generic ordinal-axis audit builder that supports either explicit ordered
  categorical files or numeric rank metadata while the stress workspace maps it
  onto the existing Sigma-35 plot and metric names.
- selector-backed reference-collapse summaries that compare named reference
  collections through workspace config and emit explicit missing/incomplete
  status rows.

Remaining outcomes are intentionally scoped to later work: deeper stress-study
active-learning export policy and weighting choices for combining reference
collapse versus design structure.

### Context and Orientation

Package boundary:

- LatentDNA owns downstream latent-analysis artifacts, not Infer execution, USR
  mutation, or OPAL active-learning campaign logic.
- The package follows thin CLI, thick services, workspace-owned immutable
  artifacts, and config-backed plot/notebook contracts.
- Source adapters must fail fast on absent vector/scalar payloads when an alias
  points at missing data. Planned or retired views can be declared only when the
  validation surface reports that state explicitly.
- Feature vectors and scalar diagnostics live in canonical Infer sidecars.
  Do not reintroduce legacy row-overlay embedding columns.

Primary repo surfaces:

- LatentDNA package: `src/dnadesign/latentdna/`
- LatentDNA workspace schema: `src/dnadesign/latentdna/docs/reference/workspace-schema.md`
- LatentDNA ownership boundary: `src/dnadesign/latentdna/docs/concepts/ownership-boundary.md`
- Stress workspace config:
  `src/dnadesign/latentdna/workspaces/stress_ethanol_cipro_growth/config.yaml`
- RegulonDB workspace config:
  `src/dnadesign/latentdna/workspaces/regulondb_native_promoter_panel/config.yaml`
- Notebook scaffold/runtime:
  `src/dnadesign/latentdna/src/notebooks/scaffold_panels.py`,
  `src/dnadesign/latentdna/src/notebooks/browser_runtime.py`, and
  `src/dnadesign/latentdna/src/notebooks/browser_runtime_plot_review.py`
- Scalar builders:
  `src/dnadesign/latentdna/src/scalars/preassay.py`,
  `src/dnadesign/latentdna/src/scalars/common.py`, and
  `src/dnadesign/latentdna/src/scalars/build.py`
- Geometry math helpers:
  `src/dnadesign/latentdna/src/geometry/`
- Plot specs and rendering:
  `src/dnadesign/latentdna/src/plots/recipes.py` and
  `src/dnadesign/latentdna/src/plots/render.py`
- Plot semantics contract:
  `src/dnadesign/latentdna/src/contracts/plot_semantics.py`
- Study records:
  `docs/studies/stress_ethanol_cipro_growth/status.md` and
  `docs/studies/regulondb_native_promoter_panel/status.md`

Current stress study facts to preserve:

- Declared phase is `infer_batch_preparation`.
- Current active comparison is 7B construct-insert `seq_mean` anchors versus
  7B forward 1 kb context `anchor_mean`.
- `usr_prom_eth_cip_anchor` has 157,279 rows; `construct_prom_eth_cip_context`
  has 314,558 rows.
- Reference core60 and reference context 7B sidecars exist locally and are
  diagnostic until a review promotes them into decision geometry.
- Mean-pooled output-layer vectors and log-likelihood scalar sidecars are
  diagnostics and completion targets, not current default decision geometry.
- Reverse-complement context products must be materialized upstream by USR and
  Construct; LatentDNA must not synthesize reverse complements.

Current RegulonDB study facts to preserve:

- Declared phase is `local_infer_complete_7b`.
- `usr_regulondb_native_promoters` has 3,182 native records.
- `usr_regulondb_native_promoter_core60` has 3,181 canonical 60 bp records and
  3,182 `analysis_window` sequence views.
- Native source records are `source_record` products with
  `orientation=unknown` and `recommended_pooling=seq_mean`.
- Core60 records are TSS-upstream `[0,60)` windows, not -10/-35 centered boxes.
- Sigma metadata uses `regulondb__*` columns and must remain separate from
  stress-study `sig35_variant` metadata.

Scope in:

- Refactor LatentDNA primitives and configs so candidate representation,
  sequence product, pooling, model family, vector/scalar modality, row basis,
  metadata hue, and reference set semantics are explicit and orthogonal.
- Expand the RegulonDB workspace from UMAP-only exploratory review to a compact
  review path that includes representation health, native-vs-core60 paired
  context/length shift, sigma-factor high-dimensional structure, and UMAP
  orientation.
- Update the stress workspace so available and planned intermediate,
  output-layer, reference, context, reverse-complement, and scalar diagnostic
  surfaces are visible with explicit materialization status and a small
  sanctioned plot shortlist.
- Preserve Marimo as the notebook substrate, using reactive globals, native
  controls, generated `notebook.py` plus `controls.json`, and progressive
  disclosure through plot semantics.
- Add tests that fail on study-specific runtime branches, stale plot semantics,
  hidden missing data, and notebook control regressions.

Scope out:

- Running new Evo2 GPU inference.
- Mutating USR datasets or Construct sequence products except through existing
  upstream workflows if the executor explicitly chooses to regenerate local
  artifacts.
- Choosing a phenotype-validated final active-learning X. LatentDNA can produce
  a pre-assay working shortlist only.
- Implementing OPAL active learning or supervised downstream benchmarking.
- Committing large generated outputs or parquet sidecars without explicit
  maintainer approval.
- Adding a large gallery of lookalike plots that do not answer a distinct
  decision question.

### Goal Plugin Handoff

Use this objective if starting a goal-tracked Codex implementation run:

```text
Refactor and extend LatentDNA so the stress_ethanol_cipro_growth and
regulondb_native_promoter_panel workspaces expose a study-agnostic, contract-
driven Evo2 representation EDA surface. Preserve LatentDNA's generic sequence-
representation boundary, keep Marimo as the generated notebook substrate, make
vector versus scalar and current versus planned surfaces explicit, reduce plot
cruft to decision-useful review paths, and validate both workspaces plus the
generated latent_geometry_browser notebooks without committing generated large
artifacts.
```

Executor stop conditions:

- Stop before code changes if current workspace validation contradicts the
  study records in a way that changes the plan's scope.
- Stop before promoting output-layer or log-likelihood surfaces if sidecar
  coverage is missing or stale and cannot be represented explicitly.
- Stop before adding any runtime branch keyed to `stress_ethanol_cipro_growth`,
  `regulondb_native_promoter_panel`, `promoter`, `sigma35`, `Anderson`, or
  `W collection`; those terms belong in config, metadata, and study docs.
- Stop before committing generated output artifacts unless the user approves.

### Plan of Work

Phase 0: baseline inventory and contract confirmation.

Run cheap source-of-truth checks before edits. Capture current workspace
validation, materialized/planned view inventory, candidate sets, deliverable
status, and notebook generation posture for both studies. This phase prevents
the executor from designing against stale memory.

Phase 1: clarify the LatentDNA representation ontology.

Introduce or tighten small contract objects and helper functions that describe
candidate representations by neutral axes:

- `view_id`
- `source_id`
- `modality` as `vector` or `scalar`
- `model_family` such as `evo2_7b`
- `feature_family` such as `intermediate_embedding` or `output_layer_mean`
- `sequence_scope` such as `anchor_60bp`, `native_source_record`,
  `core60_tss_upstream`, `full_context_1kb`, or `reference_context`
- `pooling` such as `seq_mean`, `anchor_mean`, or `core60_mean`
- `orientation`
- `row_basis`
- `role` as `primary`, `appendix`, `diagnostic`, `planned`, or `retired`
- materialization and freshness state

Prefer extending existing `candidate_sets`, view tags, notebook controls, and
workspace validation before adding a new top-level abstraction. If a new type is
needed, keep it under `src/dnadesign/latentdna/src/contracts/` or a service
module, not a root-level barrel API.

Phase 2: split overloaded pre-assay scalar logic into reusable primitives.

`src/dnadesign/latentdna/src/scalars/preassay.py` currently owns many study-
facing metrics. Extract only when it reduces coupling and test size. The target
shape is small generic builders:

- representation health over any vector candidate set;
- paired-view context or length-shift summaries over explicit alignments;
- reference collapse and landmark-neighborhood summaries over config-declared
  reference sets or group columns;
- ordinal-axis audit parameterized by one declared group/order/numeric scale;
- candidate decision frontier over named metric inputs.

Do not make `sigma35` a package primitive. Implement a generic ordinal audit,
then configure Sigma-35, Anderson iGEM strength, W collection strength, and any
future ordinal axis through workspace config and study input files.

Phase 3: expand RegulonDB from UMAP-only to compact high-dimensional review.

Keep the UMAP plots as appendix orientation, but add the minimal decision-useful
surfaces:

- representation health across native/core60 intermediate and output-layer mean
  vector views;
- native source-record versus core60 paired shift summary aligned by parent
  promoter identity;
- sigma-factor structure summary using high-dimensional metrics such as
  centroid separation, kNN label enrichment or purity, and a shuffled-label
  baseline where sample sizes allow it;
- optional log-likelihood distributions or scalar summaries as diagnostics,
  not UMAP geometry.

The main categorical hue is `regulondb__sigma_factor_set`; secondary hues may
include confidence level, metadata completeness, source strata, regulator
composition, and log-likelihood scalar joins when available.

Phase 4: tighten the stress study review path and candidate inventory.

Keep the current primary review order unless evidence shows a plot is
redundant. Make these changes:

- representation health must include all current vector candidates in the
  selected candidate set, not only embeddings, while planned output-layer or
  missing reverse-complement views remain visible as planned/unavailable;
- scree and health plots should rank all materialized vector candidates and
  report the omitted planned candidates explicitly;
- reference collapse should compare Native MG1655, Anderson iGEM, W collection,
  spyP/sulAp, SFXI, and derived core60/context reference sets where rows exist;
- ordinal audit should become a selector-backed surface that can run Sigma-35,
  Anderson strength, or W collection strength without pooling incompatible
  scales;
- context robustness should make the anchor-only, 1 kb sequence mean, 1 kb
  anchor mean, and reverse-complement context relationships explicit by
  alignment, not projection layout.

Phase 5: simplify and harden the Marimo browser.

Keep the generated notebook read-only. Use two top-level tabs:

- `Review`: curated plot shortlist, with a plot selector and internal `Grid`
  and `Explore` tabs. `Explore` should carry the accordion sections sourced
  from plot semantics and study deliverable markdown.
- `Geometry browser`: single/custom view controls for candidate set, layout,
  model, family, context/scope, hue, and reference overlay. This tab is for
  fine-tuned inspection and export-like review, not for ranking by UMAP
  aesthetics.

Use marimo-native controls and the reactive DAG. Do not add imperative callback
state beyond existing `mo.state` usage unless the behavior cannot be expressed
reactively. Keep controls config-backed through `controls.json`; do not make
the notebook source the canonical place where study policy lives.

Phase 6: regenerate only necessary artifacts and record evidence.

Run workspace validation first, then refresh only stale or newly introduced
views, scalars, plots, and notebooks. Generated outputs stay out of git unless
small tracked contract artifacts are already expected by tests or the user
approves.

### Concrete Steps

1. Run baseline inventory commands from the repo root:

   ```sh
   MPLCONFIGDIR=/tmp/dnadesign_mpl uv run latentdna validate workspace --workspace stress_ethanol_cipro_growth --deep --json
   MPLCONFIGDIR=/tmp/dnadesign_mpl uv run latentdna validate workspace --workspace regulondb_native_promoter_panel --deep --json
   MPLCONFIGDIR=/tmp/dnadesign_mpl uv run latentdna workspace snapshot --workspace stress_ethanol_cipro_growth --json
   MPLCONFIGDIR=/tmp/dnadesign_mpl uv run latentdna workspace snapshot --workspace regulondb_native_promoter_panel --json
   uv run latentdna deliverable list --workspace stress_ethanol_cipro_growth --json
   uv run latentdna deliverable list --workspace regulondb_native_promoter_panel --json
   ```

   Record current materialized, planned, missing, stale, and retired states in
   this plan's `Surprises & Discoveries` section before changing code.

2. Inspect current generated notebook posture without treating generated output
   as source:

   ```sh
   MPLCONFIGDIR=/tmp/dnadesign_mpl uv run latentdna notebook generate latent_geometry_browser --workspace stress_ethanol_cipro_growth --dry-run --json
   MPLCONFIGDIR=/tmp/dnadesign_mpl uv run latentdna notebook generate latent_geometry_browser --workspace regulondb_native_promoter_panel --dry-run --json
   ```

   If existing default deliverables are stale, refresh the owning deliverable
   before notebook generation rather than forcing overwrite.

3. Add representation-inventory contract tests before refactoring. Target
   paths:

   - `src/dnadesign/latentdna/tests/contracts/test_workspace_config.py`
   - `src/dnadesign/latentdna/tests/test_notebook_controls_service.py`
   - `src/dnadesign/latentdna/tests/package/test_dependency_boundaries.py`

   Tests should assert that candidate sets expose role/materialization state,
   planned vector views remain visible but are not silently ranked, scalar
   diagnostics are not treated as vector projections, and runtime modules do not
   import study-specific packages.

4. Refactor representation and scalar helper boundaries in small slices. Use
   existing modules first. If extraction is needed, keep new code under existing
   domains such as `contracts`, `services`, `scalars`, `geometry`, or
   `notebooks`. Avoid a new root-level `api.py` or broad facade.

5. Generalize ordinal-axis scoring. Replace Sigma-35-only assumptions with a
   config-driven ordinal axis definition that accepts either ordered categorical
   labels or numeric strength metadata. Preserve the existing Sigma-35 output
   contract through config:

   - stress Sigma-35 f/e/d/c/b ladder from `study_inputs/sig35_order.yaml`;
   - Anderson iGEM strength as its own selected ordinal/numeric group;
   - W collection strength as its own selected ordinal/numeric group.

   Add tests near `src/dnadesign/latentdna/tests/test_scalar_build.py` for
   degenerate groups, mixed scales, missing values, and no pooled incompatible
   strength collections.

6. Add or generalize reference-collapse scoring. The builder should accept
   configured reference-set selectors or group columns and emit group size,
   median pairwise cosine distance, IQR, optional nearest-neighbor rank, and
   missing-data status per candidate view. It must work for stress reference
   collections and for smaller RegulonDB groupings without promoter-specific
   branches.

7. Add paired-view shift summaries for length/context questions. Use explicit
   alignments rather than string matching:

   - stress: anchor vs forward 1 kb `anchor_mean`, forward 1 kb `seq_mean`,
     reverse-complement `anchor_mean`, and reverse-complement `seq_mean` when
     materialized;
   - RegulonDB: native 81 bp source-record `seq_mean` vs TSS-upstream core60
     `core60_mean`, plus output-layer equivalents when materialized.

   Emit distance correlation, paired cosine similarity, rowwise shift
   distribution summaries, and group-conditioned shifts where configured.

8. Update `src/dnadesign/latentdna/workspaces/regulondb_native_promoter_panel/config.yaml`
   so its deliverables include the compact review path:

   - `representation_health_summary`
   - `native_core60_shift_summary`
   - `sigma_factor_structure_summary`
   - existing sigma UMAP plots as appendix orientation
   - `latent_geometry_browser`

   Add plot semantics YAML for each new plot under
   `src/dnadesign/latentdna/workspaces/regulondb_native_promoter_panel/plot_semantics/`.
   Add or update study deliverable markdown under
   `src/dnadesign/studies/regulondb_native_promoter_panel/deliverables/`.

9. Update `src/dnadesign/latentdna/workspaces/stress_ethanol_cipro_growth/config.yaml`
   without broad churn:

   - make candidate sets reflect materialized current, expanded reference, and
     planned output-layer/log-likelihood diagnostics explicitly;
   - add selector-backed ordinal groups for Sigma-35, Anderson strength, and W
     collection strength;
   - add or update reference-collapse plot specs and semantics;
   - keep `dataset_overview`, `representation_health_summary`,
     `design_structure_summary`, `sigma35_ordinal_audit`,
     `context_robustness_summary`, and `candidate_decision_frontier` as the
     primary review path unless a replacement carries the same decision role.

10. Update notebook controls and scaffold only after config-driven data is
    available. Target paths:

    - `src/dnadesign/latentdna/src/services/notebook_controls_service.py`
    - `src/dnadesign/latentdna/src/notebooks/scaffold_panels.py`
    - `src/dnadesign/latentdna/src/notebooks/browser_runtime.py`
    - `src/dnadesign/latentdna/src/notebooks/browser_runtime_plot_review.py`

    Use `Geometry browser` as the only second-tab and control-plane surface
    name. Do not accept legacy surface aliases. Preserve `mo.accordion`
    progressive disclosure and avoid duplicate global definitions in generated
    notebooks.

11. Add plot-render tests for any new plot kind or generalized render path.
    Target existing suites first:

    - `src/dnadesign/latentdna/tests/test_browser_runtime_plot_review.py`
    - `src/dnadesign/latentdna/tests/integrations/test_plot_gallery_rendering.py`
    - `src/dnadesign/latentdna/tests/integrations/test_notebook_generation_workflow.py`

12. Run targeted tests after each slice, then broader checks:

    ```sh
    uv run pytest -q src/dnadesign/latentdna/tests/test_scalar_build.py
    uv run pytest -q src/dnadesign/latentdna/tests/test_browser_runtime_plot_review.py
    uv run pytest -q src/dnadesign/latentdna/tests/test_notebook_controls_service.py
    uv run pytest -q src/dnadesign/latentdna/tests/contracts
    uv run pytest -q src/dnadesign/latentdna/tests/integrations/test_notebook_generation_workflow.py
    uv run pytest -q src/dnadesign/latentdna/tests
    ```

13. Regenerate and smoke notebooks only after deliverables are fresh:

    ```sh
    MPLCONFIGDIR=/tmp/dnadesign_mpl uv run latentdna notebook generate latent_geometry_browser --workspace stress_ethanol_cipro_growth --json
    MPLCONFIGDIR=/tmp/dnadesign_mpl uv run latentdna notebook smoke --workspace stress_ethanol_cipro_growth --json
    uv run marimo check src/dnadesign/latentdna/workspaces/stress_ethanol_cipro_growth/outputs/notebooks/latent_geometry_browser/notebook.py

    MPLCONFIGDIR=/tmp/dnadesign_mpl uv run latentdna notebook generate latent_geometry_browser --workspace regulondb_native_promoter_panel --json
    MPLCONFIGDIR=/tmp/dnadesign_mpl uv run latentdna notebook smoke --workspace regulondb_native_promoter_panel --json
    uv run marimo check src/dnadesign/latentdna/workspaces/regulondb_native_promoter_panel/outputs/notebooks/latent_geometry_browser/notebook.py
    ```

14. Update docs after behavior is stable:

    - `src/dnadesign/latentdna/docs/reference/workspace-schema.md`
    - `src/dnadesign/latentdna/docs/workflows/promoter-study-representation-comparison.md`
    - `src/dnadesign/latentdna/docs/reference/performance-budgets.md` if live
      pressure timings change materially
    - study deliverable markdown for any plot whose interpretation changed

15. Run repo-level validation from the root:

    ```sh
    uv run ruff check .
    uv run ruff format --check .
    uv run pytest -q
    uv run python -m dnadesign.devtools.docs.checks
    ```

### Validation and Acceptance

Functional acceptance:

- Both workspaces pass deep validation.
- Current, planned, missing, stale, and retired candidate surfaces are visible in
  machine-readable inventory. Missing sidecars are not silently ignored.
- RegulonDB review no longer depends only on UMAP. It includes health,
  native/core60 paired shift, sigma-factor structure, and UMAP orientation.
- Stress review includes all materialized vector candidate families in health
  and scree diagnostics, with output-layer and reverse-complement gaps surfaced
  explicitly when incomplete.
- Log-likelihood diagnostics can be selected as scalar hues or scalar summary
  plots where available, but are not projected as vector geometry.
- Reference collapse surfaces report collection-specific metrics for stress
  reference sets and degrade explicitly when a collection is absent from a
  candidate view.
- Ordinal audit supports one selected axis at a time and prevents pooled
  comparisons across incompatible Sigma-35, Anderson, and W collection scales.
- Notebook generation emits `notebook.py`, `controls.json`, `manifest.json`, and
  `health.json` under `outputs/notebooks/latent_geometry_browser/`.
- Marimo checks pass for generated notebooks, with no duplicate globals or
  dependency cycles.
- No package runtime branch is keyed to a specific study id or promoter
  collection name.

Performance acceptance:

- Before optimizing any slow path, capture a baseline workload, row count,
  memory posture, and runtime.
- Use existing memory policy thresholds for heavy view, reducer, projection,
  neighbors, cluster, and export operations.
- For large stress-study plots, prefer sampled or precomputed reducer/projection
  summaries unless a full-population run is explicitly required.
- Any accepted performance optimization must include before/after evidence
  across at least three comparable runs, plus correctness regression checks.

Documentation acceptance:

- Every rendered plot has a plot semantics sidecar satisfying
  `PlotSemantics`, including question, role, encoding, scope, guardrails,
  caption, alt text, preprocessing, math, rationale, limitations, and failure
  modes.
- Study deliverable markdown explains interpretation without becoming package
  policy.
- LatentDNA docs describe generic representation/scalar/view contracts rather
  than promoter-only semantics.

Repo acceptance:

- `git diff` contains no generated large artifacts unless explicitly approved.
- `uv run ruff check .` passes.
- `uv run ruff format --check .` passes.
- `uv run pytest -q` passes or any unrelated pre-existing failure is documented
  with exact command output.
- `uv run python -m dnadesign.devtools.docs.checks` passes.

### Open Questions

1. Should output-layer mean vectors become first-class candidate X surfaces as
   soon as their sidecars complete, or remain diagnostic until a separate review
   promotes them?
2. For reference collapse, is the ranking objective to maximize separation
   among synthetic standards, preserve native MG1655 landmarks, or avoid
   collapse of both? The implementation can report all three, but the decision
   frontier should not combine them without an explicit weighting rule.
3. Should RegulonDB sigma-factor structure be summarized as centroid separation,
   kNN enrichment, cluster agreement, or all three? Start with the smallest
   high-dimensional metric that survives fixture tests and add others only if
   they answer a distinct question.

### Links

- Proposal: this document
- Related design spec:
  [sequence-view ontology and Infer completion hardening](../../dev/plans/2026-04-28-sequence-view-ontology-and-infer-completion-hardening-spec.md)
- LatentDNA workflow:
  [promoter-study representation comparison](../../../src/dnadesign/latentdna/docs/workflows/promoter-study-representation-comparison.md)
- Stress study record:
  [stress ethanol cipro growth status](../../studies/stress_ethanol_cipro_growth/status.md)
- RegulonDB study record:
  [RegulonDB native promoter panel status](../../studies/regulondb_native_promoter_panel/status.md)
- PR: pending
- ADR: pending

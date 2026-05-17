## DenseGen + Promoter Status Quality Audit

Date: 2026-04-17
Study: `stress_ethanol_cipro_growth`
Audience: dev-spec author

### Purpose

Audit DenseGen and the promoter-study status surface against repo-wide
`dnadesign` architecture, design, reliability, and pragmatic API-contract
standards.

This note is not a feature plan. It is a gap inventory for a follow-on
alignment spec.

### Scope

In scope:

- DenseGen analysis/plot/notebook surface contracts
- promoter-study status and study-record organization
- cross-tool boundary quality where promoter status surfaces reach into DenseGen
- drift against `ARCHITECTURE.md`, `DESIGN.md`, `RELIABILITY.md`,
  `QUALITY_SCORE.md`, and pragmatic no-silent-fallback principles

Out of scope:

- downstream LatentDNA/Cluster/OPAL implementation internals
- solver/math correctness inside DenseGen generation core

### Repo-Wide Standards Used

- `DESIGN.md` requires explicit interfaces, fail-fast behavior, and forbids
  cross-tool internal `dnadesign.<tool>.src.*` imports.
- `ARCHITECTURE.md` requires file/event/public-API coupling across tool
  boundaries and explicit ownership planes.
- `RELIABILITY.md` requires cheap record-plane snapshots, distinct preflight
  surfaces, and no hidden fallback paths.
- `QUALITY_SCORE.md` says stable quality requires evidence-backed contracts and
  CI-enforced drift control.

### Findings

#### 1. Cross-tool boundary violation in promoter-study status

Severity: high

At the time of this audit,
`src/dnadesign/studies/status_adapters/promoter_status/analysis_surfaces.py`
imported DenseGen plot registry internals directly, even though the repo-wide
rule forbids cross-tool internal imports. The current implementation consumes
DenseGen through the public `dnadesign.densegen.inspect_analysis_surface`
surface instead.

Why this matters:

- It violates the repo boundary contract directly.
- It couples the study-status surface to DenseGen internals instead of a public
  contract.
- It makes a record-plane route dependent on a non-contractual package path.

Evidence:

- `DESIGN.md:54-60`
- `ARCHITECTURE.md:76-80`
- `src/dnadesign/studies/status_adapters/promoter_status/analysis_surfaces.py:294-299`
- `src/dnadesign/densegen/__init__.py:14-29`
- `src/dnadesign/densegen/tests/config/test_public_api_module_layout.py:20-34`

#### 2. Silent fallback and duplicated DenseGen knowledge inside promoter status

Severity: high

The promoter-study snapshot swallows DenseGen import failures and substitutes
empty plot specs. It also hardcodes DenseGen default and optional plot ids in
study status adapter code.

Why this matters:

- This is a direct violation of the repo's no-silent-fallback rule.
- The record-plane route can degrade without telling the operator.
- Tool-local knowledge is duplicated in the study status adapter instead of
  consumed through one public contract.

Evidence:

- `RELIABILITY.md:18-31`
- `DESIGN.md:20-23`
- `src/dnadesign/studies/status_adapters/promoter_status/analysis_surfaces.py:83-85`
- `src/dnadesign/studies/status_adapters/promoter_status/analysis_surfaces.py:294-312`

#### 3. DenseGen default plot surface has multiple sources of truth

Severity: high

The study pipeline, DenseGen workspace config, manifest, and notebook gallery do
not share one authoritative plot-surface contract.

Current drift:

- `pipeline.yaml` declares `dataset_metadata_heatmap` as a default plot.
- the DenseGen workspace config omits it from `plots.default`
- the notebook gallery hides it
- the plot manifest can retain it from prior runs

Why this matters:

- The status surface can report a default plot that the workspace does not
  currently generate by default.
- The notebook can hide generated artifacts without an explicit generated vs
  visible taxonomy.
- The manifest behaves as a cumulative ledger, but surrounding surfaces read it
  like a current inventory.

Evidence:

- `docs/studies/stress_ethanol_cipro_growth/pipeline.yaml:6-18`
- `src/dnadesign/densegen/workspaces/study_stress_ethanol_cipro/config.yaml:308-340`
- `src/dnadesign/densegen/src/viz/plot_inventory.py:24`
- `src/dnadesign/densegen/src/cli/notebook_cells_template_gallery.py:25-75`
- `src/dnadesign/densegen/src/viz/plotting.py:179-208`

#### 4. The Stage-A companion/ridgeline plot is not contract-safe

Severity: high

The registry says the ridgeline companion shows accepted TFBS length counts, but
the plot family only requires Stage-A pool artifacts. The implementation can
fall back to retained Stage-A lengths when accepted output annotations are
missing, while still presenting the right panel as if it were a Stage-B usage
view. The left-panel "Retained cutoff" is also semantically wrong for MMR.

Why this matters:

- The figure can silently mix different data populations.
- The legend implies a top-N score cutoff even though retention is MMR-based.
- This violates explicit contract and no-hidden-degraded-mode expectations.

Evidence:

- `src/dnadesign/densegen/src/viz/plot_registry.py:185-224`
- `src/dnadesign/densegen/src/viz/plot_stage_a.py:269-287`
- `src/dnadesign/densegen/src/viz/plot_stage_a_sampling_length_ridgeline.py:321-331`
- `src/dnadesign/densegen/src/viz/plot_stage_a_sampling_length_ridgeline.py:381-390`
- `src/dnadesign/densegen/src/viz/plot_stage_a_sampling_length_ridgeline.py:455-457`

#### 5. Promoter-study status overstates DenseGen surface health

Severity: medium

`analysis_surfaces.densegen.state` becomes `ok` if any rendered plots exist or
the notebook file exists. It does not compare generated artifacts to
`default_plot_ids`, does not reason about freshness, and does not check for
hidden/generated mismatches.

Why this matters:

- A partially generated or drifted analysis surface can still report `ok`.
- The state is too coarse for a contract surface that claims to guide operator
  routing.

Evidence:

- `src/dnadesign/studies/status_adapters/promoter_status/analysis_surfaces.py:87-92`
- `src/dnadesign/studies/status_adapters/promoter_status/analysis_surfaces.py:95-120`

#### 6. Study-owned route/status docs drift from repo doc-model rules

Severity: medium

The repo design rules say cross-tool route/runbook/workflow docs must declare
`Type`, `Plane`, `Owner-boundary`, `Entry artifact`, and `Exit artifact`.
`routes.md` does not declare those fields. `status.md` also leaks tool-local
guidance and even routes the reader to a LatentDNA snapshot command when asking
for the current record, which conflicts with the declared status ladder.

Why this matters:

- The study route surface is not metadata-complete by repo standards.
- The status note is no longer a short factual record; it leaks downstream tool
  heuristics and a competing route.

Evidence:

- `DESIGN.md:61-80`
- `ARCHITECTURE.md:90-99`
- `docs/studies/stress_ethanol_cipro_growth/routes.md:1-9`
- `docs/studies/stress_ethanol_cipro_growth/status.md:45-51`
- `docs/studies/README.md:16-35`
- `src/dnadesign/usr/docs/operations/promoter-study-status-contract.md:25-31`

#### 7. DenseGen exposes a public contract for USR output, but not for its
analysis surface

Severity: medium

DenseGen already has explicit public contracts for USR destination and producer
shape. There is no comparable public contract for plot registry, plot inventory,
notebook visibility, or analysis-surface status. Study-owned surfaces therefore
scrape internals, duplicate defaults, or infer meaning from manifests.

Why this matters:

- The runtime output contract is stronger than the operator analysis contract.
- Cross-tool consumers have no sanctioned stable API for "what DenseGen shows".

Evidence:

- `src/dnadesign/densegen/contracts.py:1-212`
- `src/dnadesign/studies/status_adapters/promoter_status/analysis_surfaces.py:56-120`

### Pragmatic-Principles Readout

Skill composition: paired `deep-introspection` + `promoter-study-status` +
`pragmatic-programming-principles`

Principles currently not met cleanly:

- Explicit contracts: generated plots, visible plots, default plots, and route
  inventory are not defined in one place.
- DRY knowledge: DenseGen plot taxonomy is duplicated across workspace config,
  pipeline record, notebook inventory, and study status adapter code.
- Orthogonality: record-plane status must consume a small public surface
  contract instead of reaching into DenseGen internals.
- No silent fallback: study status adapter analysis must report
  contract-visible degradation when declared analysis surfaces cannot be read.
- Fail fast: current behavior often returns plausible-but-degraded output
  instead of a contract-visible degraded state.

### Dev-Spec Targets

1. Define a public DenseGen analysis-surface contract.
   - Export stable plot/default/visibility/freshness semantics from DenseGen
     without requiring `dnadesign.densegen.src.*` imports.

2. Separate surface taxonomies explicitly.
   - `generated_surface`
   - `default_surface`
   - `operator_visible_surface`
   - `optional_surface`
   - `historical_or_stale_surface`

3. Make degraded modes explicit.
   - If plot registry import fails, surface an error or degraded-state payload.
   - If ridgeline lacks accepted-output annotations, fail closed or label the
     figure as retained-only.

4. Eliminate duplicate plot-default declarations.
   - One owner for default plot ids.
   - Other surfaces should reference that contract, not restate it.

5. Tighten study-doc routing.
   - `status.md` stays factual and short.
   - `routes.md` gets explicit metadata headers and only one-hop routing.
   - record refresh commands point to the sanctioned record-plane snapshot.

### Verification Evidence

Verified during this audit:

- `uv run ops progress show usr.data-plane.promoter-study-status --json`
- `uv run pytest -q src/dnadesign/studies/tests/test_promoter_snapshot.py src/dnadesign/studies/tests/test_promoter_snapshot_latentdna.py src/dnadesign/densegen/tests/config/test_public_api_module_layout.py src/dnadesign/usr/tests/test_usr_docs_contract.py`

Result:

- `62` tests passed

### Next Smallest Increment

The smallest reversible improvement is not a broad refactor. It is:

1. define one public DenseGen analysis-surface API,
2. update promoter-study status to consume only that API,
3. fail visibly on degraded analysis-surface states,
4. then align `pipeline.yaml`, notebook gallery rules, and route docs to that
   same contract.

### Mutation Boundary For The Dev Spec

This section is normative guidance for the follow-on dev spec.

#### Do not mutate the current shared DenseGen source dataset

Treat the current shared USR dataset as read-only baseline evidence:

- dataset id: `densegen/study_stress_ethanol_cipro`
- current accepted rows: `157,160`
- role: canonical shared DenseGen source for this live study

The dev spec should explicitly say:

- do not rewrite, compact, rematerialize, regenerate, or otherwise mutate the
  current `157,160`-row shared DenseGen dataset as part of the contract-alignment
  work
- do not change row identity, row count, overlay semantics, or metadata payloads
  on that dataset just to fix route/status/plot-surface issues
- do not make the spec depend on rerunning DenseGen generation against the live
  study as a prerequisite for the contract fix

#### Allowed change surface

The dev spec may authorize changes in these areas:

- promoter-study status skill and study status adapter wiring
- OPS/status routing that consumes study-owned analysis metadata
- `docs/studies/...` record surfaces such as `status.md`, `routes.md`, and
  `pipeline.yaml`
- DenseGen code that defines or exports a public analysis-surface contract
- DenseGen notebook/gallery/plot inventory behavior
- plot registry / plot inventory / manifest semantics
- plot-generation code, including ridgeline behavior and labeling
- generated plot artifacts and generated notebook artifacts under DenseGen
  workspace `outputs/`
- downstream derived artifacts that are explicitly regenerated from existing
  study outputs without mutating the canonical shared source dataset

#### Preferred fix strategy

The spec should prefer:

1. read-only inspection of the existing shared dataset,
2. contract and wiring fixes in code/docs/status surfaces,
3. regeneration of local workspace analysis artifacts only,
4. optional regeneration of downstream derived artifacts only when they are
   clearly outside the immutable shared source baseline.

#### Explicit non-goals

The spec should say these are out of scope unless separately approved:

- rebuilding the `157k` DenseGen study from scratch
- changing the study’s shared USR source dataset contents
- changing DenseGen Stage A or Stage B generation behavior in ways that would
  require a new canonical source dataset for this study
- retroactively editing shared USR records to paper over analysis-surface
  contract problems

#### Suggested spec language

Use wording close to this:

> The alignment work is contract-surface only. The canonical shared DenseGen
> source dataset `densegen/study_stress_ethanol_cipro` at `157,160` rows is
> treated as immutable baseline evidence for this change. Implementation may
> read that dataset, validate it, and derive local analysis artifacts from it,
> but must not mutate, rewrite, or regenerate that shared source dataset as part
> of this scope. Changes are allowed in DenseGen’s public analysis-surface API,
> study-status wiring, OPS/studies docs, plot-generation logic, notebook/gallery
> logic, and regenerated workspace-local analysis artifacts.

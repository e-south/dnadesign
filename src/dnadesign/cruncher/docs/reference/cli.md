## Cruncher CLI

**Owner:** dnadesign-maintainers
**Last verified:** 2026-08-09


**Last updated by:** cruncher-maintainers on 2026-07-13

### Contents
- [Cruncher CLI](#cruncher-cli)
- [Workspace discovery and config resolution](#workspace-discovery-and-config-resolution)
- [Quick command map](#quick-command-map)
- [Core lifecycle commands](#core-lifecycle-commands)
- [Cassette workflows](#cassette-workflows)
- [Snapback workflows](#snapback-workflows)
- [Scar-nick workflows](#scar-nick-workflows)
- [YIU workflows](#yiu-workflows)
- [Study workflows](#study-workflows)
- [Portfolio workflows](#portfolio-workflows)
- [Discovery and inspection](#discovery-and-inspection)
- [Global options](#global-options)

This reference summarizes the Cruncher CLI surface, grouped by lifecycle stage and workflow family.

> **Workflow families:** Cruncher currently registers seven peer command families.
>
> - `fetch|lock|parse|sample|analyze|export` cover the `sample` family. This lane uses Gibbs annealing MCMC plus MMR elite selection and is not posterior inference.
> - `cassette init-workspace|validate|design|solve|show` cover cassette workspaces. This lane uses explicit cassette planning plus bounded solve search and keeps separate artifact contracts.
> - `snapback init-workspace|validate|design|solve|target-search|released-design|released-target-search|released-solve|show|released-show` cover single-nick foldback workspaces. This lane uses explicit geometry contracts, bounded deterministic search, a strict QA triptych publication surface for preserved-site explicit bundles, and a separate released-product precursor contract.
> - `scar-nick validate|design|show` cover retained-scar terminal-nick workspaces. This lane uses explicit Type IIS scar and nickase geometry contracts, deterministic candidate ranking, and BaseRender-ready QA handoffs.
> - `yiu init-workspace|validate|render|show` cover payload-centric YIU workflows. This lane uses the strict `split_yiu_payload_rendering_v4` contract, deterministic exhaustive optimization over a 4 nt junction, three published payload views, and one bundle-local `visual_inventory.json`.
> - `study run|summarize|show` cover the `study` family. This lane orchestrates aggregate sweeps over workspace outputs rather than defining a new design topology.
> - `portfolio run|show` cover the `portfolio` family. This lane aggregates study-ready workspaces into cross-study tables and plots.
>
> Choose the command family by the workspace contract you need. `cassette`, `snapback`, `scar-nick`, `yiu`, `study`, and `portfolio` do not fall back to `sample`, and `sample` runs do not reuse their artifacts.
>
> For Snapback specifically, `cli/commands/snapback.py` is the registry surface only. Subcommand handlers are split across `snapback_workspace.py`, `snapback_explicit.py`, `snapback_released.py`, and `snapback_show.py`, while typed request construction lives in `app/snapback_cli_requests.py`.

#### Workspace discovery and config resolution

Cruncher resolves config from `--config/-c` or `--workspace/-w`, then `<workspace>/configs/config.yaml` from the current directory/parents (or `config.yaml` when CWD is already `<workspace>/configs`), then known workspace roots. If multiple workspaces are found, **cruncher** prompts for a selection (interactive shells only).

See available workspaces with:

```
cruncher workspaces list
```

List workspace-scoped Study specs and Study runs with:

```
cruncher study list
```

---

#### Quick command map

* **Cache data** → `fetch motifs` / `fetch sites`
* **Inspect cache** → `sources ...` / `catalog ...`
* **Pin TFs** → `lock`
* **Validate motifs** → `parse`
* **Render logos** → `catalog logos`
* **Optimize fixed-length sequences** → `sample`
* **Analyze optimization runs** → `analyze`, `notebook`
* **Design and search cassettes** → `cassette init-workspace|validate|design|solve|show`
* **Validate and search single-nick foldbacks** → `snapback init-workspace|validate|design|solve|target-search|released-design|released-target-search|released-solve|show|released-show`
* **Evaluate retained-scar terminal nicks** → `scar-nick validate|design|show`
* **Render split YIU payloads** → `yiu init-workspace|validate|render|show`
* **Study sweeps** → `study list|run|summarize|show|clean`
* **Cross-workspace handoff aggregation** → `portfolio run|show`
* **Export sequences** → `export sequences`
* **Run management** → `runs list/show/latest/best/watch/clean`
* **Workspace health + machine runbooks** → `status`, `workspaces run|reset`

---

#### Core lifecycle commands

#### `cruncher fetch motifs`

Caches motif matrices into `<catalog.root>/normalized/motifs/...`.

Inputs:

* optional config path (`--config/-c`), otherwise resolved from workspace/CWD
* at least one of `--tf` or `--motif-id`

Network:

* yes by default; use `--offline` to restrict to cached motifs only

When to use:

* you want `cruncher.catalog.pwm_source: matrix`
* you want to reuse alignment/matrix payloads across runs

Examples:

* `cruncher fetch motifs --tf lexA --tf cpxR <config>`
* `cruncher fetch motifs --motif-id RDBECOLITFC00214 <config>`
* `cruncher fetch motifs --source omalley_ecoli_meme --tf lexA <config>`
* `cruncher fetch motifs --dry-run --tf lexA <config>`

Common options:

* `--tf`, `--motif-id`, `--source`
* `--dry-run`, `--all`, `--offline`, `--update`
* `--summary/--no-summary`, `--paths`

Outputs:

* writes cached motif JSON files and updates `catalog.json`
* prints a summary table by default (or raw paths with `--paths`)

Note:

* `--source` defaults to the first available entry in `catalog.source_preference` (skipping entries that are
  not registered ingest sources); if the list is empty or none are available you must pass `--source` explicitly.

---

#### `cruncher fetch sites`

Caches binding-site instances into `<catalog.root>/normalized/sites/...`.

Inputs:

* optional config path (`--config/-c`), otherwise resolved from workspace/CWD
* at least one of `--tf`, `--motif-id`, or `--hydrate`

Network:

* yes by default; use `--offline` to restrict to cached sites only

When to use:

* you want `cruncher.catalog.pwm_source: sites`
* you want curated or HT site sets cached locally
* you need hydration for coordinate-only peaks

Examples:

* `cruncher fetch sites --tf lexA --tf cpxR <config>`
* `cruncher fetch sites --dry-run --tf lexA <config>`
* `cruncher fetch sites --dataset-id <id> --tf lexA <config>`
* `cruncher fetch sites --genome-fasta genome.fna <config>`

Common options:

* `--tf`, `--motif-id`, `--dataset-id`, `--limit`, `--source`
* `--hydrate` (hydrates missing sequences)
* `--offline`, `--update`
* `--genome-fasta`
* `--summary/--no-summary`, `--paths`

Outputs:

* writes cached site JSONL files and updates `catalog.json`
* prints a summary table by default (or raw paths with `--paths`)

Note:

* `--hydrate` with no `--tf/--motif-id` hydrates all cached site sets by default.
* `--source` defaults to the first available entry in `catalog.source_preference` (skipping entries that are
  not registered ingest sources); if the list is empty or none are available you must pass `--source` explicitly.
* with both curated and HT enabled, `--limit` requires explicit mode (`--dataset-id` or one source class disabled).
* HT mode is strict: if HT discovery/fetch fails or returns zero rows for the selected mode, the command errors.
* if `tfbinding` returns zero rows for a known dataset, switch `ingest.regulondb.ht_binding_mode` to `peaks`.

---

#### `cruncher lock`

Resolves TF names to exact cached artifacts (IDs + hashes) from `workspace.regulator_sets`.
Writes `<workspace>/.cruncher/locks/<config>.lock.json`.

Inputs:

* optional config path (`--config/-c`), otherwise resolved from workspace/CWD
* cached motifs/sites for the configured regulators

Network:

* no (cache-only)

When to use:

* before `parse` and `sample`
* whenever you change anything affecting TF resolution (PWM source, site kinds, dataset selection, etc.)

Example:

* `cruncher lock <config>`

---

#### `cruncher parse`

Validates cached PWMs (matrix- or site-derived) and writes parse-cache artifacts in workspace state.

Inputs:

* CONFIG (explicit or resolved)
* lockfile (from `cruncher lock`)
* optional `--force-overwrite` to replace an existing run directory

Network:

* no (cache-only)

Example:

* `cruncher parse <config>`

Precondition:

* lockfile exists (`cruncher lock <config>`)

Notes:

* Logos are rendered via `cruncher catalog logos`; default logo settings are `--bits-mode information` and `--dpi 150`.
* `cruncher parse` always uses the lockfile to pin exact motif IDs/hashes.
  If you add new motifs (e.g., via `discover motifs`) or change `catalog` preferences,
  re-run `cruncher lock <config>` to refresh what parse will use.
* Parse requires overwrite intent when output already exists; use `--force-overwrite` to replace.
* Parse artifacts live under `<workspace>/.cruncher/parse/inputs/` and are intentionally
  separate from user-facing sample outputs in `workspace.out_dir`.
* Use `cruncher catalog logos` to render PWM logos with provenance subtitles.

---

#### `cruncher sample`

Runs MCMC optimization to design sequences scoring well across TFs.

Inputs:

* CONFIG (explicit or resolved)
* lockfile (from `cruncher lock`)

Network:

* no (cache-only)

Example:

* `cruncher sample <config>`
* `cruncher sample --verbose <config>`
* `cruncher sample --no-progress <config>`
* `cruncher sample --debug <config>`

Precondition:

* lockfile exists (`cruncher lock <config>`)

Notes:

* `sample.output.save_sequences: true` is required for later analysis.
* `sample.output.save_trace: true` enables trace-based diagnostics.
* `--no-progress` disables progress bars and periodic progress logging for quieter non-interactive runs.
* `sample.output.save_trace: false` skips ArviZ trace construction and reduces sample runtime/memory overhead.
* Sampling uses `sample.optimizer.*` (`kind: gibbs_anneal`) with chain count and cooling schedule under one explicit surface.
* Replica exchange is disabled in `gibbs_anneal`; chains are tracked directly in trajectory outputs.
* `--verbose` enables periodic progress logging; `--debug` enables very verbose debug logs.

---

#### `cruncher analyze`

Generates diagnostics and plots for one or more sample runs.

Inputs:

* CONFIG (explicit or resolved)
* runs via `analysis.run_selector`/`analysis.runs` or `--run` (defaults to latest sample run if empty)
* run artifacts: `optimize/tables/sequences.parquet` (required), `optimize/tables/elites.parquet` (required),
  `optimize/tables/elites_hits.parquet` (required), `optimize/tables/random_baseline*.parquet` (required when
  `sample.output.save_random_baseline=true`, default with `sample.output.random_baseline_n=10000`), and
  `optimize/trace.nc` for trace-based plots

Network:

* no (run artifacts only)

Examples:

* `cruncher analyze --latest <config>`
* `cruncher analyze --run <run_name|run_dir> <config>`
* `cruncher analyze --summary <config>`

Preconditions:

* provide runs via `analysis.runs`/`--run` or rely on the default latest run
* selected sample runs must be completed; analyze fails fast when latest run status is still `running`
* run selection preflight happens before plotting/cache initialization, so missing/incomplete runs fail quickly with no Matplotlib/ArviZ cache setup
* trace-dependent plots require `optimize/trace.nc`
* each sample run snapshots the lockfile under `provenance/lockfile.json`; analysis uses that snapshot to avoid mismatch if the workspace lockfile changes later

---

#### Cassette workflows

The cassette workflow is separate from `sample`. It expects an explicit spec file at
`<workspace>/configs/cassettes/<name>.cassette.yaml` plus a local nickase catalog, validates the dual-context invariant set,
and writes cassette-specific artifacts under `outputs/cassettes/`.

Current scope:

* deterministic validation/materialization of an authored cassette spec
* strict local nickase catalog loading
* explicit unsatisfied reports when no valid left/right nick pair exists
* separate solve/search entrypoint for patterned stem/loop exploration

Current non-scope:

* no downstream excision/removal semantics after nicking

Deep contracts live in:

* [`../demos/demo_cassette_workspace.md`](../demos/demo_cassette_workspace.md)
* [`reference/cassette_spec.md`](cassette_spec.md)
* [`reference/cassette_solve_spec.md`](cassette_solve_spec.md)
* [`reference/nickase_catalog.md`](nickase_catalog.md)
* [`reference/cassette_artifacts.md`](cassette_artifacts.md)
* [`../guides/cassette_workflow.md`](../guides/cassette_workflow.md)
* [`../guides/cassette_solve_workflow.md`](../guides/cassette_solve_workflow.md)

#### `cruncher cassette init-workspace`

Scaffold a cassette-specific workspace root with pressure-tested solve specs for different runtime budgets.

Inputs:

* required `WORKSPACE` or `--output <path>`
* optional `--root <dir>` when using `WORKSPACE`
* optional `--force-overwrite` to replace a scaffold previously generated by this command

Network:

* no

Examples:

* `uv run cruncher cassette init-workspace cassette_lab`
* `uv run cruncher cassette init-workspace cassette_lab --root ./workspaces`
* `uv run cruncher cassette init-workspace --output ./cassette_lab --force-overwrite`

Outputs:

* writes `README.md`
* writes `runbook.md`
* writes `cassette_workspace_manifest.json`
* writes `configs/runbook.yaml`
* writes `configs/cassettes/demo_hairpin_fast.cassette.solve.yaml`
* writes `configs/cassettes/demo_hairpin_balanced.cassette.solve.yaml`
* writes `configs/cassettes/demo_hairpin_deep_mmr.cassette.solve.yaml`
* creates `inputs/nickases/`, `outputs/cassettes/`, and `outputs/cassette_solves/`

Notes:

* this is a cassette-specific scaffold, not a full general Cruncher sampling workspace
* the scaffold stays isolated and refuses to overwrite a non-empty unowned root
* the scaffold rejects symlinked output roots and symlinked ancestor directories so it writes only to the path you named
* generated solve specs are pressure-tested profiles for fast, balanced, and deeper MMR-biased search runs
* `cassette_workspace_manifest.json` records the profile search/selection settings so operators can diff fast vs balanced vs deep MMR without reopening each YAML
* `configs/runbook.yaml` makes the scaffold discoverable through `cruncher workspaces list` as `runbook-only`
* use the explicit `configs/cassettes/*.cassette.solve.yaml` paths for balanced and deep cassette runs; the runbook keeps the fast smoke path discoverable through `workspaces run`
* emitted solve jobs keep the `views/` -> `baserender_jobs/` -> `renders/` flow inside the scaffold root
* use `cruncher cassette catalog init-neb --output inputs/nickases/neb_nicking_v1.yaml` if you want a local editable copy of the built-in preset
* see [`../demos/demo_cassette_workspace.md`](../demos/demo_cassette_workspace.md) for the shortest end-to-end tutorial

#### `cruncher cassette validate`

Validate one cassette spec and print a deterministic planning report.

Inputs:

* `--spec <workspace>/configs/cassettes/<name>.cassette.yaml`
* local catalog path from `cassette.catalog.path`

Network:

* no

Examples:

* `uv run cruncher cassette validate --spec configs/cassettes/demo_hairpin.cassette.yaml`
* `uv run cruncher cassette validate --spec configs/cassettes/demo_hairpin.cassette.yaml --json`

Notes:

* `--spec` must point to a `.cassette.yaml` file path under a workspace `configs/` tree.
* Nick windows are matched against cassette-relative reported nick boundaries.
* The report distinguishes `bounded_segment` from downstream removal/excision semantics.
* `--json` prints machine-readable JSON with no Rich formatting.
* Unsupported tracer-bullet mode flags fail at load time rather than degrading silently.

#### `cruncher cassette design`

Validate a cassette spec and write deterministic cassette artifacts.

Inputs:

* `--spec <workspace>/configs/cassettes/<name>.cassette.yaml`
* optional `--force-overwrite` to replace an existing deterministic run directory

Network:

* no

Examples:

* `uv run cruncher cassette design --spec configs/cassettes/demo_hairpin.cassette.yaml`
* `uv run cruncher cassette design --spec configs/cassettes/demo_hairpin.cassette.yaml --force-overwrite`

Outputs:

* writes under `<workspace>/outputs/cassettes/<spec.name>/<design_id>/`
* writes `meta/cassette_manifest.json`, `meta/cassette_status.json`, `analysis/reports/report.{json,md}`
* writes `export/table__candidates.csv`
* writes `views/linear_duplex.v1.json`, `views/ssdna_hairpin.v1.json`, and `views/views_manifest.v1.json` when `output.emit_visual_contracts: true` and the explicit planner materializes a concrete candidate
* writes `baserender_jobs/linear_duplex.job.yaml` and `baserender_jobs/ssdna_hairpin.job.yaml` when `output.emit_baserender_jobs: true` and the corresponding view files were published
* `--json` prints the machine-readable report only; the run directory is included at `report.run_dir`

Notes:

* cassette runs are additive and do **not** register in workspace `run_index.json`
* unsatisfied specs still write a cassette run directory with explicit issue codes, but view/job publication is skipped when no concrete candidate is available
* there is no fallback to `sample`
* unsupported tracer-bullet mode flags fail before materialization

#### `cruncher cassette solve`

Search for ranked cassette hits from a separate `.cassette.solve.yaml` spec and optionally materialize the top hits as
explicit cassette bundles.

Inputs:

* `--spec <workspace>/configs/cassettes/<name>.cassette.solve.yaml`
* built-in preset catalog via `catalog.preset`, optional overlays via `catalog.additional_paths`
* optional `--force-overwrite` to replace an existing deterministic solve directory

Network:

* no

Examples:

* `uv run cruncher cassette solve --spec configs/cassettes/demo_hairpin.cassette.solve.yaml`
* `uv run cruncher cassette solve --spec configs/cassettes/demo_hairpin.cassette.solve.yaml --json`

Outputs:

* writes under `<workspace>/outputs/cassette_solves/<solve_id>/`
* writes `solve_report.{json,md}`, `table__hits.csv`, `solve_manifest.json`, `solve_status.json`
* writes `views/top_hits.linear_duplex.v1.jsonl` and `views/top_hits.ssdna_hairpin.v1.jsonl` when `output.emit_visual_contracts: true`
* writes `baserender_jobs/top_hits_duplex.job.yaml` and `baserender_jobs/top_hits_hairpin.job.yaml` when `output.emit_baserender_jobs: true`
* writes per-hit explicit bundles, view bundles, and job files under `hits/hit_<rank>_<solution_id>/...`
* `invalid_spec` and `invalid_catalog` preflight failures still write a top-level solve bundle when a workspace can be derived, but they do not write per-hit bundles
* `--json` prints the machine-readable solve report only; the run directory is included at `report.run_dir`

Notes:

* `nick_goal.target_strand` is required in solve mode; there is no implicit default
* solve mode is bounded by `search.max_enumerated_candidates` and `search.max_search_nodes`
* final hit selection is controlled by `search.selection.policy`:
  `score_only`, `greedy_hamming` (compatibility default), or opt-in `mmr`
* `search.selection.pool_size` bounds the accepted pool retained before final selection
* plain CLI output now surfaces selected-hit ids, default-vs-explicit policy status, pool/search boundedness, and policy-limited underfill
* solve warnings are surfaced in plain CLI output and in `solve_report.json`
* `solve_status.json` also preserves warning details, accepted-pool telemetry, `search_truncated`, policy-underfill flags, and the top-hit JSONL/job paths for lightweight machine consumers
* warning codes include `ACCEPTED_POOL_TRUNCATED`, `SELECTION_RESULTS_POOL_BOUNDED`, `SELECTION_RESULTS_SEARCH_BOUNDED`, and `SELECTION_POLICY_LIMITED_HITS` when the returned hit set is constrained below the available accepted pool
* each accepted hit round-trips through the explicit cassette planner
* shared view contracts are file-based handoffs for baserender; Cruncher does not import baserender internals
* there is no fallback from `solve` into `sample`

#### `cruncher cassette catalog init-neb`

Write the built-in `neb_nicking_v1` cassette nickase preset to a local YAML path.

Inputs:

* `--output <path>`
* optional `--force-overwrite` to replace an existing output file

Network:

* no

Examples:

* `uv run cruncher cassette catalog init-neb --output configs/catalogs/neb_nicking_v1.yaml`

Outputs:

* writes the packaged preset YAML to the requested path

Notes:

* the command exports the built-in packaged preset exactly as shipped
* this is a convenience helper for local inspection or overlay authoring; `cassette solve` can still reference `catalog.preset: neb_nicking_v1` directly

#### `cruncher cassette show`

Read one cassette run directory and print its key artifact paths.

Inputs:

* `--run <workspace>/outputs/cassettes/<spec.name>/<design_id>`

Network:

* no

Example:

* `uv run cruncher cassette show --run outputs/cassettes/demo_hairpin/<design_id>`

Outputs:

* prints the explicit cassette run directory
* prints `meta/cassette_manifest.json`
* prints `meta/cassette_status.json`
* prints `analysis/reports/report.json`
* prints `analysis/reports/report.md`
* prints `views/views_manifest.v1.json` when present
* prints `baserender_jobs/linear_duplex.job.yaml` and `baserender_jobs/ssdna_hairpin.job.yaml` when present

Notes:

* `show` remains an explicit-lane inspection command; solve runs are inspected via the solve report bundle
* `show` does not read from the workspace `run_index.json`

---

#### Snapback workflows

The snapback workflow is separate from `sample`, `cassette`, and `yiu`. It expects one explicit preserved-site spec at
`<workspace>/configs/snapback/<name>.snapback.yaml`, one solve spec at
`<workspace>/configs/snapback/<name>.snapback.solve.yaml`, or one released-product explicit spec at
`<workspace>/configs/snapback/<name>.released.snapback.yaml`, plus local or preset-backed nickase catalogs and, for the released-product lane, release-enzyme catalogs.

Current scope:

* deterministic validation of one authored single-nick foldback design
* bounded co-design search over nick boundary, retained homology length, cap extension, motif-compatible site edits, and foldback-arm choices
* target-first catalog search for shortest preserved-site hits at an exact requested geometry
* target-first paired nickase plus release-enzyme search in exposed-bottom geometry space
* stable explicit, solve, and released-product bundles under `outputs/design/`, `outputs/solve/`, `outputs/released_solve/`, and `outputs/released_design/`
* strict QA-triptych publication plus fail-fast `show` integrity checks

Current non-scope:

* no thermodynamic folding prediction
* no ligation-yield or retron/processivity scoring
* no fallback to `sample`, `cassette`, `scar-nick`, or `yiu`

Deep contracts live in:

* [`../guides/snapback_workflow.md`](../guides/snapback_workflow.md)
* [`../guides/snapback_released_workflow.md`](../guides/snapback_released_workflow.md)
* [`snapback_artifacts.md`](snapback_artifacts.md)
* [`released_snapback_artifacts.md`](released_snapback_artifacts.md)
* [`release_enzyme_catalogs.md`](release_enzyme_catalogs.md)
* [`architecture.md`](architecture.md)
* [`../../workspaces/demo_released_snapback/README.md`](../../workspaces/demo_released_snapback/README.md)

#### `cruncher snapback init-workspace`

Scaffold a snapback-specific workspace with one explicit Bpu10I example, one broader catalog-scan solve example, one local nickase catalog, and one machine runbook.

Examples:

* `uv run cruncher snapback init-workspace snapback_lab`
* `uv run cruncher snapback init-workspace --output ./snapback_lab --force-overwrite`

Outputs:

* writes `README.md`
* writes `runbook.md`
* writes `configs/runbook.yaml`
* writes `configs/snapback/demo_teto_bpu10i_cap.snapback.yaml`
* writes `configs/snapback/demo_teto_catalog_scan.snapback.solve.yaml`
* writes `inputs/nickases/local.nickases.yaml`

Notes:

* this is a snapback-specific scaffold, not a fixed-length optimization workspace
* the scaffold is designed around stable `outputs/design/` and `outputs/solve/` roots
* the explicit lane stays pinned to the local `Nt.Bpu10I` overlay, while the solve lane searches built-in `neb_nicking_v1` plus `thermo_nicking_v1`
* use the shipped runbook for the shortest validate -> design -> show -> solve -> show path

#### `cruncher snapback validate`

Validate one explicit snapback spec and print a deterministic report.

Examples:

* `uv run cruncher snapback validate --spec configs/snapback/demo_teto_bpu10i_cap.snapback.yaml`
* `uv run cruncher snapback validate --spec configs/snapback/demo_teto_bpu10i_cap.snapback.yaml --json`

Notes:

* `--spec` must point to a `.snapback.yaml` file under a workspace `configs/snapback/` tree
* the explicit lane requires one intended nick inside the requested boundary and duplex windows
* the effective cap loop is fixed at `3 nt`
* `--json` prints machine-readable JSON with no Rich formatting

#### `cruncher snapback design`

Validate one explicit snapback spec and write the explicit bundle.

Examples:

* `uv run cruncher snapback design --spec configs/snapback/demo_teto_bpu10i_cap.snapback.yaml`
* `uv run cruncher snapback design --spec configs/snapback/demo_teto_bpu10i_cap.snapback.yaml --force-overwrite`

Outputs:

* writes under `<workspace>/outputs/design/`
* writes `meta/snapback_manifest.json`, `meta/snapback_status.json`, and `analysis/reports/report.{json,md}`
* writes `export/table__candidates.csv`
* writes `analysis/views/` QA views and `snapback_visual_v1` contracts when the planner materializes a truthful candidate
* writes `baserender_jobs/snapback_triptych.job.yaml` when `output.emit_baserender_jobs: true`

Notes:

* unsatisfied explicit specs still write a bundle so the issue report is preserved
* `show` is the supported way to inspect that bundle later
* snapback bundles do not register in workspace `run_index.json`

#### `cruncher snapback solve`

Search for ranked single-nick foldback hits and optionally materialize the top hits as explicit bundles.

Examples:

* `uv run cruncher snapback solve --spec configs/snapback/demo_teto_catalog_scan.snapback.solve.yaml`
* `uv run cruncher snapback solve --spec configs/snapback/demo_teto_catalog_scan.snapback.solve.yaml --json`

Outputs:

* writes under `<workspace>/outputs/solve/`
* writes `analysis/reports/solve_report.{json,md}`
* writes `export/table__hits.csv` and `export/table__frontier.csv`
* writes materialized explicit top-hit bundles under `analysis/materialized_hits/hit_<rank>/`

Notes:

* solve is bounded by `search.max_search_nodes` and `search.max_enumerated_candidates`
* returned status can be `satisfied`, `no_hits`, or `search_truncated`
* the live ranking is deterministic and geometry-first; see the snapback workflow guide for the current order
* materialized hits round-trip through the explicit `single_nick_snapback_v2` contract

#### `cruncher snapback target-search`

Search the resolved nickase catalog for the shortest preserved-site hit at an exact snapback geometry without assuming an authored top strand.

Examples:

* `uv run cruncher snapback target-search --json`
* `uv run cruncher snapback target-search --nick-boundary 0 --paired-bp 3 --cap-nt 3`
* `uv run cruncher snapback target-search --preset neb_nicking_v1 --additional-preset thermo_nicking_v1 --json`

Outputs:

* prints one typed report to stdout
* reports exact hits first when any exist
* reports nearest preserved-site later-boundary hits when no exact boundary hit exists for an entry/orientation
* includes a feasibility table for every evaluated target-strand placement

Notes:

* this mode is target-first and separate from the fixed-input `solve` lane
* the effective cap loop is still fixed at `3 nt`
* recognition sites are preserved exactly in this mode; the search does not mutate the RE site
* when no catalog source is provided, the command defaults to `neb_nicking_v1` plus `thermo_nicking_v1`

#### `cruncher snapback released-design`

Validate one explicit two-stage precursor spec and write the released-product bundle.

Examples:

* `uv run cruncher snapback released-design --spec configs/snapback/example.released.snapback.yaml`
* `uv run cruncher snapback released-design --spec configs/snapback/example.released.snapback.yaml --force-overwrite`

Outputs:

* writes under `<workspace>/outputs/released_design/`
* writes `meta/released_snapback_manifest.json` and `meta/released_snapback_status.json`
* writes `analysis/report.json`, `analysis/released_product_projection.json`, `analysis/pre_nick_site.json`, and `analysis/release_site.json`
* writes `export/released_design_summary.csv` with route-policy, retained-partner fragment, and generic active-product columns

Notes:

* explicit released-design defaults to `final_geometry_source=exposed_bottom_strand` with `route_family=bottom_active_from_top_nick`, not the full precursor
* only `nick_then_release` and `retained_side=upstream` are supported in v1
* the release site and nickase site may be lost post-release when the explicit spec allows that loss
* by default, explicit released-product specs reject nickases carrying `FREQUENT_CUTTER`

#### `cruncher snapback released-target-search`

Search paired nickase plus release-enzyme combinations for released-product geometry without assuming an authored precursor.

Examples:

* `uv run cruncher snapback released-target-search --workspace-root src/dnadesign/cruncher/workspaces/demo_released_snapback --nick-preset neb_nicking_v1 --nick-additional-preset thermo_nicking_v1 --release-preset type_iis_release_v1 --release-variant-id BspQI --allow-top-active-routes --allow-precut-footprint-outside-active-product --json`
* `uv run cruncher snapback released-target-search --workspace-root src/dnadesign/cruncher/workspaces/demo_released_snapback --nick-preset neb_nicking_v1 --nick-additional-preset thermo_nicking_v1 --release-preset type_iis_release_v1 --release-variant-id BspQI --nick-boundary 0 --paired-bp 3 --cap-nt 3 --allow-top-active-routes --allow-precut-footprint-outside-active-product`

Outputs:

* prints one typed report to stdout
* reports exact hits and near hits separately
* includes blocker counts plus pre-truncation and post-truncation hit counts

Notes:

* this mode is target-first and separate from preserved-site `target-search`
* without route-policy flags, the command evaluates `final_geometry_source=exposed_bottom_strand` via `route_family=bottom_active_from_top_nick`
* broader retained-active searches use `route_family=top_active_from_bottom_nick` plus `final_geometry_source=retained_active_strand`
* `--allow-top-active-routes` plus `--allow-precut-footprint-outside-active-product` opt into retained-active auditing
* `--release-variant-id` restricts the release-enzyme cross-product; the checked-in demo uses `BspQI`
* the command requires at least one explicit nickase source and one explicit release-enzyme source
* CLI parsing delegates typed request construction to `app/snapback_cli_requests.py`; the command module does not build released search models inline
* demo-only catalog entries are excluded unless `--allow-demo-hits` is passed
* nickases carrying `FREQUENT_CUTTER` are excluded unless `--allow-frequent-cutter-nickases` is passed

#### `cruncher snapback released-solve`

Search the released-product dual-enzyme catalog space, materialize ranked hits, and optionally render one plot per hit.

Examples:

* `uv run cruncher snapback released-solve --workspace-root src/dnadesign/cruncher/workspaces/demo_released_snapback --nick-preset neb_nicking_v1 --nick-additional-preset thermo_nicking_v1 --release-preset type_iis_release_v1 --release-variant-id BspQI --allow-top-active-routes --allow-precut-footprint-outside-active-product --json`
* `uv run cruncher snapback released-solve --workspace-root src/dnadesign/cruncher/workspaces/demo_released_snapback --nick-preset neb_nicking_v1 --nick-additional-preset thermo_nicking_v1 --release-preset type_iis_release_v1 --release-variant-id BspQI --allow-top-active-routes --allow-precut-footprint-outside-active-product --run-dir outputs/released_solve --materialize-top-k 16 --render-format pdf --emit-renders --force-overwrite`

Outputs:

* writes under `<workspace>/outputs/released_solve/`
* writes `analysis/solve_report.json` and `export/table__hits.csv`
* writes materialized released-product hit bundles under `analysis/materialized_hits/hit_<rank>/`
* writes per-hit `plots/released_hit_triptych.<fmt>` when `--emit-renders` is enabled
* writes route-policy, retained-partner fragment, and generic active-product columns in `table__hits.csv`

Notes:

* `released-solve` reuses the full released-target-search cross-product and does not stop at the first exact hit
* exact hits are materialized first when any exist; otherwise the top ranked near hits are materialized
* the command requires at least one explicit nickase source and one explicit release-enzyme source
* `--max-results` is automatically raised to at least `--materialize-top-k`
* CLI parsing delegates typed request and output construction to `app/snapback_cli_requests.py`; the command module stays on the UX side of the boundary
* the solve plot keeps `Nick / origin` at the left boundary and is rendered from the released-product projection payloads
* without route-policy flags, solved hits stay on `route_family=bottom_active_from_top_nick`; retained-active searches can materialize `route_family=top_active_from_bottom_nick`
* `--allow-top-active-routes` and `--allow-precut-footprint-outside-active-product` mirror the retained-active audit path from `released-target-search`
* `--release-variant-id` restricts materialization to one release-enzyme variant; the checked-in demo uses `BspQI`
* demo-only catalog entries are excluded unless `--allow-demo-hits` is passed
* nickases carrying `FREQUENT_CUTTER` are excluded unless `--allow-frequent-cutter-nickases` is passed

#### `cruncher snapback released-show`

Read one released-product bundle and print a path-oriented summary with drift checks.

Examples:

* `uv run cruncher snapback released-show --run outputs/released_design`
* `uv run cruncher snapback released-show --run outputs/released_design --json`

Notes:

* `released-show` accepts `released-design` explicit bundle roots only
* the command fails fast on manifest/status drift and projection/report inconsistency

#### `cruncher snapback show`

Read one snapback explicit or solve bundle and print its key artifact paths.

Examples:

* `uv run cruncher snapback show --run outputs/design`
* `uv run cruncher snapback show --run outputs/solve`
* `uv run cruncher snapback show --run outputs/solve --json`

Notes:

* `show` accepts explicit or solve bundle roots only
* the command fails fast on manifest/status drift, visual drift, and materialized-hit drift
* `show` is an integrity check, not a best-effort artifact browser

---

#### Scar-nick workflows

The scar-nick workflow is separate from `sample`, `cassette`, `snapback`, and
`yiu`. It expects one retained-scar spec at
`<workspace>/configs/scar_nick/<name>.scar_nick.yaml` and writes a deterministic
bundle under `<workspace>/outputs/scar_nick/<name>/`.

Deep contracts live in:

* [`../guides/scar_nick_workflow.md`](../guides/scar_nick_workflow.md)
* [`../../src/scar_nick/README.md`](../../src/scar_nick/README.md)
* [`nickase_catalog.md`](nickase_catalog.md)
* [`architecture.md`](architecture.md)

#### `cruncher scar-nick validate`

Validate one scar-nick spec and print a deterministic feasibility report.

Examples:

* `uv run cruncher scar-nick validate --spec configs/scar_nick/teto_upstream_processing.bbsI_hf.scar_nick.yaml`
* `uv run cruncher scar-nick validate --spec configs/scar_nick/teto_upstream_processing.bbsI_hf.scar_nick.yaml --json`

Notes:

* `--spec` must point to a `.scar_nick.yaml` file under a workspace
  `configs/scar_nick/` tree
* validation is read-only
* failures are strict schema, geometry, or policy blockers rather than
  best-effort warnings

#### `cruncher scar-nick design`

Validate one scar-nick spec and write the design bundle.

Examples:

* `uv run cruncher scar-nick design --spec configs/scar_nick/teto_upstream_processing.bbsI_hf.scar_nick.yaml`
* `uv run cruncher scar-nick design --spec configs/scar_nick/teto_upstream_processing.bbsI_hf.scar_nick.yaml --force-overwrite`

Outputs:

* writes under `<workspace>/outputs/scar_nick/<name>/`
* writes `meta/scar_nick_manifest.json` and `meta/scar_nick_status.json`
* writes candidate, pair-call, and nickase-geometry audit tables under
  `export/`
* writes terminal-nick visual contracts under `analysis/views/`
* writes `baserender_jobs/scar_nick_terminal_nick.job.yaml`

Notes:

* unsatisfied specs fail without writing a misleading success bundle
* related panels should stay in one workspace as separate specs

#### `cruncher scar-nick show`

Read one scar-nick run directory and print a path-oriented summary with drift
checks.

Examples:

* `uv run cruncher scar-nick show --run outputs/scar_nick/teto_upstream_processing_bbsI_hf`
* `uv run cruncher scar-nick show --run outputs/scar_nick/teto_upstream_processing_bbsI_hf --json`

Notes:

* `show` accepts explicit scar-nick run roots only
* the command fails fast on missing reports, visual artifacts, manifest drift,
  and BaseRender job drift

---

#### YIU workflows

The YIU workflow is separate from both `sample` and `cassette`. It expects one payload-centric spec file at
`<workspace>/configs/yiu/<name>.yiu.yaml`, validates the `split_yiu_payload_rendering_v4` contract, and writes a single payload bundle under the workspace-relative `output.bundle_dir` path, typically `<workspace>/outputs/<name>/`.
Treat the bundle directory as the source of truth; the mirrored operator PDF is optional and follows `output.published_plot_path` only when configured.

Deep contracts live in:

* [`../demos/demo_yiu_workspace.md`](../demos/demo_yiu_workspace.md)
* [`../guides/yiu_workflow.md`](../guides/yiu_workflow.md)
* [`reference/yiu_spec.md`](yiu_spec.md)
* [`reference/yiu_artifacts.md`](yiu_artifacts.md)

#### `cruncher yiu init-workspace`

Scaffold a YIU workspace root with one checked-in payload example spec, one machine runbook, and an `outputs/` surface for generated YIU artifacts.

Examples:

* `uv run cruncher yiu init-workspace yiu_lab`
* `uv run cruncher yiu init-workspace --output ./yiu_lab --force-overwrite`

Outputs:

* writes `configs/runbook.yaml`
* writes one payload-centric YIU spec under `configs/yiu/*.yiu.yaml`
* creates `outputs/`

#### `cruncher yiu validate`

Validate one payload-centric YIU spec, normalize it to one optimized payload object, and print the selected plan summary.

Examples:

* `uv run cruncher yiu validate --spec configs/yiu/<workflow>.yiu.yaml`
* `uv run cruncher yiu validate --spec configs/yiu/<workflow>.yiu.yaml --json`

#### `cruncher yiu render`

Validate a payload-centric YIU spec, publish the bundle, and optionally render the three payload views through BaseRender.

Examples:

* `uv run cruncher yiu render --spec configs/yiu/<workflow>.yiu.yaml`
* `uv run cruncher yiu render --spec configs/yiu/<workflow>.yiu.yaml --emit-renders`
* `uv run cruncher yiu render --spec configs/yiu/<workflow>.yiu.yaml --json`

Outputs:

* writes under the workspace-relative `output.bundle_dir` path, usually `<workspace>/outputs/<workflow>/`
* writes the operator-facing handoff summary `bundle_summary.json`
* writes the machine-facing bundle ledgers `bundle_manifest.json`, `normalized_payload.json`, and `visual_inventory.json`
* writes the published render contracts `payload_view.json`, `split_payload_view.jsonl` (JSONL rows), and `assembled_payload_view.json`
* writes one composite operator render `payload_views.pdf` when `--emit-renders` is set
* mirrors that composite PDF to `output.published_plot_path` when configured
* writes optional debug jobs under `baserender_jobs/` only when `output.emit_render_jobs_debug: true`

#### `cruncher yiu show`

Show the normalized payload bundle summary for one published YIU bundle directory.

Example:

* `uv run cruncher yiu show --bundle outputs/<workflow>`
* `uv run cruncher yiu show --bundle outputs/<workflow> --json`

Notes:

* text output surfaces one ligation summary line, one overhang summary, payload/split-left/split-right/assembled 5' to 3' reference-vs-mismatch-present rows, compact strand-aware mismatch edits (`PS` = payload strand, `AS` = opposite strand, 1-based payload positions), and PWM status; `--verbose` adds provenance, bundle contract, render/integrity detail, machine-facing artifact paths, and split-row debug lines
* default `--json` stays operator-focused and omits machine ledger paths plus normalized payload detail; `--verbose` additionally includes those paths, `motif_context`, `optimization_decision`, and `split_row_debug`

#### `cruncher visuals validate`

Validate a published render job through the public `dnadesign.baserender` API.

Example:

* `uv run cruncher visuals validate --job outputs/<workflow>/baserender_jobs/<view>.job.yaml`

#### `cruncher visuals run`

Run a published render job through the public `dnadesign.baserender` API.

Example:

* `uv run cruncher visuals run --job outputs/<workflow>/baserender_jobs/<view>.job.yaml`

---

#### Study workflows

#### `cruncher study list`

Lists Study specs and Study runs discovered under known workspace roots.

Inputs:

* optional `--workspace <name|index|path>` to scope listing to one workspace

Network:

* no

Examples:

* `cruncher study list`
* `cruncher study list --workspace demo_pairwise`

Notes:

* Studies are workspace-scoped: specs live under `<workspace>/configs/studies/*.study.yaml`.
* Study runs live under the standard root `<workspace>/outputs/studies/<study_name>/<study_id>/`.
* `--workspace` accepts a discovered workspace name/index or a direct workspace path.
* `study list` fails fast if a discovered spec or run metadata is invalid.

#### `cruncher study run`

Executes a Study spec (`*.study.yaml`) to run sweep factors x replicate seeds, optional replay sweeps, and aggregate plots.

Inputs:

* required `--spec <workspace>/configs/studies/<name>.study.yaml`
* optional `--resume` to continue from an existing manifest
* optional `--force-overwrite` to delete/recreate the deterministic study run dir

Network:

* no (uses local cache and local run artifacts)

Examples:

* `cruncher study run --spec configs/studies/diversity_vs_score.study.yaml`
* `cruncher study run --spec configs/studies/diversity_vs_score.study.yaml --resume`
* `cruncher study run --spec configs/studies/diversity_vs_score.study.yaml --force-overwrite`

Outputs:

* `<workspace>/outputs/studies/<study_name>/<study_id>/study/spec_frozen.yaml`
* `<workspace>/outputs/studies/<study_name>/<study_id>/study/study_manifest.json`
* `<workspace>/outputs/studies/<study_name>/<study_id>/tables/table__trial_metrics.parquet`
* `<workspace>/outputs/plots/study__<study_name>__<study_id>__plot__sequence_length_tradeoff.pdf`
* `<workspace>/outputs/plots/study__<study_name>__<study_id>__plot__mmr_diversity_tradeoff.pdf` (when replay enabled)

Notes:

* Study specs are strict (`extra=forbid`); unknown keys and invalid factor dot-paths fail fast.
* `study.base_config` is required and must exist; no CWD fallback.
* Trial definitions can be explicit (`trials`) or grid-expanded (`trial_grids`); at least one source is required.
* `study.schema_version` is v3 only (`study.schema_version: 3`).
* Trial and grid definitions use `factors`, not `overrides`.
* Studies inherit non-swept behavior from `configs/config.yaml`; only sweep-factor keys are allowed in study specs.
* Every swept factor must include the base-config value in the study domain.
* Trial-grid expansion is bounded (`<=500` combinations per grid and `<=500` total expanded trials).
* Study trials do not register entries in workspace `run_index.json`.
* `study.replays.mmr_sweep.enabled=true` requires persisted sequence artifacts (`sample.output.save_sequences=true`) for every trial after profile factors and requires replay diversity values to include the base-config diversity.
* Preflight validates lockfile, target readiness, and parse-readiness before any trial executes.
* When any trial fails, `study run` records errors and skips automatic summary generation; run `study summarize --allow-partial` to summarize successful subsets.

#### `cruncher study summarize`

Recomputes aggregate Study tables/plots from an existing study run directory.

Inputs:

* required `--run <study_run_dir>`
* optional `--allow-partial` to include only successful trials when some runs/artifacts are missing

Behavior:

* with `--allow-partial`, aggregate tables include `n_missing_*` annotations (`n_missing_total`, `n_missing_non_success`, `n_missing_run_dirs`, `n_missing_metric_artifacts`, `n_missing_mmr_tables`).
* if partial data was required and the frozen spec uses `exit_code_policy=nonzero_if_any_error`, command exits non-zero after writing refreshed outputs.

Example:

* `cruncher study summarize --run outputs/studies/diversity_vs_score/<study_id>`

#### `cruncher study show`

Prints Study status, trial counts, and key table/plot paths.

Inputs:

* required `--run <study_run_dir>`

Example:

* `cruncher study show --run outputs/studies/diversity_vs_score/<study_id>`

#### `cruncher study clean`

Deletes Study output artifact directories under `outputs/studies/...` for one workspace/study target.

Inputs:

* required `--workspace <name|index|path>`
* required `--study <study_name>`
* exactly one of:
  * `--id <study_id>` for one run
  * `--all` for all runs for that study in that workspace
* optional `--confirm` to execute deletion (without `--confirm`, command is dry-run only)

Behavior:

* cleans output artifacts only; never modifies `*.study.yaml`
* fail-fast contract for invalid workspace selector, missing study spec, missing run, or invalid flag combinations

Examples:

* `cruncher study clean --workspace demo_pairwise --study diversity_vs_score --id <study_id>`
* `cruncher study clean --workspace demo_pairwise --study diversity_vs_score --all --confirm`

---

#### Portfolio workflows

#### `cruncher portfolio run`

Aggregates selected completed runs across multiple workspaces into a deterministic handoff package.

Inputs:

* required `--spec <portfolio_workspace>/configs/<name>.portfolio.yaml`
* optional `--force-overwrite` to delete/recreate the deterministic portfolio run dir
* optional `--prepare-ready {prompt|skip|rerun}` for `prepare_then_aggregate` when some sources are already ready
* optional `--studies / --no-studies` to override `portfolio.studies.enabled` for this run only

Network:

* no (run artifacts only)

Examples:

* `cruncher portfolio run --spec configs/master_all_workspaces.portfolio.yaml`
* `cruncher portfolio run --spec configs/master_all_workspaces.portfolio.yaml --force-overwrite`
* `cruncher portfolio run --spec configs/master_all_workspaces.portfolio.yaml --prepare-ready skip`
* `cruncher portfolio run --spec configs/master_all_workspaces.portfolio.yaml --studies`
* `cruncher portfolio run --spec configs/master_all_workspaces.portfolio.yaml --no-studies`

Source preconditions (per source entry in spec):

* `analysis/reports/summary.json`
* `export/export_manifest.json` (with valid `files.elites` and `files.consensus_sites` paths)

Outputs:

* `<portfolio_workspace>/outputs/<portfolio_name>/<portfolio_id>/meta/manifest.json`
* `<portfolio_workspace>/outputs/<portfolio_name>/<portfolio_id>/meta/status.json`
* `<portfolio_workspace>/outputs/<portfolio_name>/<portfolio_id>/meta/logs/prepare__<source_id>.log` (when `execution.mode=prepare_then_aggregate`)
* `<portfolio_workspace>/outputs/<portfolio_name>/<portfolio_id>/tables/table__handoff_windows_long.<csv|parquet>`
* `<portfolio_workspace>/outputs/<portfolio_name>/<portfolio_id>/tables/table__handoff_elites_summary.<csv|parquet>`
* `<portfolio_workspace>/outputs/<portfolio_name>/<portfolio_id>/tables/table__handoff_consensus_sites_long.<csv|parquet>`
* `<portfolio_workspace>/outputs/<portfolio_name>/<portfolio_id>/tables/table__workspace_elites_consensus.md`
* `<portfolio_workspace>/outputs/<portfolio_name>/<portfolio_id>/tables/table__source_summary.<csv|parquet>`
* `<portfolio_workspace>/outputs/<portfolio_name>/<portfolio_id>/tables/table__study_summary.<csv|parquet>` (when `studies.enabled: true` and source `study_spec` is declared)
* `<portfolio_workspace>/outputs/<portfolio_name>/<portfolio_id>/tables/table__handoff_sequence_length.<csv|parquet>` (when `studies.enabled: true` and `studies.sequence_length_table.enabled: true`)
* `<portfolio_workspace>/outputs/<portfolio_name>/<portfolio_id>/plots/plot__source_tradeoff_score_vs_diversity.pdf` (when source diversity metrics are available)
* `<portfolio_workspace>/outputs/<portfolio_name>/<portfolio_id>/plots/plot__elite_showcase_cross_workspace.<pdf|png>` (when `plots.elite_showcase.enabled: true`)

Default portfolio table output is parquet with CSV mirrors (`artifacts.table_format=parquet`, `artifacts.write_csv=true`).

Notes:

* Portfolio specs are strict (`extra=forbid`); unknown keys and invalid paths fail fast.
* Sources are explicit only: no latest-run fallback, no workspace auto-selection fallback.
* Source selection uses source run manifest `top_k` and export manifest tables `files.elites` and `files.consensus_sites`; there is no portfolio-level top-k setting.
* Source run manifest `top_k` must match export elites row count for each source.
* Source run manifest stage must be `sample`.
* `run_dir` must resolve inside its declared `workspace`.
* For single-set workspaces, use `run_dir: outputs`; for multi-set workspaces use the specific set path (for example `outputs/set2_lexA-cpxR`).
* Portfolio schema is v3 only (`portfolio.schema_version: 3`).
* `execution.mode`:
  * `aggregate_only`: aggregate current source runs only
  * `prepare_then_aggregate`: execute each source `prepare.runbook` + `prepare.step_ids` before aggregation
* `execution.max_parallel_sources` controls source preparation concurrency in `prepare_then_aggregate` (default `4`, must be `>= 1`).
* In `prepare_then_aggregate`, every source must provide a runbook path inside its source workspace and a non-empty step list.
* `studies.enabled` defaults to `false`; set `studies.enabled: true` to include study orchestration in portfolio runs.
* `--studies / --no-studies` overrides `studies.enabled` at runtime for one invocation without editing the spec file.
* Optional source `study_spec` adds deterministic study summary rows into `table__study_summary` only when `studies.enabled: true`.
* If `studies.enabled: true` and `study_spec` is declared, the deterministic study run and `table__trial_metrics_agg.parquet` must exist (or be produced by prepare steps).
* Optional `studies.ensure_specs` is enforced only when `studies.enabled: true`, and auto-runs/resumes missing/incomplete study runs.
* Optional `studies.sequence_length_table` is global at portfolio scope and selects the first `top_n_lengths` shortest `sequence_length` rows per source when enabled.
* `plots.elite_showcase` is enabled by default and renders a cross-workspace showcase using all processed source elites.
* Set `plots.elite_showcase.top_n_per_source` to a positive integer when you want to cap elites per source.
* `plots.elite_showcase.source_selectors` supports per-source multi-elite selection, with exactly one of `elite_ids` or `elite_ranks` per source selector.
* In `aggregate_only`, Cruncher preflights all listed sources and reports every missing/invalid source artifact with actionable nudges.
* In `prepare_then_aggregate`, `--prepare-ready` controls whether already-ready sources are reprocessed or skipped.
* If source preparation fails, Cruncher reports source id, runbook path, configured `step_ids`, preflight issues, and explicit `workspaces run` nudge commands.
* Portfolio nudges use `--workspace <source_workspace_path>` (resolved path), not workspace-name lookup, so they remain runnable for external workspaces.
* When preflight shows missing foundational run artifacts (for example missing run manifest/elites), the failure message includes a full-runbook nudge in addition to configured-step nudges.
* `--spec` must point to a `.portfolio.yaml` file path; passing a directory path fails fast with an explicit nudge.

#### `cruncher portfolio show`

Prints portfolio status plus table/plot paths for one portfolio run directory.

Inputs:

* required `--run <portfolio_run_dir>`

Example:

* `cruncher portfolio show --run outputs/master_all_workspaces/<portfolio_id>`

---

#### `cruncher export sequences`

Exports sequence-centric run tables for downstream wrappers and operators.

Inputs:

* CONFIG (explicit or resolved)
* exactly one run selector: `--run <run_name|run_dir>` (repeatable) or `--latest`
* sample run artifacts: `optimize/tables/elites.parquet`, `optimize/tables/elites_hits.parquet`, `meta/config_used.yaml`

Network:

* no (run artifacts only)

Examples:

* `cruncher export sequences --latest <config>`
* `cruncher export sequences --run sample/run_001 <config>`
* `cruncher export sequences --latest --table-format csv <config>`

Outputs (under each run):

* `export/table__elites.csv`
* `export/table__consensus_sites.<parquet|csv>`
* `export/export_manifest.json`

Notes:

* Fail-fast contract: duplicate `(elite_id, tf)` rows, out-of-bounds windows, non-numeric scores, or inconsistent motif widths terminate export with an explicit error.
* Export appends artifact entries to `meta/run_manifest.json` using stage `export`.
* Default table format is CSV unless `--table-format parquet` is set.

---

#### `cruncher notebook`

Generates a marimo notebook for interactive exploration.

Inputs:

* run directory (`<run_dir>`) and optional `--analysis-id` or `--latest`

Network:

* no (local artifacts only; marimo is a local dependency)

Example:

* `cruncher notebook <path/to/sample_run> --latest`

Notes:

* requires `marimo` to be installed (for example: `uv sync --locked`)
* useful when you want interactive slicing/filtering beyond static plots
* strict artifact contract: requires `analysis/reports/summary.json`, `analysis/manifests/plot_manifest.json`, and `analysis/manifests/table_manifest.json` to exist and parse, `analysis/reports/summary.json` must include a non-empty `tf_names` list, and `analysis/manifests/table_manifest.json` must provide `scores_summary`, `metrics_joint`, and `elites_topk` entries with existing files
* plot output status is refreshed from disk so missing files are shown accurately
* the Refresh button re-scans analysis entries and updates plot/table status without restarting marimo
* the notebook infers `run_dir` from its location; keep it under `<run_dir>/` or regenerate it
* plots are loaded from `analysis/manifests/plot_manifest.json`; the curated keys are `elite_score_space_context`, `chain_trajectory_sweep`, `elites_nn_distance`, `elites_showcase`, plus optional `chain_trajectory_video`, `health_panel`, and `optimizer_vs_fimo` entries when generated
* the notebook includes:
  * Overview tab with run metadata and explicit warnings for missing/invalid analysis artifacts
  * Tables tab with a Top-K slider and per-table previews from `analysis/manifests/table_manifest.json`
  * Plots tab with inline previews and generated/skipped status from `analysis/manifests/plot_manifest.json`

---

#### Discovery and inspection

#### `cruncher workspaces`

List discoverable workspaces and their config paths.

Inputs:

* optional `--root <workspace_parent_dir>` for explicit workspace discovery scope

Network:

* no

Example:

* `cruncher workspaces list`
* `cruncher workspaces list --root src/dnadesign/cruncher/workspaces`
* `cruncher workspaces run --runbook configs/runbook.yaml`
* `cruncher workspaces run --workspace demo_pairwise --step analyze_summary --step export_sequences_latest`
* `cruncher workspaces reset --root .`
* `cruncher workspaces reset --root . --confirm`
* `cruncher workspaces reset --root src/dnadesign/cruncher/workspaces --all-workspaces --confirm`

Notes:

* `workspaces list` is Cruncher's tool-local workspace and machine-runbook inventory. For repo-wide runbook discovery across tools, start with `docs/runbooks/README.md` or `uv run ops catalog list --section tool-sources`.
* `workspaces list` includes Study inventory columns: `Study Specs` and `Study Runs`, and reports workspace kind (`config+runbook` or `runbook-only`).
* `workspaces run` executes typed runbook steps from `configs/runbook.yaml` in fail-fast order.
* runbook steps are strict CLI-args only (`run: [<cruncher-subcommand>, ...]`); arbitrary shell is not supported.
* `workspaces run --step ...` filters to explicit step ids while preserving runbook order.
* when `--workspace` and a relative `--runbook` are both provided, `--runbook` resolves from the selected workspace root and must stay inside that workspace.
* `workspaces reset` is a confirm-gated workspace reset surface that preserves `inputs/` and `configs/` while removing generated state (`outputs/`, `.cruncher/`, transient cache files).
* `workspaces reset --all-workspaces` treats `--root` as a parent directory and applies the same reset contract to every discoverable child workspace.

---

#### `cruncher catalog`

Inspect cached motifs and site sets.

Use `catalog pwms` to compute PWMs from cached matrices or binding sites and
survey their lengths/bit scores (without sampling-time motif-width trimming), and `catalog logos` to render PNG logos for the
same selection criteria.

Inputs:

* CONFIG (explicit or resolved)

Network:

* no (catalog only)

Subcommands:

* `catalog list` — list cached motifs and site sets
* `catalog search` — search by TF name or motif ID
* `catalog resolve` — resolve a TF name to cached candidates
* `catalog show` — show metadata for a cached `<source>:<motif_id>`
* `catalog pwms` — summarize or export resolved PWMs (matrix or site-derived)
* `catalog export-densegen` — export DenseGen motif artifacts (one JSON per motif)
* `catalog export-sites` — export cached binding sites as CSV/Parquet for DenseGen
* `catalog logos` — render PWM logos for selected TFs or motif refs

Note: `catalog logos` is idempotent for identical inputs. If matching logos already exist
under `outputs/plots/`, it reports the existing path instead of writing new files.

Examples:

* `cruncher catalog list <config>`
* `cruncher catalog search <config> lexA --fuzzy`
* `cruncher catalog show <config> regulondb:RDBECOLITFC00214`
* `cruncher catalog pwms <config>`
* `cruncher catalog pwms --set 1 <config>`
* `cruncher catalog export-sites --set 1 --out densegen/sites.csv <config>`
* `cruncher catalog export-sites --set 1 --densegen-workspace demo_tfbs_baseline <config>`
* `cruncher catalog export-densegen --set 1 --out densegen/pwms <config>`
* `cruncher catalog export-densegen --set 1 --densegen-workspace demo_sampling_baseline <config>`
* `cruncher catalog logos --set 1 <config>`

`catalog export-densegen` and `catalog export-sites` accept `--densegen-workspace` (packaged DenseGen
workspace name, explicit workspace path, or name under `DNADESIGN_DENSEGEN_WORKSPACES_ROOT`).
When provided, outputs default to the workspace `inputs/` locations and must stay within that directory.
`catalog export-densegen` removes existing artifact JSONs for the selected TFs by default; use
`--no-clean` to keep prior artifacts.

---

#### `cruncher discover`

Discover motifs from cached binding sites using MEME Suite (STREME or MEME).

Inputs:

* CONFIG (explicit or resolved)

Network:

* no (local; requires MEME Suite CLI tools via PATH or tool_path/MEME_BIN)

Subcommands:

* `discover motifs` — run STREME/MEME per TF and ingest discovered motifs into the catalog
* `discover check` — validate that MEME Suite tools are available and report versions

Examples:

* `cruncher discover motifs --set 1 <config>`
* `cruncher discover motifs --tf lexA --tf cpxR --tool streme <config>`
* `cruncher discover motifs --tf lexA --tf cpxR --tool meme <config>`
* `cruncher discover motifs --tf lexA --tf cpxR --tool meme --meme-mod oops <config>`
* `cruncher discover motifs --tf lexA --tf cpxR --tool meme --meme-mod oops --meme-prior addone <config>`
* `cruncher discover motifs --tf lexA --tf cpxR --tool streme --source-id meme_suite_streme <config>`
* `cruncher discover motifs --tf lexA --tf cpxR --tool meme --meme-mod oops --meme-prior addone --source-id meme_suite_meme <config>`
* `cruncher discover motifs --tf lexA --tool streme --replace-existing <config>`
* `cruncher discover motifs --tool-path /opt/meme/bin --tool streme <config>`
* `cruncher discover check <config>`

Notes:
* `tool=auto` selects STREME when there are enough sequences; use `--tool meme` if STREME is not installed.
* Discovery reads cached binding sites (run `cruncher fetch sites` first).
  Discovery always uses cached sites regardless of `catalog.pwm_source`.
* By default discovery uses raw cached site sequences. Use `--window-sites` (or
  `discover.window_sites=true`) to pre-window with `catalog.site_window_lengths`
  before running MEME/STREME.
  If enabled without window lengths for a TF, discovery exits with a helpful error.
* If `--minw/--maxw` are omitted (and unset in config), Cruncher passes no width flags and
  MEME/STREME uses its own defaults.
* `discover motifs` output `Tool width` is the discovery-time motif length from MEME/STREME;
  `Width bounds` reports `minw/maxw` used for discovery (`tool_default` means unset).
  Sampling-time constraints (`sample.motif_width`) are applied later during `sample`.
* Use `cruncher targets stats` to set `--minw/--maxw` from site-length ranges.
* If you plan to run both MEME and STREME, set distinct `discover.source_id` values between runs to avoid lock ambiguity.
  You can also pass `--source-id` per run to avoid editing config.
* By default discovery replaces previous discovered motifs for the same TF/source
  (`discover.replace_existing=true`). Pass `--keep-existing` to retain historical runs.
* `--meme-mod` applies to MEME only; use it when each sequence is expected to contain one site.
* `--meme-prior` applies to MEME only; `addone` is a good default for sparse site sets.
* Use `--tool-path` or the `MEME_BIN` environment variable to point at a specific install.
  Relative `--tool-path` values resolve from the workspace root.
* MEME Suite is a system dependency; install `streme`/`meme` via your system package manager,
  pixi, or the official MEME Suite installer, and ensure they are discoverable.
  If you use the repo's pixi toolchain, run `pixi run cruncher -- discover ...` so MEME is on PATH
  (place `-c/--config` after the subcommand when using pixi tasks).
* See [MEME Suite dependency guide](../guides/meme_suite.md) for a reproducible setup pattern.

---

#### `cruncher doctor`

Fail-fast environment checks for external dependencies (currently MEME Suite).

Inputs:

* optional CONFIG (explicit or resolved)

Network:

* no (local)

Examples:

* `cruncher doctor <config>`
* `cruncher doctor --tool streme --tool-path /opt/meme/bin <config>`

---

#### `cruncher targets`

Check readiness for the configured `regulator_sets` (or a category preview).

Inputs:

* CONFIG (explicit or resolved)

Network:

* no (catalog + config only)

Subcommands:

* `targets list`
* `targets status`
* `targets candidates`
* `targets stats`

Examples:

* `cruncher targets status <config>`
* `cruncher targets candidates --fuzzy <config>`
* `cruncher targets list --category Category2 <config>`

---

#### `cruncher sources`

List or inspect ingestion sources.

Inputs:

* optional CONFIG (explicit or resolved)

Network:

* `sources list` is local-only
* `sources datasets` and `sources summary --scope remote|both` contact upstream services

Subcommands:

* `sources list [config]` — list registered sources (auto-detects config in CWD to include local sources; pass CONFIG when elsewhere)
* `sources info <source> [config]`
* `sources datasets <source> [config]` — list HT datasets (if supported)
* `sources summary [config]` — summarize cache + remote inventory (supports JSON output, combined view)

Example:

* `cruncher sources list configs/config.yaml`
* `cruncher sources datasets regulondb configs/config.yaml --tf lexA`
* `cruncher sources summary configs/config.yaml`
* `cruncher sources summary --view combined configs/config.yaml`
* `cruncher sources summary --scope remote --format json configs/config.yaml`
* `cruncher sources summary --json-out summary.json configs/config.yaml`

Regulator inventory for a single source:

* `cruncher sources summary --source regulondb --scope cache configs/config.yaml`
* `cruncher sources summary --source regulondb --scope remote --remote-limit 200 configs/config.yaml`
* `cruncher sources summary --source regulondb --view combined configs/config.yaml`

Note:

* `sources list` attempts full config resolution (workspace/CWD). If none is found, it lists built-in sources only.
  Pass CONFIG (or set `CRUNCHER_CONFIG`/`CRUNCHER_WORKSPACE`) to include local sources from a workspace config.
* Some sources do not expose full remote inventories; use `--remote-limit` (partial counts)
  or `--scope cache` if you only need cached regulators.
* `sources datasets --dataset-source <X>` performs a strict row-level source filter on returned datasets.

Example output (cache, abridged; captured with `CRUNCHER_LOG_LEVEL=WARNING` and `COLUMNS=200`):

```bash
        Cache overview
      (source=regulondb)
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ Metric            ┃ Value   ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━┩
│ entries           │ 2       │
│ sources           │ 1       │
│ TFs               │ 2       │
│ motifs            │ 0       │
│ site sets         │ 2       │
│ sites (seq/total) │ 203/203 │
│ datasets          │ 0       │
└───────────────────┴─────────┘
                  Cache by source (source=regulondb)
┏━━━━━━━━━━━┳━━━━━┳━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Source    ┃ TFs ┃ Motifs ┃ Site sets ┃ Sites (seq/total) ┃ Datasets ┃
┡━━━━━━━━━━━╇━━━━━╇━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ regulondb │   2 │      0 │         2 │ 203/203           │        0 │
└───────────┴─────┴────────┴───────────┴───────────────────┴──────────┘
                  Cache regulators (source=regulondb)
┏━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ TF   ┃ Sources   ┃ Motifs ┃ Site sets ┃ Sites (seq/total) ┃ Datasets ┃
┡━━━━━━╇━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ cpxR │ regulondb │      0 │         1 │ 154/154           │        0 │
│ lexA │ regulondb │      0 │         1 │ 49/49             │        0 │
└──────┴───────────┴────────┴───────────┴───────────────────┴──────────┘
```

---

#### `cruncher cache`

Inspect cache integrity.

Inputs:

* CONFIG (explicit or resolved)

Network:

* no (cache only)

* `cache stats <config>` — counts of cached motifs and site sets
* `cache verify <config>` — verify cache paths exist on disk
* `cache clean <config>` — list generated `__pycache__` / `.pytest_cache` directories (dry-run by default; use `--apply` to delete)
  * default scan scope is package root (`src/dnadesign/cruncher/`)
  * use `--scope workspace|package|repo` to change scan scope
  * use `--root <dir>` to scan an explicit directory (overrides `--scope`)

---

#### `cruncher status`

Bird’s-eye view of cache, targets, and recent runs.

Inputs:

* CONFIG (explicit or resolved)

Network:

* no (cache + run index only)

Example:

* `cruncher status <config>`
* `cruncher status --runs 10 <config>`

---

#### `cruncher runs`

Inspect past run artifacts.

Inputs:

* CONFIG (explicit or resolved)
* run name or run directory path for `show/watch`

Network:

* no (run artifacts only)

* `runs list <config>` — list run folders (optionally filter by stage).
* `runs show <config> <run>` — show manifest + artifacts (run name or run dir)
* `runs latest <config> --set-index 1` — print most recent run for a regulator set
* `runs best <config> --set-index 1` — print best run by `best_score` for a regulator set
* `runs watch <config> <run>` — live progress snapshot (run name or run dir; reads `meta/run_status.json`, optionally `optimize/state/metrics.jsonl`)
* `runs rebuild-index <config>` — rebuild `<workspace>/.cruncher/run_index.json`
* `runs repair-index <config>` — validate and optionally remove index entries missing run directories/manifests (`--apply`)
* `runs clean <config> --stale` — mark stale `running` runs as `aborted` (use `--drop` to remove from the index)
* `runs prune <config>` — archive old runs under `<out_dir>/_archive/<stage>/<YYYY-MM>/` with deterministic retention (`--keep-latest`, `--older-than-days`; dry-run unless `--apply`)
  * use `--repair-index` to drop invalid run-index entries before pruning
  * without `--repair-index`, prune requires a valid run index and exits with an actionable repair command

Tip: inside a workspace you can drop the config argument entirely (for example,
`cruncher runs show <run>` or `cruncher runs list`).

Notes:
* `runs watch --plot` writes a live PNG plot to `<run_dir>/live/live_metrics.png`.
* `runs watch --metric-points` and `--metric-width` control the trend window size.
* `runs watch --plot-path` writes plots to a custom path; `--plot-every` controls refresh cadence.

---

#### `cruncher config`

Summarize effective configuration settings.

Inputs:

* optional config path (`--config/-c`), otherwise resolved from workspace/CWD

Network:

* no

Examples:

* `cruncher config`
* `cruncher config summary`
* `cruncher config summary <config>`
* `cruncher config --config <config>`

Note:

* you can pass `--config/-c` before or after the subcommand; if omitted, Cruncher
  resolves the config from the current directory.

---

#### `cruncher optimizers`

List available optimizer kernels.

Inputs:

* none

Network:

* no

Example:

* `cruncher optimizers list`

Note:

* Cruncher defaults to `gibbs_anneal`; this list is informational for kernel development.

---

#### Global options

* `--log-level INFO|DEBUG|WARNING` (or set `CRUNCHER_LOG_LEVEL`)
* `--config/-c <path>` (or set `CRUNCHER_CONFIG`) to pin a specific config file
* `--workspace/-w <name|index|path>` (or set `CRUNCHER_WORKSPACE`) to pick a workspace config


---

@e-south

# OPAL Dev Journal

Note: freeform working notes; prune/merge as they become cruft.

## 2026-01-23
- Starting from a clean baseline commit on `opal-dashboard-debug`.
- New campaign layout uses `configs/` for `campaign.yaml` + `plots.yaml`.
- Goal: anchor campaign IO to campaign root, add native XLSX ingest, and define a clean-slate demo reset.
- Demo inputs: `inputs/r0/vec8-b0.xlsx`; records at campaign root `records.parquet`.
- Decisions: outputs layout standardized to `outputs/ledger/` + `outputs/rounds/round_<k>/`.
- Label history schema uses value wrappers `{value,dtype,schema?}` for y_obs/y_pred; y_pred accepts non-numeric JSON.
- Demo reset command added (hidden from `--help`) to prune records.parquet, remove outputs/, and delete state.json.
- Ingest-y hardened: `--unknown-sequences` handling, default inference for missing bio_type/alphabet, and guards for
  duplicate sequences + deterministic id collisions.
- Auto-drop unknown sequences that lack X data (ingest-y safety).
- Registry loading now fails fast on built-in/entry-point import errors (shared loader helper).
- Prediction flow coerces outputs to numpy arrays and preserves null sequences.
- Ledger run/label sinks now write dataset directories; run_meta uses upsert-with-compaction on duplicate run_id.
- Campaign locks now store JSON (pid, ts) and detect stale locks with remediation guidance.
- Ingest resolves missing ids by sequence, treats known sequences as known (no accidental duplication), and fills
  missing bio_type/alphabet values via prompt/infer for new sequences.

### Campaign layout notes
- Campaign root is the center of gravity; configs live in `configs/` and IO stays under the campaign root.
- `records.parquet` remains in the campaign root (input + writeback target).
- Outputs are standardized:
  - `outputs/ledger/runs.parquet`, `outputs/ledger/labels.parquet`, `outputs/ledger/predictions/part-*.parquet`
  - `outputs/rounds/round_<k>/...` for per-round artifacts
- Keep `.opal/config` in root pointing to `configs/campaign.yaml` for auto-discovery.
- Campaign reset flow: `opal campaign-reset` (hidden) or `opal prune-source --scope any` + remove `outputs/`.
- Notebook template now emits campaign context markdown, data-source dropdown + table, and `__generated_with`.
- Moved notebook theme helpers to `analysis/dashboard/theme.py`; added guard test to keep notebooks dir marimo-only.
- Label history prediction writeback now deep-coerces objective params (e.g., numpy arrays → lists) for portability;
  added test coverage for the coercion.
- Removed plot `--quick` mode and its built-in quick plots; demo plots now rely solely on plots.yaml.
- Demo plot config now uses effect_scaled vs logic_fidelity and renames the plot for semantic clarity.
- Gitignore explicitly blocks non-demo opal campaign records.parquet while keeping demo as a tracked exception.
- prom60_eda now auto-loads records.parquet (no load button), with a regression test to prevent reintroducing gating.
- Campaign dashboard defaults (explorer X/Y/color) now live in campaign metadata and are read by prom60_eda.
- CampaignInfo carries dashboard metadata; tests cover the new parse path.
- SFXI overlay now coerces object-typed vec8 columns to list float for robust list ops; tests cover object vectors.
- Dataset explorer highlights boolean categories (observed/top-k) with gray backfill and layered sizing for true points.
- prom60_eda UI trimmed redundant campaign/status blocks, removed duplicate config dropdown, and hstacked cluster controls.
- Marimo bumped to 0.19.5 (lock + spec).
- UMAP controls now show raw column names (keys) instead of label aliases to avoid ambiguity.
- Fold-change plot exports now namespace logic fidelity as obj__logic_fidelity.
- prom60 explorer defaults updated to effect_scaled for the Y axis.

## 2026-01-24
- Round artifacts reorganized into subdirectories under each `outputs/rounds/round_<k>/`:
  `model/`, `selection/`, `labels/`, `metadata/`, `logs/` (no legacy flat layout).
- Round log, round_ctx, objective_meta, and feature_importance now resolve from the new subdirs.
- Selection outputs standardized to prefixed columns (removed `selection_score`; keep `pred__y_obj_scalar`).
- Dashboard exports now prefix canonical objective metrics (`obj__score`, `obj__logic_fidelity`, `obj__effect_scaled`,
  `obj__effect_raw`) to distinguish from overlay fields.
- SFXI scatter now attaches `cluster__*` + `densegen__*` columns for hue selection; added util helper + test.
- prom60_eda: preserve SFXI color selection across setpoint updates; dataset explorer defaults now prefer
  `opal__view__score`/`opal__view__effect_scaled` and skip `__row_id` when possible.
- Guarded SFXI color state updates to avoid reactive rerun loops when opening prom60_eda.
- Dataset explorer hue registry now allows higher-cardinality categories (max_unique=100) for cluster labels.
- SFXI hue registry now allows higher-cardinality categories so cluster labels appear in color-by options.

## 2026-01-26
- Starting prom60 SFXI diagnostics + uncertainty work on branch `opal-dashboard-extend`.
- Decisions: setpoint sweep library uses 16 truth tables + current setpoint; factorial size uses `effect_scaled`.
- Uncertainty contract will support both `score` and `y_hat` kinds (default `score`).
- Added SFXI diagnostic chart builders (support, uncertainty, intensity scaling) and CLI plot plugins.
- prom60_eda now includes a Diagnostics / AL Guidance column with factorial, sweep, support, uncertainty,
  and intensity scaling panels plus derived-metrics status.
- Derived metrics (nearest gate, dist-to-labeled logic/X, uncertainty) are attached to df_view for hue options.
- UMAP explorer overlays labeled points; tests cover hue registry and chart layering.

## 2026-01-27
- prom60_eda headless marimo errors fixed (duplicate variable names, UI value access in creation cell).
- Dashboard UMAP tooltips now typed to avoid empty-data Altair errors.
- Demo plots config expanded to include all SFXI diagnostics; DEMO doc updated for setpoint plot + ingest-y replace.
- Facade read_predictions now filters before projection; added test for round filtering without as_of_round column.
- SFXI plot helpers: setpoint parsing fixed for tuple/list values; added unit test.
- sfxi_uncertainty now coerces polars Series y_ops to list; added unit test.
- Fixed setpoint sweep pool clip fractions to use label-derived denom; added regression test.
- prom60_eda now flags setpoint sweep when it falls back to labels-as-of.
- SFXI diagnostic charts now share a compact plot style + constrained layout to reduce label overlap.
- Setpoint sweep + intensity scaling subtitles include denom definition for clarity.
- Removed unused intensity median/IQR helpers from dashboard SFXI view.
- Demo doc updated for non-interactive ingest-y (`--yes`).
- Demo doc notes `opal notebook generate --force` when rerunning.
- Setpoint sweep/intensity scaling plots now use current-round labels (objective-consistent); prom60_eda no longer falls back to labels-as-of.
- Notebook template now includes a plot gallery dropdown for outputs/plots, filtered by objective (SFXI-only when relevant).
- campaign-reset removes generated notebooks; opal campaign notebooks are gitignored explicitly.
- Notebook template now escapes gallery newline literals; added AST-parse test to prevent invalid Python output.
- Notebook template removed the tri-plot Altair panel in favor of a single plot gallery dropdown; simplified UI controls.
- Plot gallery avoids marimo variable redefinition errors by using unique loop variable names.
- Plot gallery cells now avoid early returns, use unique tag variables, and pass marimo check.
- prom60_eda diagnostics refactored into `analysis/dashboard/charts/diagnostics_guidance.py` with structured panel outputs and overlay-aware view joins.
- prom60_eda column layout updated so OPAL artifact model, Diagnostics, and Export are in columns 7/8/9 respectively.
- Diagnostics panel no longer renders the derived-metrics status markdown; derived-metrics notes remain.
- Added diagnostics sizing helpers and tightened Matplotlib diagnostics styling for white backgrounds + black text.
- Added Altair-based diagnostics scatter charts (factorial/support/uncertainty) with overlay-aware hue options.
- prom60_eda dropdowns now persist in-session state for diagnostics, UMAP, cluster plots, and dataset explorer.
- Diagnostics now resolve active-record vec8s from label history (pred history first), with explicit source notes.
- Setpoint sweep output now includes setpoint vector labels and renders as a heatmap (metrics × setpoints).
- Diagnostics panel text now includes didactic explanations for each plot.
- Notebook template now emits `marimo.App(width="medium", strict=False)` to avoid strictness mismatches.
- Dashboard SFXI params now validate setpoints via `sfxi_math.parse_setpoint_vector` (invalid values raise).
- Removed unused `fallback_percentile` plumbing from prom60_eda and SFXI params/export.
- Added test coverage for invalid SFXI setpoint values in dashboard utils.
- Removed setpoint decomposition plot from dashboard + CLI; docs/demo config updated accordingly.
- Diagnostics and SFXI plot artifacts now render full datasets; sampling params removed and rejected.
- Support diagnostics default hue now prefers nearest gate class; added clearer factorial math + uncertainty provenance notes.
- Setpoint sweep default view omits denom to preserve heatmap contrast.
- Refactored RF uncertainty to streaming ensemble score std (no `predict_per_tree`, no `y_hat` modes), added
  ensemble protocol under `analysis/sfxi`, and moved y-ops inverse helper out of dashboard.

## 2026-01-28
- Cached parquet loads for dashboards using a path+mtime key to avoid repeated reads.
- Added column non-null count helper and threaded it through hue registries to reduce repeated scans.
- Reduced redundant pred_y_hat vector conversion in prom60_eda diagnostics derivations.
- Added dashboard util test coverage for non-null count driven hue registry inclusion.
- Enforced fail-fast SFXI view contracts (no empty returns or silent invalid filtering).
- Moved derived diagnostics metrics (nearest logic, support distance, uncertainty attach) into a view helper.
- Removed diagnostics fallback panels; missing required inputs now raise explicit errors.

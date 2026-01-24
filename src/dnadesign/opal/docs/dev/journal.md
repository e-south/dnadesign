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

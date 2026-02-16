# OPAL Data Contracts (v2)

## Safety and validation

OPAL is assertive by default and fails fast on inconsistent inputs.

- `opal validate` checks essentials + X presence; if Y exists it must be finite and expected length.
- `label_hist` is required input for `run`/`explain` and the canonical dashboard source.
- Labels in Y but missing from `label_hist` are rejected.
- Ledger writes are strict: unknown columns are errors (override only with `OPAL_LEDGER_ALLOW_EXTRA=1`).
- Duplicate handling on ingest is explicit via `ingest.duplicate_policy` (`error|keep_first|keep_last`).

## Records schema

Required columns in `records.parquet`:

| column | type | notes |
| --- | --- | --- |
| `id` | string | unique per record |
| `bio_type` | string | `"dna"` or `"protein"` |
| `sequence` | string | case-insensitive |
| `alphabet` | string | e.g. `dna_4` |

X and Y representation:

- X: Arrow `list<float>` or JSON array string; fixed length across used rows
- Y: Arrow `list<float>`; label history stored in `opal__<campaign>__label_hist`

## Records label history (OPAL-managed)

| column | type | purpose |
| --- | --- | --- |
| `opal__<slug>__label_hist` | list<struct> | Append-only per-record history of observed labels and run-aware predictions. |

Prediction entries store objective channel metadata and selected metrics (`score_ref`, `uncertainty_ref`) so readers can reconstruct selection semantics without implicit defaults.

## Ledger output schema (append-only)

### labels (`outputs/ledger/labels.parquet`)

- `event`: `"label"`
- `observed_round`, `id`, `sequence` (if available)
- `y_obs`: `list<float>`
- `src`, `note`

### run_pred (`outputs/ledger/predictions/`)

- `event`: `"run_pred"`, plus `run_id`, `as_of_round`, `id`, `sequence`
- `pred__y_dim`, `pred__y_hat_model`
- `pred__score_selected`, `pred__score_ref`
- `pred__selection_score` (selection plugin score if different)
- `pred__uncertainty_selected`, `pred__uncertainty_ref`
- `pred__score_channels`, `pred__uncertainty_channels` (row-level channel payloads)
- `sel__rank_competition`, `sel__is_selected`
- Optional row diagnostics under `obj__*`
- Contract checks are strict: all row-level vectors must match candidate count; score/uncertainty vectors and channel payload values must be finite; emitted uncertainty must be non-negative.

### run_meta (`outputs/ledger/runs.parquet`)

- `event`: `"run_meta"`, plus `run_id`, `as_of_round`
- Config snapshot: `model__*`, `x_transform__*`, `y_ingest__*`, `objective__*`, `selection__*`, `training__y_ops`
- Objective declarations: `objective__defs_json`
- Selection channel refs: `selection__score_ref`, `selection__uncertainty_ref`
- Counts + summaries: `stats__*`, `objective__summary_stats`, `objective__denom_*`
- `stats__unc_mean_sd_targets` is the mean of the selected uncertainty channel for the run when uncertainty is emitted; otherwise null.
- `selection__score_ref` is always required and non-empty; `selection__uncertainty_ref` is null or a non-empty channel ref.
- `objective__denom_percentile` is populated only when the objective emits denominator-percentile metadata; otherwise null.
- Provenance: `artifacts` (paths + hashes), `schema__version`, `opal__version`

## Channel conventions

- Score channel refs: `<objective_name>/<score_channel_name>`
- Uncertainty channel refs: `<objective_name>/<uncertainty_channel_name>`
- `selection.params.score_ref` must resolve to an emitted score channel.
- `selection.params.uncertainty_ref` is required for uncertainty-driven selection (for example `expected_improvement`).

## Design notes

- Keep row-level diagnostics in `run_pred`, run-level summaries in `run_meta`.
- Prefer explicit channels and references over implicit single-score columns.
- Treat `schema__version` as compatibility guardrail for evolution.

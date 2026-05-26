## OPAL Data Contracts

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-25


This page documents the data and ledger contracts that OPAL reads and writes during ingest and round execution. Use it to validate schema expectations for `records.parquet`, shared label sidecars, label history, and append-only ledger sinks.

### Safety and validation

OPAL is assertive by default and fails fast on inconsistent inputs.

- `opal validate` checks essentials plus a non-null, finite, fixed-length X column; if Y exists it must be finite and expected length.
- `campaign_history` label sources require `label_hist` for `run`/`explain`.
- `usr_sidecar` label sources require the configured sidecar to exist for
  `run`/`explain` and contain only candidate IDs present in `records.parquet`.
  `opal validate` reports a missing pre-ingest sidecar as
  `label_source.exists=false` and validates schema once the file exists.
- `usr_sidecar` campaigns require explicit `writeback.prediction_records`;
  `ledger_only` keeps run predictions out of the shared candidate table.
- `usr_sidecar` appends are serialized by a local sidecar path lock; this
  protects local multi-campaign ingest, not distributed multi-host writes.
- Labels in Y but missing from `label_hist` are rejected for
  `campaign_history` runs; operators must ingest labels or explicitly attach
  the current Y column to label history.
- Ledger writes are strict: unknown columns are errors (override only with `OPAL_LEDGER_ALLOW_EXTRA=1`).
- Duplicate handling on ingest is explicit via `ingest.duplicate_policy` (`error|keep_first|keep_last`).
- `verify-outputs` is strict: selection IDs must be unique, selected IDs must
  exist in the target run ledger predictions, and run-scoped ledger prediction
  IDs must be unique before score comparisons are trusted.
- Batched scoring is ID-strict. Candidate X may be streamed from Parquet in
  storage order and coalesced into score-sized chunks, but model predictions are
  realigned to the requested candidate ID order before objective scoring,
  selection, or ledger writes. Missing, extra, or duplicate streamed IDs are
  fatal errors.

### Records schema

Required columns in `records.parquet`:

| column | type | notes |
| --- | --- | --- |
| `id` | string | unique per record |
| `bio_type` | string | `"dna"` or `"protein"` |
| `sequence` | string | case-insensitive |
| `alphabet` | string | e.g. `dna_4` |

X and Y representation:

- X: canonical Arrow `fixed_size_list<float32>[x_dim]` or
  `fixed_size_list<float64>[x_dim]` in Parquet. Values must be non-null,
  finite, and fixed length across every used row.
- Noncanonical vector encodings such as ragged Arrow lists, scalar cells, or JSON
  array strings are import/normalization inputs only. They are not accepted as
  the runtime campaign contract.
- Y: Arrow `list<float>` when using a current-Y column. Training labels may
  instead come from a shared sidecar when `labels.source.kind: usr_sidecar`.

When `records.parquet` is generated from a larger representation artifact, it
may be an ordered subset of that artifact. It must not silently include
reference/control rows that are outside the declared OPAL candidate universe.

`opal run` and `opal review` both validate the configured X column through the
public `validate_x_parquet_column` contract. Invalid physical schema, nulls,
nonfinite values, or ragged vectors fail before campaign execution or review
evidence is treated as trustworthy.

### Shared observed-label sidecar

`usr_sidecar` label sources are dataset-local Parquet tables such as
`_opal/observed_labels.parquet`.

Required columns:

| column | type | purpose |
| --- | --- | --- |
| `id` | string | Candidate ID matching `records.parquet` |
| `observed_round` | int | Round or batch index at which the assay label became available |
| `batch_id` | string | Operator/study batch identifier |
| `y_space` | string | Label space such as `sfxi_vec8` |
| `y_obs` | list<float> | Observed assay vector |

Optional provenance columns include `src`, `ts`, schema metadata, and assay
artifact references. A configured shared sidecar is fail-fast: OPAL does not
fall back to campaign-local label history when the file is missing, malformed,
or contains unknown candidate IDs. Appends lock the sidecar path while OPAL
loads existing labels, applies the duplicate policy, and replaces the Parquet
file.

For `usr_sidecar` campaigns, the candidate table must not carry active observed
labels through campaign-local surfaces. OPAL rejects non-empty configured
current-Y values and campaign-local observed-label entries because they can make
the sidecar and records table disagree about the training set. In
`writeback.prediction_records: ledger_only`, any campaign-local
`opal__<slug>__label_hist` entries are rejected; prediction and selection truth
must live in the ledgers.

### Records label history (OPAL-managed)

| column | type | purpose |
| --- | --- | --- |
| `opal__<slug>__label_hist` | list<struct> | Append-only per-record observed labels for `campaign_history` campaigns and run-aware predictions only when `writeback.prediction_records: label_history`. |

Prediction entries store objective channel metadata and selected metrics (`score_ref`, `uncertainty_ref`) so readers can reconstruct selection behavior without implicit defaults.

Shared-label campaigns should use `writeback.prediction_records: ledger_only`
unless mutating the shared `records.parquet` with campaign prediction history is
an explicit operator choice. In `ledger_only` mode, `run_pred` and `run_meta`
ledgers are the prediction/selection truth.

### Ledger output schema (append-only)

Append-only ledger datasets:

`labels` (`outputs/ledger/labels.parquet`)

- `event`: `"label"`
- `observed_round`, `id`, `sequence` (if available)
- `y_obs`: `list<float>`
- `src`, `note`

`run_pred` (`outputs/ledger/predictions/`)

- `event`: `"run_pred"`, plus `run_id`, `as_of_round`, `id`, `sequence`
- `pred__y_dim`, `pred__y_hat_model`
- `pred__score_selected`, `pred__score_ref`
- `pred__selection_score` (selection plugin score if different)
- `pred__uncertainty_selected`, `pred__uncertainty_ref`
- `pred__score_channels`, `pred__uncertainty_channels` (row-level channel payloads)
- `sel__rank_competition`, `sel__is_selected`
- Optional row diagnostics under `obj__*`
- Contract checks are strict: all row-level vectors must match candidate count; score/uncertainty vectors and channel payload values must be finite; emitted uncertainty must be non-negative (some objective+selection paths enforce strict positivity, for example `sfxi_v1` uncertainty consumed by `expected_improvement`).

`run_meta` (`outputs/ledger/runs.parquet`)

- `event`: `"run_meta"`, plus `run_id`, `as_of_round`
- Config snapshot: `model__*`, `x_transform__*`, `y_ingest__*`, `objective__*`, `selection__*`, `training__y_ops`
- Objective declarations: `objective__defs_json`
- Selection controls: `selection__score_ref`, `selection__uncertainty_ref`, `selection__objective` (`maximize|minimize`), `selection__tie_handling`
- Counts + summaries: `stats__*`, `objective__summary_stats`, `objective__denom_*`
- `stats__unc_mean_sd_targets` is the mean of the selected uncertainty channel for the run when uncertainty is emitted; otherwise null.
- `selection__score_ref` is always required and non-empty; `selection__uncertainty_ref` is null or a non-empty channel ref.
- `objective__denom_percentile` is populated only when the objective emits denominator-percentile metadata; otherwise null.
- Provenance: `artifacts` (paths + hashes), `schema__version`, `opal__version`

### Channel conventions

- Score channel refs: `<objective_name>/<score_channel_name>`
- Uncertainty channel refs: `<objective_name>/<uncertainty_channel_name>`
- `selection.params.score_ref` must resolve to an emitted score channel.
- `selection.params.uncertainty_ref` is required for uncertainty-driven selection (for example `expected_improvement`).

### Design notes

- Keep row-level diagnostics in `run_pred`, run-level summaries in `run_meta`.
- Prefer explicit channels and references over implicit single-score columns.
- Treat `schema__version` as the schema-evolution guardrail.

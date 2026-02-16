## OPAL Architecture and Data Flow

This page describes how OPAL executes one round and how config keys map to runtime behavior.

### Round lifecycle

1. Load `configs/campaign.yaml` and validate schema + plugin names.
2. Resolve labels up to `--labels-as-of` from `opal__<slug>__label_hist`.
3. Build feature matrices with `transforms_x`.
4. Fit `model` and predict `y_pred` (and optional predictive std-dev).
5. Apply `training.y_ops` inversion to both mean and std-dev when configured.
6. Evaluate configured `objectives` into named score and uncertainty channels.
7. Run `selection` using explicit refs (`score_ref`, optional `uncertainty_ref`) and persist outputs.

### Runtime surfaces

- Source records: `records.parquet`
- Label history column: `opal__<slug>__label_hist`
- Round artifacts: `outputs/rounds/round_<k>/...`
- Ledger sinks:
  - `outputs/ledger/labels.parquet`
  - `outputs/ledger/predictions/`
  - `outputs/ledger/runs.parquet`

### Config to stage mapping

- `campaign`, `data`: workspace and dataset resolution.
- `transforms_y`: ingest-only label construction.
- `transforms_x`: feature matrix for training/scoring.
- `training.y_ops`: fit-time Y transforms and inference-time inversion.
- `model`: fit/predict implementation.
- `objectives`: score/uncertainty channel emission.
- `selection`: ranking policy over explicit channel refs.
- `scoring`: prediction batch size.
- `safety`: preflight guards before writes.

### Channel contract

- Objectives emit score channels and optional uncertainty channels.
- Selection reads only configured refs:
  - `selection.params.score_ref = "<objective>/<score_channel>"`
  - `selection.params.uncertainty_ref = "<objective>/<uncertainty_channel>"` for uncertainty-based methods.
- `objective_mode` and `tie_handling` are explicit required controls.

### Failure model

OPAL is fail-fast by design:
- unknown plugins fail at config load/validation
- unresolved score/uncertainty refs fail before selection
- non-finite/invalid model/objective/selection outputs fail before writeback
- ledger schema violations fail at write time

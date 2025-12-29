## OPAL vNext Migration Notes

This release makes **strict, breaking** changes to runtime carriers, plugin contracts, and ledger handling.

### Required updates

1. **Campaign config**
   - Add:
     ```yaml
     ingest:
       duplicate_policy: "error"  # or keep_first | keep_last
     ```
2. **Ledger usage**
   - `ledger.index.parquet` and `events.parquet` are removed.
   - Tools now read **typed sinks** only:
     - `outputs/ledger.predictions/`
     - `outputs/ledger.runs.parquet`
     - `outputs/ledger.labels.parquet`
3. **Runtime carriers**
   - Contracts are enforced for model/selection/objective/transform_x/y‑ops.
   - Inspect with `opal ctx show|audit|diff`.
4. **Label history**
   - `label_hist` is strictly validated on `run`/`explain`.
   - If you have legacy/malformed entries, run:
     ```bash
     opal label-hist repair --config <campaign.yaml> --apply
     ```
5. **Plugin signatures (breaking)**
   - `transform_x` factories must accept `params: dict`, and callables must accept `ctx`.
   - `transform_y` functions must accept `(df_tidy, params, ctx)`.
   - `model` factories must accept `params: dict` (no fallback call patterns).

6. **Predict requires RoundCtx when Y‑ops were used**
   - If a model was trained with `training.y_ops`, `opal predict` now requires
     `round_ctx.json` next to the model to invert Y‑ops.
   - If you do not have it, you must pass `--assume-no-yops` explicitly.

7. **State schema updates**
   - `state.json` round entries now include `run_id` and `round_log_jsonl`.
   - New states write `version: 2`. Existing states remain readable but will
     not have `run_id` for historical rounds.

### Legacy fallbacks removed

- `opal record-show` and `opal objective-meta` no longer accept `--legacy`.

### Troubleshooting

- **“Unknown ledger columns”**: Fix upstream code or set `OPAL_LEDGER_ALLOW_EXTRA=1` for an explicit override.
- **“manual labels present without label_hist”**: Re‑ingest with `opal ingest-y` or repair.

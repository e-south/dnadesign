## OPAL vNext Migration Notes

This release makes **strict, breaking** changes to ledger handling and label history validation.

### Required updates

1. **Campaign config**
   - Add:
     ```yaml
     ingest:
       duplicate_policy: "error"  # or keep_first | keep_last
     ```
2. **Ledger usage**
   - `ledger.index.parquet` is deprecated and no longer written.
   - Tools now read **typed sinks**:
     - `outputs/ledger.predictions/`
     - `outputs/ledger.runs.parquet`
     - `outputs/ledger.labels.parquet`
3. **Label history**
   - `label_hist` is strictly validated on `run`/`explain`.
   - If you have legacy/malformed entries, run:
     ```bash
     opal label-hist repair --config <campaign.yaml> --apply
     ```

### Legacy fallbacks (explicit only)

- `opal record-show` and `opal objective-meta` no longer auto-fallback to `events.parquet`.
- Use `--legacy` to read legacy sinks when needed.

### Troubleshooting

- **“Unknown ledger columns”**: Fix upstream code or set `OPAL_LEDGER_ALLOW_EXTRA=1` for an explicit override.
- **“manual labels present without label_hist”**: Re‑ingest with `opal ingest-y` or repair.

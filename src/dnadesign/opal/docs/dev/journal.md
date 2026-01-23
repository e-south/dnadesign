# OPAL Dev Journal

Note: freeform working notes; prune/merge as they become cruft.

## 2026-01-23
- Starting from a clean baseline commit on `opal-dashboard-debug`.
- New campaign layout uses `configs/` for `campaign.yaml` + `plots.yaml`.
- Goal: anchor campaign IO to campaign root, add native XLSX ingest, and define a clean-slate demo reset.
- Demo inputs: `inputs/r0/vec8-b0.xlsx`; records at campaign root `records.parquet`.

### Campaign layout notes (draft)
- Campaign root is the center of gravity; configs live in `configs/` and IO stays under the campaign root.
- `records.parquet` remains in the campaign root (input + writeback target).
- Current outputs mix `ledger.runs.parquet` (file) and `ledger.predictions/` (dir). Consider a single ledger folder for clarity:
  - `outputs/ledger/runs.parquet`, `outputs/ledger/labels.parquet`, `outputs/ledger/predictions/part-*.parquet`
  - `outputs/rounds/round_<k>/...` for per-round artifacts
- Keep `.opal/config` in root pointing to `configs/campaign.yaml` for auto-discovery.
- Demo reset flow: `opal prune-source --scope any` + remove `outputs/` to start clean.

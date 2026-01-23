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

### Campaign layout notes
- Campaign root is the center of gravity; configs live in `configs/` and IO stays under the campaign root.
- `records.parquet` remains in the campaign root (input + writeback target).
- Outputs are standardized:
  - `outputs/ledger/runs.parquet`, `outputs/ledger/labels.parquet`, `outputs/ledger/predictions/part-*.parquet`
  - `outputs/rounds/round_<k>/...` for per-round artifacts
- Keep `.opal/config` in root pointing to `configs/campaign.yaml` for auto-discovery.
- Campaign reset flow: `opal campaign-reset` (hidden) or `opal prune-source --scope any` + remove `outputs/`.

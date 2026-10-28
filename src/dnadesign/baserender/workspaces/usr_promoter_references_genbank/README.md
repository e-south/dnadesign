# SeqLedger Promoter References GenBank Render

BaseRender workspace for GenBank-backed rows in `usr_promoter_references`.

SeqLedger remains the source of truth. Generate the local file-contract projection through the SeqLedger `Dataset.scan(include_overlays=...)` API before rendering:

```bash
uv run python src/dnadesign/seqledger/scripts/export_genbank_baserender_projection.py \
  --dataset usr_promoter_references \
  --out src/dnadesign/baserender/workspaces/usr_promoter_references_genbank/inputs/input.parquet

uv run baserender job validate \
  --workspace usr_promoter_references_genbank \
  --workspace-root src/dnadesign/baserender/workspaces

uv run baserender job run \
  --workspace usr_promoter_references_genbank \
  --workspace-root src/dnadesign/baserender/workspaces
```

Generated inputs and rendered outputs stay local under `inputs/input.parquet` and `outputs/`.
The projection intentionally includes only rows with `seq_annot__features`; unannotated rows are skipped until they receive annotation overlays.

The checked-in contract is intentionally small:

- `job.yaml` defines the BaseRender visual contract and styling.
- `inputs/README.md` documents that `input.parquet` is generated from SeqLedger.
- `outputs/.gitkeep` keeps the output directory visible while rendered PNG/MP4/report artifacts remain local.

Expected current projection: 48 GenBank-annotated promoter-reference rows: 19 primer-flank-stripped MG1655 inserts plus 29 synthetic promoter standards.

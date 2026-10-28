# Inputs

`input.parquet` is generated from SeqLedger with:

```bash
uv run python src/dnadesign/seqledger/scripts/export_genbank_baserender_projection.py \
  --dataset usr_promoter_references \
  --out src/dnadesign/baserender/workspaces/usr_promoter_references_genbank/inputs/input.parquet
```

Do not hand-edit this projection; fix the SeqLedger dataset, overlays, or export helper instead.

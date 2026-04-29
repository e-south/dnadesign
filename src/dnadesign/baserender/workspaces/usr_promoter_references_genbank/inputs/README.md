# Inputs

`input.parquet` is generated from USR with:

```bash
uv run python src/dnadesign/usr/scripts/export_genbank_baserender_projection.py \
  --dataset usr_promoter_references \
  --out src/dnadesign/baserender/workspaces/usr_promoter_references_genbank/inputs/input.parquet
```

Do not hand-edit this projection; fix the USR dataset, overlays, or export helper instead.

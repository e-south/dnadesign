# Inputs

`input.parquet` is generated from USR with:

```bash
uv run usr export usr_sfxi_pdual10_densegen_promoters \
  --fmt parquet \
  --columns id,sequence,usr_label__primary,densegen__used_tfbs_detail \
  --out src/dnadesign/baserender/workspaces/usr_sfxi_pdual10_densegen_promoters/inputs/input.parquet
```

Do not hand-edit this projection; fix the USR dataset or export command instead.

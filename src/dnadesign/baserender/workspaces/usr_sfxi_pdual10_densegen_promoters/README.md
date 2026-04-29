# USR SFXI pDual10 DenseGen Promoters

BaseRender workspace for the USR-backed `usr_sfxi_pdual10_densegen_promoters` dataset.

USR remains the source of truth. Generate the workspace input from the materialized USR view before rendering:

```bash
uv run usr export usr_sfxi_pdual10_densegen_promoters \
  --fmt parquet \
  --columns id,sequence,usr_label__primary,densegen__used_tfbs_detail \
  --out src/dnadesign/baserender/workspaces/usr_sfxi_pdual10_densegen_promoters/inputs/input.parquet

uv run baserender job validate \
  --workspace usr_sfxi_pdual10_densegen_promoters \
  --workspace-root src/dnadesign/baserender/workspaces

uv run baserender job run \
  --workspace usr_sfxi_pdual10_densegen_promoters \
  --workspace-root src/dnadesign/baserender/workspaces
```

Generated inputs and rendered outputs stay local under `inputs/input.parquet` and `outputs/`.

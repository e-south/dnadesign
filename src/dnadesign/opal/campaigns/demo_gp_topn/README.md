# Demo Campaign: GP + SFXI + top_n

Guide: `src/dnadesign/opal/docs/guides/demos/gp-sfxi-topn.md`

Run from this directory:

```bash
cp ../demo/records.parquet ./records.parquet
uv run opal init -c configs/campaign.yaml
uv run opal validate -c configs/campaign.yaml
uv run opal ingest-y -c configs/campaign.yaml --round 0 --csv inputs/r0/vec8-b0.xlsx --unknown-sequences drop --if-exists replace --apply
uv run opal run -c configs/campaign.yaml --round 0
uv run opal verify-outputs -c configs/campaign.yaml --round latest
```

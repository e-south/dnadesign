## OPAL Workflow Pressure-Test Matrix

**Owner:** dnadesign-maintainers
**Last verified:** 2026-07-13


The matrix below exercises maintained workflows in isolated campaign copies.
Operational usage starts at:

- [Workflows](../index.md#workflows)

### What this does

Runs each demo campaign end-to-end in an isolated temp copy:

- `validate -> init -> ingest-y -> run -> verify-outputs`
- `ctx audit -> explain -> record-show -> predict -> plot`

### Matrix script (round 0)

```bash
set -euo pipefail

tmp_root="$(mktemp -d /tmp/opal-demo-audit-XXXXXX)"
echo "tmp_root=${tmp_root}"

for flow in demo_rf_sfxi_topn demo_gp_topn demo_gp_ei; do
  src="src/dnadesign/opal/campaigns/${flow}"
  dst="${tmp_root}/${flow}"

  cp -R "${src}" "${dst}"
  cp src/dnadesign/opal/campaigns/demo_rf_sfxi_topn/records.parquet "${dst}/records.parquet"

  uv run opal campaign-reset -c "${dst}/configs/campaign.yaml" --apply --no-backup
  uv run opal validate -c "${dst}/configs/campaign.yaml"
  uv run opal init -c "${dst}/configs/campaign.yaml"

  uv run opal ingest-y -c "${dst}/configs/campaign.yaml" \
    --round 0 \
    --csv "${dst}/inputs/r0/vec8-b0.xlsx" \
    --unknown-sequences drop \
    --if-exists replace \
    --apply

  uv run opal run -c "${dst}/configs/campaign.yaml" --round 0

  uv run opal verify-outputs -c "${dst}/configs/campaign.yaml" --view primary --round latest --json > "${dst}/verify_r0.json"
  uv run opal ctx audit -c "${dst}/configs/campaign.yaml" --round latest --json > "${dst}/ctx_r0.json"
  uv run opal explain -c "${dst}/configs/campaign.yaml" --round 1 --json > "${dst}/explain_r1.json"

  selected_id="$(uv run opal selection-set show -c "${dst}/configs/campaign.yaml" --view primary --round latest --json | uv run python -c 'import json,sys; print(json.load(sys.stdin)["rows"][0]["id"])')"
  uv run opal record-show -c "${dst}/configs/campaign.yaml" --id "${selected_id}" --run-id latest --json > "${dst}/record_r0.json"

  uv run opal predict -c "${dst}/configs/campaign.yaml" --round latest --out "${dst}/predict_r0.parquet"
  uv run opal plot -c "${dst}/configs/campaign.yaml" --view primary --name score_vs_rank_latest --round latest

  echo "FLOW_OK: ${flow}"
done
```

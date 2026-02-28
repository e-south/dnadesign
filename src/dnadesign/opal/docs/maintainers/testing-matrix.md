## OPAL Workflow Pressure-Test Matrix

**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-27


This page is for maintainers and CI-style validation. New users should start at:

- [Workflows](../index.md#workflows)

### What this does

Runs each demo campaign end-to-end in an isolated temp copy:

- `init -> validate -> ingest-y -> run -> verify-outputs`
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
  cp src/dnadesign/opal/campaigns/demo/records.parquet "${dst}/records.parquet"

  uv run opal campaign-reset -c "${dst}/configs/campaign.yaml" --apply --no-backup
  uv run opal init -c "${dst}/configs/campaign.yaml"
  uv run opal validate -c "${dst}/configs/campaign.yaml"

  uv run opal ingest-y -c "${dst}/configs/campaign.yaml" \
    --observed-round 0 \
    --in "${dst}/inputs/r0/vec8-b0.xlsx" \
    --unknown-sequences drop \
    --if-exists replace \
    --apply

  uv run opal run -c "${dst}/configs/campaign.yaml" --labels-as-of 0

  uv run opal verify-outputs -c "${dst}/configs/campaign.yaml" --round latest --json > "${dst}/verify_r0.json"
  uv run opal ctx audit -c "${dst}/configs/campaign.yaml" --round latest --json > "${dst}/ctx_r0.json"
  uv run opal explain -c "${dst}/configs/campaign.yaml" --labels-as-of 1 --json > "${dst}/explain_r1.json"

  selected_id="$(tail -n +2 "${dst}/outputs/rounds/round_0/selection/selection_top_k.csv" | head -n 1 | cut -d, -f1)"
  uv run opal record-show -c "${dst}/configs/campaign.yaml" --id "${selected_id}" --run-id latest --json > "${dst}/record_r0.json"

  uv run opal predict -c "${dst}/configs/campaign.yaml" --round latest --out "${dst}/predict_r0.parquet"
  uv run opal plot -c "${dst}/configs/campaign.yaml" --name score_vs_rank_latest --round latest

  echo "FLOW_OK: ${flow}"
done
```

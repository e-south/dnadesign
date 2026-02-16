## OPAL Demo Flows

Each demo flow is campaign-scoped and documented as a full command narrative.
Use one campaign directory per flow so outputs/state stay isolated.

If you are new to OPAL, start with **RF + SFXI + top_n** first.

## Flow index

| Flow | Use-case | Campaign | Guide |
| --- | --- | --- | --- |
| RF + SFXI + top_n | deterministic baseline | `src/dnadesign/opal/campaigns/demo_rf_sfxi_topn/` | [RF + SFXI + top_n](./rf-sfxi-topn.md) |
| GP + SFXI + top_n | GP model with deterministic ranking | `src/dnadesign/opal/campaigns/demo_gp_topn/` | [GP + SFXI + top_n](./gp-sfxi-topn.md) |
| GP + SFXI + expected_improvement | uncertainty-aware acquisition | `src/dnadesign/opal/campaigns/demo_gp_ei/` | [GP + SFXI + expected_improvement](./gp-sfxi-ei.md) |

## Shared setup note

Each flow bootstraps campaign-local `records.parquet` from:

- `src/dnadesign/opal/campaigns/demo/records.parquet`

Each flow guide includes:

- optional clean rerun (`opal campaign-reset --apply --no-backup`)
- `init/validate/ingest-y/run/verify-outputs`
- inspection commands in sequence: `ctx`, `explain`, `record-show`, `predict`, `plot`

Flow guides:

- [RF + SFXI + top_n](./rf-sfxi-topn.md)
- [GP + SFXI + top_n](./gp-sfxi-topn.md)
- [GP + SFXI + expected_improvement](./gp-sfxi-ei.md)

## Pressure-test matrix

Run all three flows in isolated temp copies and execute the same end-to-end command path used in the guides:

```bash
tmp_root="$(mktemp -d /tmp/opal-demo-audit-XXXXXX)"
for flow in demo_rf_sfxi_topn demo_gp_topn demo_gp_ei; do
  src="src/dnadesign/opal/campaigns/${flow}"
  dst="${tmp_root}/${flow}"
  cp -R "${src}" "${dst}"
  cp src/dnadesign/opal/campaigns/demo/records.parquet "${dst}/records.parquet"

  uv run opal campaign-reset -c "${dst}/configs/campaign.yaml" --apply --no-backup
  uv run opal init -c "${dst}/configs/campaign.yaml"
  uv run opal validate -c "${dst}/configs/campaign.yaml"
  uv run opal ingest-y -c "${dst}/configs/campaign.yaml" --round 0 --csv "${dst}/inputs/r0/vec8-b0.xlsx" --unknown-sequences drop --if-exists replace --apply
  uv run opal run -c "${dst}/configs/campaign.yaml" --round 0
  uv run opal verify-outputs -c "${dst}/configs/campaign.yaml" --round latest --json > "${dst}/verify_r0.json"
  uv run opal status -c "${dst}/configs/campaign.yaml"
  uv run opal runs list -c "${dst}/configs/campaign.yaml" --json > "${dst}/runs_r0.json"
  uv run opal ctx audit -c "${dst}/configs/campaign.yaml" --round latest --json > "${dst}/ctx_r0.json"
  uv run opal explain -c "${dst}/configs/campaign.yaml" --round 1 --json > "${dst}/explain_r1.json"
  head -n 6 "${dst}/outputs/rounds/round_0/selection/selection_top_k.csv"
  selected_id="$(tail -n +2 "${dst}/outputs/rounds/round_0/selection/selection_top_k.csv" | head -n 1 | cut -d, -f1)"
  uv run opal record-show -c "${dst}/configs/campaign.yaml" --id "${selected_id}" --run-id latest --json > "${dst}/record_r0.json"
  uv run opal predict -c "${dst}/configs/campaign.yaml" --round latest --out "${dst}/predict_r0.parquet"
  uv run opal plot -c "${dst}/configs/campaign.yaml" --name score_vs_rank_latest --round latest

  uv run opal ingest-y -c "${dst}/configs/campaign.yaml" --round 1 --csv "${dst}/inputs/r0/vec8-b0.xlsx" --unknown-sequences drop --if-exists replace --apply
  uv run opal run -c "${dst}/configs/campaign.yaml" --round 1 --resume
  uv run opal verify-outputs -c "${dst}/configs/campaign.yaml" --round latest --json > "${dst}/verify_r1.json"
done
echo "tmp_root=${tmp_root}"
```

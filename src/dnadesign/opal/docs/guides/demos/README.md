## OPAL Demo Flows

Each flow has a campaign-scoped demo directory plus a dedicated guide.

## Flow index

| Flow | Use-case | Campaign | Guide |
| --- | --- | --- | --- |
| RF + SFXI + top_n | Baseline deterministic ranking with fast training | `src/dnadesign/opal/campaigns/demo_rf_sfxi_topn/` | [RF + SFXI + top_n](./rf-sfxi-topn.md) |
| GP + SFXI + top_n | Same objective with uncertainty-capable model, deterministic selector | `src/dnadesign/opal/campaigns/demo_gp_topn/` | [GP + SFXI + top_n](./gp-sfxi-topn.md) |
| GP + SFXI + expected_improvement | Uncertainty-aware acquisition for exploration/exploitation | `src/dnadesign/opal/campaigns/demo_gp_ei/` | [GP + SFXI + expected_improvement](./gp-sfxi-ei.md) |

## Shared demo data model

These campaigns use campaign-local `records.parquet` to keep each demo run isolated.
Bootstrap each campaign by copying seed records from:

- `src/dnadesign/opal/campaigns/demo/records.parquet`

Each flow guide includes the exact command sequence.

## Quick pressure-test matrix

From repo root, run each flow in an isolated temp copy to avoid mutating committed demo state:

```bash
tmp_root="$(mktemp -d /tmp/opal-demo-audit-XXXXXX)"
for flow in demo_rf_sfxi_topn demo_gp_topn demo_gp_ei; do
  src="src/dnadesign/opal/campaigns/${flow}"
  dst="${tmp_root}/${flow}"
  cp -R "${src}" "${dst}"
  cp src/dnadesign/opal/campaigns/demo/records.parquet "${dst}/records.parquet"
  uv run opal init -c "${dst}/configs/campaign.yaml"
  uv run opal validate -c "${dst}/configs/campaign.yaml"
  uv run opal ingest-y -c "${dst}/configs/campaign.yaml" --round 0 --csv "${dst}/inputs/r0/vec8-b0.xlsx" --unknown-sequences drop --if-exists replace --yes
  uv run opal run -c "${dst}/configs/campaign.yaml" --round 0
  uv run opal verify-outputs -c "${dst}/configs/campaign.yaml" --round latest

  # Round-1 continuation for full closed-loop flow
  uv run opal ingest-y -c "${dst}/configs/campaign.yaml" --round 1 --csv "${dst}/inputs/r0/vec8-b0.xlsx" --unknown-sequences drop --if-exists replace --yes
  uv run opal run -c "${dst}/configs/campaign.yaml" --round 1 --resume
  uv run opal verify-outputs -c "${dst}/configs/campaign.yaml" --round latest
done
```

This matrix confirms all three end-to-end flows are operational with campaign-local input data.

To confirm the input data differentiates GP selectors, compare selected IDs from the latest GP runs:

```bash
uv run python - <<'PY'
import csv
import json
from pathlib import Path

tmp_root = Path("<replace-with-your-tmp_root>")
def latest_selection_ids(flow: str):
    verify = json.loads((tmp_root / flow / "verify_r1.json").read_text())["summary"]
    ids = []
    with open(verify["selection_path"], newline="") as f:
        for row in csv.DictReader(f):
            ids.append(row["id"])
    return set(ids)

topn = latest_selection_ids("demo_gp_topn")
ei = latest_selection_ids("demo_gp_ei")
print("intersection", len(topn & ei))
print("topn_only", sorted(topn - ei))
print("ei_only", sorted(ei - topn))
PY
```

## OPAL — Optimization with Active Learning

**OPAL** is an active-learning engine for biological sequence design.

It runs a strict, auditable round loop:

1. ingest labels,
2. train a model,
3. evaluate objective channels,
4. select candidates,
5. persist artifacts + ledger events.

The pipeline is plugin-driven (`transforms_x`, `transforms_y`, `model`, `objectives`, `selection`) so behavior is configured in YAML rather than hardcoded.

## Quick start (5 commands)

```bash
# 1) Initialize campaign workspace
uv run opal init -c path/to/configs/campaign.yaml

# 2) Validate config + records contracts
uv run opal validate -c path/to/configs/campaign.yaml

# 3) Ingest labels for observed round 0
uv run opal ingest-y -c path/to/configs/campaign.yaml --round 0 --csv path/to/labels.csv --apply

# 4) Train/score/select using labels with observed_round <= 0
uv run opal run -c path/to/configs/campaign.yaml --round 0

# 5) Inspect status and verify outputs
uv run opal status -c path/to/configs/campaign.yaml
uv run opal verify-outputs -c path/to/configs/campaign.yaml --round latest
```

Use `uv run opal --help` for command inventory.

## Demo flows

Use campaign-scoped demos for end-to-end workflows:

| Flow | Campaign | Guide |
| --- | --- | --- |
| RF + SFXI + top_n | `src/dnadesign/opal/campaigns/demo_rf_sfxi_topn/` | [RF + SFXI + top_n](./docs/guides/demos/rf-sfxi-topn.md) |
| GP + SFXI + top_n | `src/dnadesign/opal/campaigns/demo_gp_topn/` | [GP + SFXI + top_n](./docs/guides/demos/gp-sfxi-topn.md) |
| GP + SFXI + expected_improvement | `src/dnadesign/opal/campaigns/demo_gp_ei/` | [GP + SFXI + expected_improvement](./docs/guides/demos/gp-sfxi-ei.md) |

For the matrix runner and flow index, see [`docs/guides/demos/README.md`](./docs/guides/demos/README.md).

## Mental model

- `transforms_y` is ingest-time only: table -> canonical `y` labels.
- `transforms_x` is train/predict-time: stored X representation -> numeric model matrix.
- Objectives emit **named score/uncertainty channels**.
- Selection consumes explicit channel refs (`score_ref`, optional `uncertainty_ref`).
- Ledger sinks under `outputs/ledger/` are append-only and run-aware.

## Artifacts and ledgers

Per round (`outputs/rounds/round_<k>/`):

- `model/model.joblib`, `model/model_meta.json`
- `selection/selection_top_k.csv`
- `labels/labels_used.parquet`
- `metadata/round_ctx.json`, `metadata/objective_meta.json`
- `logs/round.log.jsonl`

Campaign-wide ledgers (`outputs/ledger/`):

- `labels.parquet` (`label` events)
- `predictions/` (`run_pred` events)
- `runs.parquet` (`run_meta` events)

## Campaign layout

```text
src/dnadesign/opal/campaigns/<campaign>/
├─ configs/campaign.yaml
├─ records.parquet
├─ state.json
└─ outputs/
```

## Documentation map

Use [`docs/README.md`](./docs/README.md) as the docs hub.

- Concepts
  - [Architecture](./docs/concepts/architecture.md)
  - [RoundCtx](./docs/concepts/roundctx.md)
  - [Strategy matrix](./docs/concepts/strategy-matrix.md)
- References
  - [Configuration](./docs/reference/configuration.md)
  - [Data contracts](./docs/reference/data-contracts.md)
  - [CLI](./docs/reference/cli.md)
  - [Plots](./docs/reference/plots.md)
  - [Plugin docs](./docs/reference/plugins/)
- Objectives
  - [SFXI objective math](./docs/objectives/sfxi.md)

## Config resolution

Pass `--config` explicitly, or set `OPAL_CONFIG` in your shell/CI environment.

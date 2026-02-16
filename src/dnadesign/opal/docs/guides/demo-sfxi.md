## OPAL Demo Matrix (Campaign-Scoped)

This page is the entrypoint for OPAL SFXI demo workflows.

Use one campaign per flow so runtime state, ledger outputs, and round logs stay isolated.

## Flow map

| Flow | Campaign directory | Full guide |
| --- | --- | --- |
| RF + SFXI + top_n | `src/dnadesign/opal/campaigns/demo_rf_sfxi_topn/` | [RF + SFXI + top_n](./demos/rf-sfxi-topn.md) |
| GP + SFXI + top_n | `src/dnadesign/opal/campaigns/demo_gp_topn/` | [GP + SFXI + top_n](./demos/gp-sfxi-topn.md) |
| GP + SFXI + expected_improvement | `src/dnadesign/opal/campaigns/demo_gp_ei/` | [GP + SFXI + expected_improvement](./demos/gp-sfxi-ei.md) |

## Shared setup pattern

Each campaign guide uses the same staged command sequence:

1. Bootstrap campaign-local records (`cp ../demo/records.parquet ./records.parquet`)
2. `opal init`
3. `opal validate`
4. `opal ingest-y --round <r>`
5. `opal run --round <r>`
6. `opal verify-outputs`
7. `opal status` / `opal runs list`

For full didactic walkthroughs, use the flow-specific docs in `./demos/`.

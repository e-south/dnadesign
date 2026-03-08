![infer banner](assets/infer-banner.svg)

infer runs model-agnostic sequence inference and writes namespaced outputs back to datasets with explicit fail-fast contracts.

## Documentation map

Read in this order:

1. [infer docs index by workflow](docs/README.md)
2. [infer docs index by type](docs/index.md)
3. [getting-started guide](docs/getting-started/README.md)
4. [CLI quickstart](docs/getting-started/cli-quickstart.md)
5. [workspaces guide](workspaces/README.md)
6. [operations index](docs/operations/README.md)
7. [SCC Evo2 GPU environment runbook](docs/operations/scc-evo2-gpu-uv-runbook.md)
8. [agnostic-model pressure-test runbook](docs/operations/pressure-test-agnostic-models.md)
9. [end-to-end pressure-test demo (infer + usr + ops + notify)](docs/tutorials/demo_pressure_test_usr_ops_notify.md)
10. [reference index](docs/reference/README.md)
11. [source-tree map](src/README.md)
12. [developer docs](docs/dev/README.md)
13. [repository docs index](../../../docs/README.md)

## Entrypoint contract

1. Audience: infer operators and maintainers.
2. Primary entrypoints: `uv run infer ...` and `python -m dnadesign.infer ...`.
3. Prerequisites: installed adapter dependencies (for example `evo2`), valid input alphabet, and writable target when using `--write-back`.
4. Verify next: [pressure-test runbook](docs/operations/pressure-test-agnostic-models.md).

## Boundary reminder

- USR write-back columns use this namespaced contract:
  - `infer__<model_id>__<job_id>__<out_id>`
- Infer writes USR outputs in chunk-sized attaches for resumable long runs.
- Infer reset is namespaced and dataset-scoped: `uv run infer prune --usr <dataset-id> --usr-root <usr-root>`.
- Resume scanning requires a readable USR `records.parquet`; unreadable tables fail fast.
- Cross-tool workflow routes are maintained in [repository docs index](../../../docs/README.md).

![construct banner](docs/assets/construct-banner.svg)

construct places focal DNA parts into larger sequence contexts. Use it to expand promoters and other parts into explicit regions, plasmids, and related template-backed constructs. Any canonical sequence record can serve as an input part or a template; the construct config decides the role.

## Start here

- Want the shortest validated run: start with [Getting started](docs/getting-started.md). Verify next with the [Outputs reference](docs/reference/outputs.md).
- Want one packaged shared-dataset tracer bullet before infer: start with [Run the packaged source-of-truth demo](workspaces/demo_promoter_swap_pdual10_source_of_truth/README.md). Verify next with the shared [Construct -> USR -> Infer source-of-truth runbook](../usr/docs/operations/construct-infer-source-of-truth-runbook.md).
- Want the downstream feature-matrix branch after construct expansion: start with [Promoter characterization feature matrix](../usr/docs/operations/promoter-characterization-feature-matrix.md).

## Documentation map

1. [Getting started](docs/getting-started.md): shortest path to a validated demo run or blank custom workspace.
2. [Docs overview](docs/README.md): choose the next document by task.
3. [Docs index](docs/index.md): choose the next document by type.
4. [Shared cross-tool handoff routes (USR-owned)](docs/README.md): route into shared USR-backed source-of-truth and downstream feature-matrix procedures after construct materializes a dataset.
5. [Workspaces guide](workspaces/README.md): scaffold a workspace or copy the packaged demo.
6. [Developer notes](docs/dev/README.md): maintainer notes, internal architecture, and journal entries.

## Primary entrypoints

- `uv run construct --help`
- `uv run construct validate config --config <path> --runtime`
- `uv run construct run --config <path>`
- `uv run construct seed import-manifest --manifest <path>`
- `uv run construct workspace init --id <workspace-id>`
- `uv run construct workspace doctor --workspace <workspace-dir>`

## Boundary reminder

- `construct` owns sequence realization, placement semantics, and `construct__*` lineage.
- USR owns dataset persistence, dataset ids, and downstream reuse.
- One construct job uses one template plus one or more placed parts.
- Larger studies stay explicit as multiple workspace projects, not one oversized config.
- Packaged workspaces default to workspace-local `outputs/usr_datasets`; shared USR roots are always explicit.

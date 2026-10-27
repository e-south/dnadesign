![permuter banner](assets/permuter-banner.svg)

Permuter generates sequence variants, scores them with pluggable evaluators,
and stores results in explicit dataset records. Runnable scopes live under
`workspaces/<scope>/config.yaml` and write generated artifacts to that scope's
local `outputs/` directory. Sibling tools use the public package facade at
`dnadesign.permuter`.

## Documentation

- [Permuter docs index](docs/README.md): CLI usage, data contracts, and Retron
  variant workflows.
- [CLI and data contracts](docs/cli-and-data-contracts.md): commands, dataset
  layout, built-in protocols, and output columns.
- [Repository docs index](../../../docs/README.md): cross-tool workflow routing.

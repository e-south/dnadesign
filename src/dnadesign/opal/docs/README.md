# OPAL Docs Hub

Use this page as the navigation entrypoint.

## Start paths

### New user

1. Read the [Quick start](../README.md#quick-start-5-commands).
2. Pick a flow from [Demo flows](./guides/demos/README.md).
3. Keep [CLI reference](./reference/cli.md) open while running commands.

### Campaign operator

- Round planning: [`opal explain`](./reference/cli.md#explain)
- Run execution: [`opal run`](./reference/cli.md#run)
- Runtime audit: [`opal ctx`](./reference/cli.md#ctx)
- Output validation: [`opal verify-outputs`](./reference/cli.md#verify-outputs)

### Plugin/config author

- Runtime contracts: [RoundCtx](./concepts/roundctx.md)
- Config schema: [Configuration](./reference/configuration.md)
- Strategy wiring: [Strategy matrix](./concepts/strategy-matrix.md)
- Plugin references: [`reference/plugins/`](./reference/plugins/)

## Core references

- Concepts
  - [Architecture and data flow](./concepts/architecture.md)
  - [RoundCtx and contract auditing](./concepts/roundctx.md)
  - [Model/selection strategy matrix](./concepts/strategy-matrix.md)
- Reference
  - [Configuration (`campaign.yaml`)](./reference/configuration.md)
  - [Data contracts and ledger schemas](./reference/data-contracts.md)
  - [CLI reference](./reference/cli.md)
  - [Plots reference](./reference/plots.md)
  - [Model plugins](./reference/plugins/models.md)
  - [Selection plugins](./reference/plugins/selection.md)
  - [X transforms](./reference/plugins/transforms-x.md)
  - [Y transforms](./reference/plugins/transforms-y.md)
- Objective math
  - [SFXI objective](./objectives/sfxi.md)

## Demo docs

- [Demo flow index](./guides/demos/README.md)
- [Demo matrix summary](./guides/demo-sfxi.md)

## Maintainer notes

- [Dev journal](./dev/journal.md)
- [PROM60 diagnostics notes](./internal/prom60_sfxi_diagnostics_plots.md)

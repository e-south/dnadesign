## OPAL Documentation

This documentation covers end-to-end workflows, plugin contracts and math, runtime concepts, and exact configuration and CLI references.
Use the workflow guides for operational command order, then use plugin/concept/reference pages for deeper details.

### Workflows

These guides are the primary user path and show complete command sequences for each supported flow.
- [RF + SFXI + top_n](./workflows/rf-sfxi-topn.md)
- [GP + SFXI + top_n](./workflows/gp-sfxi-topn.md)
- [GP + SFXI + expected_improvement](./workflows/gp-sfxi-ei.md)

### Plugin docs

These pages define plugin-level contracts, channel semantics, and model/objective/selection behavior.
- [Models](./plugins/models.md)
- [Gaussian Process behavior and math](./plugins/model-gaussian-process.md)
- [Selection](./plugins/selection.md)
- [Expected Improvement behavior and math](./plugins/selection-expected-improvement.md)
- [Objectives and channel refs](./plugins/objectives.md)
- [SFXI behavior and math](./plugins/objective-sfxi.md)
- [SPOP objective draft](./plugins/objective-spop.md)
- [X transforms](./plugins/transforms-x.md)
- [Y transforms and Y-ops](./plugins/transforms-y.md)

### Concepts

These pages describe runtime architecture and RoundCtx contract auditing behavior.
- [Architecture and data flow](./concepts/architecture.md)
- [RoundCtx and contract auditing](./concepts/roundctx.md)

### Reference

These pages are contract-oriented lookups for schema, data surfaces, CLI behavior, and plotting.
- [Configuration (`campaign.yaml`)](./reference/configuration.md)
- [Data contracts and ledgers](./reference/data-contracts.md)
- [CLI commands](./reference/cli.md)
- [Plots](./reference/plots.md)

### Maintainers

These pages are maintainer-focused notes and validation runbooks.
- [Development journal](./maintainers/journal.md)
- [History](./maintainers/history.md)
- [Workflow pressure-test matrix](./maintainers/testing-matrix.md)

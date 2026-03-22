## OPAL Documentation

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-16


This documentation covers end-to-end workflows, plugin contracts and math, runtime concepts, and exact configuration and CLI references. Use the workflow guides for operational command order, then use plugin/concept/reference pages for deeper details.

### Start here

1. Only if infer has already written the chosen feature column into USR, start with [USR dataset with infer-derived X -> OPAL active learning](./workflows/usr-infer-x-active-learning.md).
2. If you want a tool-local baseline campaign, start with [RF + SFXI + top_n](./workflows/rf-sfxi-topn.md).
3. If you need config or CLI lookup before running a workflow, jump to [Configuration (`campaign.yaml`)](./reference/configuration.md) or [CLI commands](./reference/cli.md).

### Workflows

These guides are the primary user path and show complete command sequences for each supported flow.
- [USR dataset with infer-derived X -> OPAL active learning](./workflows/usr-infer-x-active-learning.md): downstream active-learning path once infer has already written the chosen `X` column into a USR dataset.
- [RF + SFXI + top_n](./workflows/rf-sfxi-topn.md): baseline campaign flow from config to selected candidates.
- [GP + SFXI + top_n](./workflows/gp-sfxi-topn.md): GP-driven scoring flow with top_n selection.
- [GP + SFXI + expected_improvement](./workflows/gp-sfxi-ei.md): GP-driven scoring flow with EI selection.

### Plugin docs

These pages define plugin-level contracts, channel semantics, and model/objective/selection behavior.
- [Models](./plugins/models.md): model plugin contracts and configuration surface.
- [Gaussian Process behavior and math](./plugins/model-gaussian-process.md): GP assumptions, fitting behavior, and outputs.
- [Selection](./plugins/selection.md): selection-plugin contracts and channel requirements.
- [Expected Improvement behavior and math](./plugins/selection-expected-improvement.md): EI formulation and selection semantics.
- [Objectives and channel refs](./plugins/objectives.md): objective plugin contracts and channel naming rules.
- [SFXI behavior and math](./plugins/objective-sfxi.md): SFXI objective mechanics and output interpretation.
- [SPOP objective draft](./plugins/objective-spop.md): draft objective behavior and current constraints.
- [X transforms](./plugins/transforms-x.md): input transform contracts for feature generation.
- [Y transforms and Y-ops](./plugins/transforms-y.md): target transform contracts and label operations.

### Concepts

These pages describe runtime architecture and RoundCtx contract auditing behavior.
- [Architecture and data flow](./concepts/architecture.md): end-to-end runtime structure and component boundaries.
- [RoundCtx and contract auditing](./concepts/roundctx.md): round-state contract and audit semantics.

### Reference

These pages are contract-oriented lookups for schema, data surfaces, CLI behavior, and plotting.
- [Configuration (`campaign.yaml`)](./reference/configuration.md): campaign schema and field meanings.
- [Data contracts and ledgers](./reference/data-contracts.md): artifact schemas, ledgers, and persistence surfaces.
- [CLI commands](./reference/cli.md): command interfaces and argument contracts.
- [Plots](./reference/plots.md): plotting outputs, expectations, and usage.

### Maintainers

These pages are maintainer-focused notes and validation runbooks.
- [Development journal](./maintainers/journal.md): maintainer investigations and decision records.
- [History](./maintainers/history.md): chronology of major design and behavior changes.
- [Workflow pressure-test matrix](./maintainers/testing-matrix.md): repeatable validation matrix for workflow hardening.

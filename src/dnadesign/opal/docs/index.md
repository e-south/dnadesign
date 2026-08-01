## OPAL Documentation

**Owner:** dnadesign-maintainers
**Last verified:** 2026-08-01

Start with a workflow for command order. Use the plugin, concept, and reference
pages for mathematical and interface contracts.

### Start here

1. Only if infer has already written the chosen feature column into USR, start with [USR dataset with infer-derived X -> OPAL active learning](./workflows/usr-infer-x-active-learning.md).
2. For a tool-local baseline campaign, start with [RF + SFXI + top_n](./workflows/rf-sfxi-topn.md).
3. For configuration or CLI lookup, use [Configuration (`campaign.yaml`)](./reference/configuration.md) or [CLI commands](./reference/cli.md).
4. If a downstream study needs physically valid selected sequences, check the
   generic [candidate and label contracts](./reference/configuration.md#candidate-and-label-contracts)
   and the public [`selection-set`](./reference/cli.md#selection-set) surface
   before adding study-owned logistics.
5. For maintained examples and study-owned configs, use the
   [campaign route index](../campaigns/README.md).

### Workflows

- [USR dataset with infer-derived X -> OPAL active learning](./workflows/usr-infer-x-active-learning.md): downstream active-learning path once infer has already written the chosen `X` column into a USR dataset.
- [RF + SFXI + top_n](./workflows/rf-sfxi-topn.md): baseline campaign flow from config to selected candidates.
- [GP + SFXI + top_n](./workflows/gp-sfxi-topn.md): GP-driven scoring flow with top_n selection.
- [GP + SFXI + expected_improvement](./workflows/gp-sfxi-ei.md): GP-driven scoring flow with EI selection.

### Plugin docs

- [Models](./plugins/models/README.md): model plugin contracts and configuration surface.
- [Gaussian Process behavior and math](./plugins/models/gaussian-process.md): GP assumptions, fitting behavior, and outputs.
- [Selection](./plugins/selection/README.md): selection-plugin contracts and channel requirements.
- [Expected Improvement behavior and math](./plugins/selection/expected-improvement.md): EI formulation and selection semantics.
- [Objectives and channel refs](./plugins/objectives/README.md): objective plugin contracts and channel naming rules.
- [SFXI behavior and math](./plugins/objectives/sfxi.md): SFXI objective mechanics and output interpretation.
- [Response-Magnitude Feasibility (RMF) behavior and math](./plugins/objectives/response-magnitude-feasibility.md): non-compensatory response separation and reference-relative magnitude constraints.
- [Multistate Response Behavior math and contract](./plugins/objectives/multistate-response-behavior.md): threshold-free, strictly monotone response ordering, ON expression, and OFF suppression.
- [X transforms](./plugins/transforms/x.md): input transform contracts for feature generation.
- [Y transforms and Y-ops](./plugins/transforms/y.md): target transform contracts and label operations.

### Concepts

- [Architecture and data flow](./concepts/architecture.md): end-to-end runtime structure and component boundaries.
- [RoundCtx and contract auditing](./concepts/roundctx.md): round-state contract and audit semantics.

### Reference

- [Configuration (`campaign.yaml`)](./reference/configuration.md): campaign schema and field meanings.
- [Data contracts and ledgers](./reference/data-contracts.md): artifact schemas, ledgers, and persistence surfaces.
- [CLI commands](./reference/cli.md): command interfaces and argument contracts, including `selection-set` inspection/export for downstream handoffs.
- [Plots](./reference/plots.md): plotting outputs, expectations, and usage.
- [Review manifests](./reference/review-manifests.md): campaign review bundle schema and stale-artifact behavior.
- [Notebooks](./reference/notebooks.md): generated marimo notebook contract and public view-model surface.

### Maintainers

- [Development journal](./maintainers/journal.md): maintainer investigations and decision records.
- [History](./maintainers/history.md): chronology of major design and behavior changes.
- [Workflow pressure-test matrix](./maintainers/testing-matrix.md): repeatable validation matrix for workflow hardening.

---
doc_id: opal-docs
title: OPAL documentation
owner: dnadesign-maintainers
status: active
last_verified: 2026-08-08
---

# OPAL documentation

Start with the campaign round for command order. Objective pages describe
individual scoring plugins; they are not alternate OPAL lifecycles.

### Start here

1. Start with the [campaign round](./workflows/campaign-round.md) for the canonical validate, initialize, label, run, and verify loop.
2. If Infer has already written the chosen feature column into USR, use the [USR-to-OPAL route](./workflows/usr-infer-x-active-learning.md).
3. For configuration or CLI lookup, use [Configuration (`campaign.yaml`)](./reference/configuration.md) or [CLI commands](./reference/cli.md).
4. If a downstream study needs physically valid selected sequences, check the
   generic [candidate and label contracts](./reference/configuration.md#candidate-and-label-contracts)
   and the public [`selection-set`](./reference/cli.md#selection-set) surface
   before adding study-owned logistics.
5. For maintained examples and external campaign placement rules, use the
   [campaign route index](../campaigns/README.md).

### Workflows

- [Campaign round](./workflows/campaign-round.md): the one OPAL lifecycle, independent of model, objective, and selector choices.
- [USR dataset with infer-derived X -> OPAL active learning](./workflows/usr-infer-x-active-learning.md): use a feature column already written into a USR dataset.

### Plugin contracts

- [Models](./plugins/models/README.md): model plugin contracts and configuration surface.
- [Gaussian Process behavior and math](./plugins/models/gaussian-process.md): GP assumptions, fitting behavior, and outputs.
- [Selection](./plugins/selection/README.md): selection-plugin contracts and channel requirements.
- [Expected Improvement behavior and math](./plugins/selection/expected-improvement.md): EI formulation and selection semantics.
- [Objectives and channel refs](./plugins/objectives/README.md): the generic objective contract, channel naming rules, and the registry of available implementations. Objective meaning and readiness remain with the caller that selects a plugin.
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

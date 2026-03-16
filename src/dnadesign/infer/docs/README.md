## infer docs

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-16

### Read order

1. [Top README](../README.md): package boundary and fast route map.
2. [Getting started index](getting-started/README.md): first local command flow.
3. [Workspaces guide](../workspaces/README.md): deterministic workspace scaffold and template contract.
4. [Operations index](operations/README.md): pressure-test paths for local and scheduler workflows.
5. [Reference index](reference/README.md): stable command and runtime contracts.
6. [Source-tree map](../src/README.md): internal implementation layout under `infer/src/`.
7. [Architecture map](architecture/README.md): package boundary map and extension seams.
8. [Dev index](dev/README.md): maintainer process and journal.

### Documentation by workflow

#### Validate local command path
- [CLI quickstart](getting-started/cli-quickstart.md): run `validate`, ad-hoc `extract`, and ad-hoc `generate`.
- [Reference index](reference/README.md): command and contract lookups before automation.

#### Pressure-test agnostic model writes into USR
- [Agnostic-model pressure-test runbook](operations/pressure-test-agnostic-models.md): standalone CLI and ops-runbook paths.
- [End-to-end pressure-test demo](tutorials/demo_pressure_test_usr_ops_notify.md): reproducible infer -> usr -> ops -> notify walkthrough.
- [Workspaces guide](../workspaces/README.md): initialize workspace roots with `infer workspace init`.

#### Continue after infer-derived `X` exists
- [cluster exploratory clustering workflow](../../cluster/docs/workflows/exploratory-clustering.md): Leiden, UMAP, and exploratory analysis once one chosen `infer__...` column exists in a file or USR dataset.
- [USR dataset with infer-derived X -> OPAL active learning](../../opal/docs/workflows/usr-infer-x-active-learning.md): start the label/train/select loop only after infer has written the chosen `infer__...` column into a USR dataset and OPAL `campaign.yaml` uses `data.location.kind: usr`.

#### Run scheduler-oriented infer flows
- [Operations index](operations/README.md): run no-submit preflight, then submit.
- [SCC Evo2 GPU environment runbook (UV + infer)](operations/scc-evo2-gpu-uv-runbook.md): deterministic GPU environment bootstrap and infer capability verification.
- [Agnostic-model pressure-test runbook](operations/pressure-test-agnostic-models.md): contract-first ops workflow.

#### Extend and maintain infer internals
- [Architecture map](architecture/README.md): runtime module boundaries.
- [Source-tree map](../src/README.md): internal module locations.
- [Dev index](dev/README.md): maintainer loop and evidence logging.
- [Development journal](dev/journal.md): refactor slices and validation record.

### Cross-tool handoff routes

- [Multi-source source-of-truth assembly](../../usr/docs/operations/multi-source-source-of-truth-assembly.md): shared multi-source consolidation route before construct and infer share one downstream dataset.
- [Construct -> USR -> Infer source-of-truth runbook](../../usr/docs/operations/construct-infer-source-of-truth-runbook.md): shared construct-led consolidation route once construct owns the current handoff.
- [Promoter characterization feature matrix](../../usr/docs/operations/promoter-characterization-feature-matrix.md): cross-tool feature-matrix assembly before downstream cluster or OPAL use.
- [cluster exploratory workflow](../../cluster/docs/workflows/exploratory-clustering.md): exploratory clustering and UMAP branch once one explicit `infer__...` column is chosen as `X`.
- [USR dataset with infer-derived X -> OPAL active learning](../../opal/docs/workflows/usr-infer-x-active-learning.md): downstream OPAL workflow once infer-derived `X` already exists.

### Documentation by type

- [docs index by type](index.md)
- [getting-started/](getting-started/): first-run commands and prerequisites.
- [tutorials/](tutorials/): full end-to-end walkthroughs.
- [operations/](operations/): operational runbooks and pressure-test routes.
- [reference/](reference/): command and contract documentation.
- [architecture/](architecture/): package boundary map.
- [dev/](dev/): maintainer process and journal.

## infer docs

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-16

### Read order

1. [Top README](../README.md): package boundary and quick links.
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

#### Run a study-specific Evo2 feature bundle
- [Evo2 promoter-study feature runbook](operations/evo2-promoter-features.md): repo-aligned feature-bundle flow with explicit anchor-only and templated contexts for promoter datasets.
- [Evo2 provider reference](reference/evo2-provider.md): supported checkpoints, default block 26 selector, and storage/debug defaults.
- [Feature schema and selector contract](reference/feature-schema.md): output ids, metadata out ids, digests, and OPAL export ordering.
- [evo2_feature_bundle_smoke workspace](../workspaces/evo2_feature_bundle_smoke/README.md): packaged feature-bundle smoke path with generic anchor and templated records.

#### Pressure-test agnostic model writes into USR
- [Agnostic-model pressure-test runbook](operations/pressure-test-agnostic-models.md): standalone CLI and ops-runbook paths.
- [End-to-end pressure-test demo](tutorials/demo_pressure_test_usr_ops_notify.md): reproducible infer -> usr -> ops -> notify walkthrough.
- [Workspaces guide](../workspaces/README.md): initialize workspace roots with `infer workspace init`.

#### Continue after infer writes the feature columns you want
- If infer writes a bundle of `infer__...` columns, choose the one feature column or exported matrix you want to carry forward before moving to Cluster or OPAL.
- [cluster exploratory clustering workflow](../../cluster/docs/workflows/exploratory-clustering.md): Leiden, UMAP, and exploratory analysis once the chosen feature column or exported matrix exists.
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

### Shared dataset handoffs into infer

- [Multi-source shared dataset assembly](../../usr/docs/operations/multi-source-shared-dataset-assembly.md): shared multi-source consolidation route before construct and infer share one downstream dataset.
- [Construct -> USR -> Infer shared dataset runbook](../../usr/docs/operations/construct-infer-shared-dataset-runbook.md): shared construct-led consolidation route once construct owns the current handoff.
- [Promoter characterization feature matrix](../../usr/docs/operations/promoter-characterization-feature-matrix.md): study-specific feature-dataset assembly before downstream cluster or OPAL use.

### Documentation by type

- [Section index](index.md)
- [getting-started/](getting-started/): first-run commands and prerequisites.
- [tutorials/](tutorials/): full end-to-end walkthroughs.
- [operations/](operations/): operational runbooks and pressure-test routes.
- [reference/](reference/): command and contract documentation.
- [architecture/](architecture/): package boundary map.
- [dev/](dev/): maintainer process and journal.

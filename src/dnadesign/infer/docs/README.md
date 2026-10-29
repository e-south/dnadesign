<!--
**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-28
-->

[Infer](../README.md)

# Infer guide

Infer turns DNA sequences into model outputs such as likelihood scores, pooled feature vectors and generated sequences. You choose the input, model and requested outputs in a job configuration.

**Start with [Validate your first job](getting-started/cli-quickstart.md).** This creates a small workspace and checks its configuration and input without loading a model. Running inference also requires the model's dependencies and suitable hardware.

## Choose your next task

| I want to… | Guide |
| --- | --- |
| Use my own sequence files | [Workspaces and input formats](../workspaces/README.md) |
| Extract Evo2 likelihoods and pooled features | [Evo2 feature bundles](operations/evo2-sequence-features.md) |
| Save model outputs alongside sequence records | [SeqLedger extraction](operations/pressure-test-agnostic-models.md) |
| Understand validation, storage and resuming | [How a SeqLedger job works](tutorials/demo_pressure_test_usr_ops_notify.md) |
| Configure Evo2 on BU's Shared Computing Cluster | [SCC GPU setup](operations/scc-evo2-gpu-uv-runbook.md) |

## Connect inputs and outputs

[Construct](../../construct/) can assemble input sequences and supply coordinates for pooling over an inserted region. When several producers contribute to one dataset, follow the [shared dataset workflow](../../seqledger/docs/operations/assembly/construct-infer-shared-dataset-runbook.md).

For downstream analysis, select a feature column or export a matrix from the bundle. [Cluster](../../cluster/docs/workflows/exploratory-clustering.md) uses features for exploratory analysis. [OPAL](../../opal/docs/workflows/seqledger-infer-x-active-learning.md) uses a declared feature column to learn from measurements and select candidates. Sequence-view payloads have a separate storage layout and must be read or exported through the [feature schema interfaces](reference/feature-schema.md).

## Look up details

- [Commands](reference/command-contracts.md): modes, input formats, validation and write behavior.
- [Evo2 settings](reference/evo2-provider.md): supported feature-bundle models and layer selectors.
- [Feature schema](reference/feature-schema.md): pooling, stored values, identities and integrity checks.
- [All documentation](index.md): pages grouped by purpose.

For contributors, the [development guide](dev/README.md) covers public APIs and checks. The [architecture map](architecture/README.md) locates the implementation.

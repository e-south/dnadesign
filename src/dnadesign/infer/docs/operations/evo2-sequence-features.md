<!--
**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-28
-->

[Infer guide](../README.md) / [Operations](README.md)

# Extract Evo2 feature bundles

A feature bundle collects likelihood summaries, mean-pooled output logits and intermediate activations in one request. Use it to represent a sequence numerically, either on its own or within an explicit flanking context.

## 1. Check the example configuration

The [packaged bundle example](../../workspaces/evo2_feature_bundle_smoke/README.md) contains short toy records for two cases: an isolated sequence and a sequence placed between flanks. From an activated Infer environment in the repository root:

```bash
# Check the bundle configuration.
infer validate config \
  --config src/dnadesign/infer/workspaces/evo2_feature_bundle_smoke/config.yaml
# Resolve the toy inputs without loading weights.
infer run \
  --config src/dnadesign/infer/workspaces/evo2_feature_bundle_smoke/config.yaml \
  --dry-run
```

These validate inputs and selectors without loading weights. The example's CPU setting is for validation; it is not a promise of practical CPU inference.

## 2. Choose sequence context and pooling

For an isolated sequence, a minimal configuration is:

```yaml
# Extract pooled features from an isolated DNA sequence on a configured GPU.
model:
  id: evo2_7b
  device: cuda:0
  precision: bf16
  alphabet: dna

jobs:
  - id: sequence_features
    operation: extract
    ingest:
      source: records
      path: inputs/sequences.jsonl
      field: sequence
    feature_bundle:
      intermediate_block: 26
      context:
        kind: anchor_only
```

`anchor_only` averages token-position values across the entire input. For a flanked sequence, `template_custom` or `template_1kb` also pools over the inserted region, called the *anchor*. These jobs require the resolved sequence and `construct__context_id`, `construct__template_id`, `construct__anchor_start` and `construct__anchor_end` fields. [Construct](../../../construct/) can produce those records; Infer does not assemble the flanks.

The model receives the full sequence even when pooling only the anchor. Evo2 is causal, so a token's activation depends on the preceding DNA in the supplied orientation. A reverse-complement comparison requires a separate input with matching coordinates. See the [pooling contract](../reference/feature-schema.md#row-contract).

Infer uppercases DNA before model calls and feature-key construction. Uppercase and lowercase are different tokenizer symbols, so features computed from unnormalized inputs are not interchangeable with these outputs.

## 3. Configure the runtime and execute

Feature bundles support `evo2_7b` and `evo2_20b`. The default intermediate selector resolves to block 26 for 7B and block 23 for 20B. Changing the model also requires checking device capacity and dependencies; see the [provider reference](../reference/evo2-provider.md).

Install the `infer-evo2` runtime using the [GPU installation instructions](../../../../../docs/setup/installation.md#gpu-extra-infer-evo2). On BU's Shared Computing Cluster, use the [SCC setup procedure](scc-evo2-gpu-uv-runbook.md). Once the environment, input and configuration are ready:

```bash
# This executes the model and may download weights if they are not cached.
infer run --config config.yaml
```

The local CLI displays an output summary. For persistent dataset columns, configure [SeqLedger write-back](pressure-test-agnostic-models.md). For in-memory values, use the Python API below.

## 4. Use the returned features

This example executes a model, then flattens its bundle for downstream analysis. Run it only in a configured GPU environment:

```python
from dnadesign.infer import (
    export_evo2_sequence_opal_matrix,
    run_evo2_sequence_features,
)

sequences = ["ACGTACGT", "ACGTTCGT"]
bundle = {"context": {"kind": "anchor_only"}}
columnar = run_evo2_sequence_features(
    sequences,
    model_id="evo2_7b",
    bundle=bundle,
    device="cuda:0",
    precision="bf16",
)
# Keep row IDs in the same order as the input sequences.
payload = export_evo2_sequence_opal_matrix(
    row_ids=["sequence_1", "sequence_2"],
    columnar=columnar,
    model_id="evo2_7b",
    bundle=bundle,
)
print(len(payload["x"]))  # 2 rows, one per sequence
print(len(payload["feature_names"]))  # One name per matrix column
```

Bundles return scalar summaries, pooled vectors and request metadata. SeqLedger row outputs use `infer__<model_id>__<job_id>__<out_id>` names. Explicit sequence-view jobs use aliases and payload sidecars instead; consult the [feature schema](../reference/feature-schema.md) before reading or exporting those results.

## Troubleshooting

- Missing anchor coordinates: supply the resolved Construct metadata before using a templated context.
- Invalid pooling: pool over sequence positions, not the batch dimension; `pool.dim` must be at least `1`.
- Unsupported device: check the provider's capacity gate, including compute capability `9.0` or higher for `evo2_20b`.
- Changed request: resume checks request metadata. A selector or context change can require recomputation rather than reuse.

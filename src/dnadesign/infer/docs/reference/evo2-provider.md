## Evo2 Provider Reference

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-18

Use this page for the repo-aligned Evo2 promoter feature contract inside `infer`.

### Supported promoter lane set

- `evo2_7b`
- `evo2_20b`

`evo2_7b` is the default tracer-bullet lane. `evo2_20b` is the higher-capacity comparison lane once the 7B path is green.

### Project default intermediate selector

`infer` treats block `26` as a project default for promoter feature extraction.

- this is a repo default, not a universal scientific claim
- `evo2_7b` uses block `26` because it is a defensible default from public interpretability work
- `evo2_20b` keeps the same default until repo-local benchmarks justify a different lane

User-facing config should set:

```yaml
feature_bundle:
  intermediate_block: 26
```

The bundle resolves that to the canonical selector `block26_mlp_out`, and the adapter maps that to the provider-layer path currently used by the Evo2 surface.

### Feature groups

Default promoter bundles collect all three groups:

- `log_likelihood`
- `output_layer_mean`
- `intermediate_embedding`

Interpret these names as feature families, not persisted tensor shapes:

- `log_likelihood` is scalar by definition
- `output_layer_mean` refers to the final-layer embedding family
- `intermediate_embedding` refers to the selected internal-layer embedding family

`output_embedding` is accepted as a config/docs alias for continuity, but stored schema and new docs should prefer `output_layer_mean`.

### Pooling modes

For promoter bundles, `infer` uses:

- `seq_mean`: mean across the full resolved sequence
- `anchor_mean`: mean across the anchor span inside a templated context

Rules:

- `anchor_only` contexts emit `seq_mean` only
- templated contexts emit both `seq_mean` and `anchor_mean`
- tokenwise tensors are pooled in memory and discarded; tokenwise persistence is not part of the v1 repo-aligned contract

When writing to USR, the persisted outputs for `output_layer_mean` and
`intermediate_embedding` are the pooled summaries. The pooling mode is part of
the stored semantic id, for example `seq_mean` or `anchor_mean`.

### Context ownership

- `construct` owns anchor/template resolution and `construct__*` coordinate metadata
- `infer` reads that metadata and computes feature pooling
- `infer` does not build templates internally

Templated promoter bundles require these construct columns alongside the resolved sequence:

- `construct__context_id`
- `construct__template_id`
- `construct__anchor_start`
- `construct__anchor_end`

### Stored outputs

The generic USR persistence contract remains:

- `infer__<model_id>__<job_id>__<out_id>`

Promoter bundles emit stable out ids such as:

- `log_likelihood__total`
- `log_likelihood__mean_per_token`
- `output_layer_mean__seq_mean`
- `output_layer_mean__anchor_mean`
- `intermediate_embedding__block26_mlp_out__seq_mean`
- `intermediate_embedding__block26_mlp_out__anchor_mean`

Read these ids literally:

- `output_layer_mean__seq_mean` is the mean-pooled final-layer embedding across sequence positions
- `intermediate_embedding__block26_mlp_out__seq_mean` is the mean-pooled block-26 representation across sequence positions
- bare names such as `output_layer_mean` or `intermediate_embedding` are bundle categories, not raw persisted tensors

Structured bundle metadata is persisted as additional infer out ids such as:

- `metadata__context_id`
- `metadata__template_id`
- `metadata__intermediate_selector`
- `metadata__feature_request_digest`

### Resume and digest behavior

Feature-bundle resume does not trust column presence alone.

- each row carries `metadata__feature_request_digest`
- when the persisted digest does not match the current bundle request, `infer` recomputes that row instead of silently reusing stale outputs

### Hardware and performance posture

- `evo2_7b` is the default local and SCC smoke lane
- `evo2_20b` requires Hopper-class GPUs in the current upstream contract
- first-run latency is dominated by upstream Evo2 backend startup, not the bundle wrapper

Use [SCC Evo2 GPU environment runbook](../operations/scc-evo2-gpu-uv-runbook.md) for GPU environment setup and failure-mode handling.

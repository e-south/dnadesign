## Evo2 Promoter Feature Schema

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-18

Use this page when you need the stable bundle contract for promoter feature extraction.

### Row contract

The semantic bundle is one resolved context per row:

- one `(sequence_id, context_id, model_name)` combination
- scalar metadata beside scalar summaries and vector outputs
- vectors stay list-valued in storage; flattening happens only at export time

### Default bundle shape

```yaml
feature_bundle:
  kind: evo2_promoter_v1
  intermediate_block: 26
  collect_log_likelihood: true
  collect_output_layer_mean: true
  collect_intermediate_embedding: true
  context:
    kind: anchor_only
  pooling:
    seq_mean: true
    anchor_mean_for_templated: true
```

Supported context kinds:

- `anchor_only`
- `template_1kb`
- `template_custom`

### Default output ids

- `log_likelihood__total`
- `log_likelihood__mean_per_token`
- `output_layer_mean__seq_mean`
- `output_layer_mean__anchor_mean` for templated contexts
- `intermediate_embedding__block26_mlp_out__seq_mean`
- `intermediate_embedding__block26_mlp_out__anchor_mean` for templated contexts

### Metadata out ids

The v1 bundle persists these metadata out ids:

- `metadata__sequence_id`
- `metadata__anchor_id`
- `metadata__is_wildtype`
- `metadata__context_id`
- `metadata__context_kind`
- `metadata__template_id`
- `metadata__resolved_length`
- `metadata__anchor_start`
- `metadata__anchor_end`
- `metadata__model_name`
- `metadata__provider_name`
- `metadata__provider_version`
- `metadata__intermediate_block`
- `metadata__intermediate_selector`
- `metadata__pooling_modes`
- `metadata__feature_schema_version`
- `metadata__construct_version`
- `metadata__timestamp`
- `metadata__feature_request_digest`

### Selector contract

Users configure:

- `intermediate_block: 26`

`infer` resolves that to:

- canonical selector: `block26_mlp_out`
- current provider-layer path: `blocks.26.mlp.l3`

The canonical selector is the stable contract. Provider-layer strings are adapter internals.

### Export contract for OPAL

Use `dnadesign.infer.export_evo2_promoter_opal_matrix(...)` when OPAL needs a deterministic flattened matrix.

Export guarantees:

- stable feature-group ordering
- stable vector-dimension ordering
- one feature name per flattened scalar dimension

Example exported feature names:

- `infer.evo2.evo2_7b.anchor_only.log_likelihood.total`
- `infer.evo2.evo2_7b.anchor_only.output_layer_mean.seq_mean[0]`
- `infer.evo2.evo2_7b.template_1kb.intermediate_embedding.block26_mlp_out.anchor_mean[17]`

### Fail-fast rules

- `feature_bundle` and `outputs` are mutually exclusive on one extract job
- at least one feature group must be enabled
- `pool.dim < 1` is rejected during config parsing
- templated contexts without required `construct__*` metadata fail before model execution
- unsupported model ids for this bundle fail before adapter execution
- tokenwise persistence is intentionally unsupported in the repo-aligned v1 bundle

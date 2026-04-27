## Evo2 Promoter Feature Schema

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-18

Use this page when you need the stable bundle contract for promoter feature extraction.
The repository now supports two sibling surfaces:

- legacy promoter-context bundles driven by `context.kind` plus bundle-level pooling flags
- explicit sequence-view bundles driven by `sequence_view_inputs[]`

### Row contract

The semantic bundle is one resolved context per row:

- one `(sequence_id, context_id, model_name)` combination
- scalar metadata beside scalar summaries and vector outputs
- vectors stay list-valued in storage; flattening happens only at export time

### Default bundle shape

```yaml
feature_bundle:
  kind: evo2_promoter_v1
  intermediate_block: 26  # legacy config default; evo2_20b resolves it to block 23
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

Sequence-view bundles add an explicit view-selection surface:

```yaml
feature_bundle:
  kind: evo2_promoter_v1
  collect_log_likelihood: false
  sequence_view_inputs:
    - dataset: construct_prom_eth_cip_reference_core60
      root: src/dnadesign/usr/datasets
      view_selector:
        product_kind: analysis_core60
      pooling:
        operation: core60_mean
    - dataset: construct_prom_eth_cip_reference_contexts
      root: src/dnadesign/usr/datasets
      view_selector:
        product_kind: context1kb_reverse_complement
      pooling:
        operation: anchor_mean
        bounds_from: sequence_view
```

Sequence-view rules:

- each resolved row is one explicit semantic view id, not one implicit promoter lane
- `seq_mean`, `anchor_mean`, and `core60_mean` are row-level pooling operations
- `core60_mean` aliases `seq_mean` only when the emitted sequence is exactly 60 bp and the pooling span is the full row
- view-aware bundles persist feature aliases under `_derived/infer/feature_aliases.parquet`

### Default output ids

- `log_likelihood__total`
- `log_likelihood__mean_per_token`
- `output_layer_mean__seq_mean`
- `output_layer_mean__anchor_mean` for templated contexts
- `output_layer_mean__core60_mean` for explicit 60 bp sequence-view bundles
- `intermediate_embedding__block26_mlp_out__seq_mean` for `evo2_7b`
- `intermediate_embedding__block26_mlp_out__anchor_mean` for templated `evo2_7b` contexts
- `intermediate_embedding__block26_mlp_out__core60_mean` for explicit 60 bp `evo2_7b` sequence-view bundles
- `intermediate_embedding__block23_mlp_out__seq_mean` for `evo2_20b`
- `intermediate_embedding__block23_mlp_out__anchor_mean` for templated `evo2_20b` contexts
- `intermediate_embedding__block23_mlp_out__core60_mean` for explicit 60 bp `evo2_20b` sequence-view bundles

### Metadata out ids

The v1 bundle persists these metadata out ids:

- `metadata__sequence_id`
- `metadata__anchor_id`
- `metadata__is_wildtype`
- `metadata__context_id`
- `metadata__context_kind`
- `metadata__view_id`
- `metadata__view_name`
- `metadata__product_kind`
- `metadata__orientation`
- `metadata__template_id`
- `metadata__resolved_length`
- `metadata__anchor_start`
- `metadata__anchor_end`
- `metadata__pooling_operation`
- `metadata__pooling_start_0`
- `metadata__pooling_end_0`
- `metadata__model_name`
- `metadata__provider_name`
- `metadata__provider_version`
- `metadata__intermediate_block`
- `metadata__intermediate_selector`
- `metadata__pooling_modes`
- `metadata__forward_pass_key`
- `metadata__feature_vector_key`
- `metadata__parent_sequence_id`
- `metadata__derivation_id`
- `metadata__feature_schema_version`
- `metadata__construct_version`
- `metadata__timestamp`
- `metadata__feature_request_digest`

### Selector contract

Users configure:

- `intermediate_block: 26` as the stable config default

`infer` resolves that to:

- `evo2_7b`: canonical selector `block26_mlp_out`, provider-layer path `blocks.26.mlp.l3`
- `evo2_20b`: canonical selector `block23_mlp_out`, provider-layer path `blocks.23.mlp.l3`

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
- `infer.evo2.evo2_7b.analysis_core60.intermediate_embedding.block26_mlp_out.core60_mean[17]`
- `infer.evo2.evo2_20b.template_1kb.intermediate_embedding.block23_mlp_out.anchor_mean[17]`

### Fail-fast rules

- `feature_bundle` and `outputs` are mutually exclusive on one extract job
- at least one feature group must be enabled
- `pool.dim < 1` is rejected during config parsing
- templated contexts without required `construct__*` metadata fail before model execution
- unsupported model ids for this bundle fail before adapter execution
- tokenwise persistence is intentionally unsupported in the repo-aligned v1 bundle

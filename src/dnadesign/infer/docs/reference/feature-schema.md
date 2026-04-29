## Evo2 Sequence-View Feature Schema

**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-28

Use this page when you need the stable bundle contract for Evo2 feature extraction from explicit sequence views.
The repository supports two sibling surfaces:

- context bundles driven by `context.kind` plus bundle-level pooling flags
- explicit sequence-view bundles driven by `sequence_view_inputs[]`

The v1 schema literal is `evo2_sequence_feature_v1`. It describes the generic
Evo2 sequence-feature contract used by both row-context and explicit
sequence-view bundles; promoter studies are inputs to that contract, not a
separate Infer primitive.

### Row contract

The semantic bundle is one resolved context per row:

- one `(sequence_id, context_id, model_name)` combination
- scalar metadata beside scalar summaries and vector outputs
- vectors stay list-valued in storage; flattening happens only at export time

### Default bundle shape

```yaml
feature_bundle:
  kind: evo2_sequence_feature_v1
  intermediate_block: 26  # stable config default; evo2_20b resolves it to block 23
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
  kind: evo2_sequence_feature_v1
  collect_log_likelihood: false
  sequence_view_inputs:
    - dataset: construct_prom_eth_cip_reference_core60
      root: src/dnadesign/usr/datasets
      view_selector:
        product_kind: analysis_window
      pooling:
        operation: core60_mean
    - dataset: construct_prom_eth_cip_reference_contexts
      root: src/dnadesign/usr/datasets
      view_selector:
        product_kind: realized_context
        orientation: reverse_complement
      pooling:
        operation: anchor_mean
        bounds_from: sequence_view
```

Sequence-view rules:

- each resolved row is one explicit semantic view id, not one implicit study lane
- `construct_insert` views map to `context_kind=anchor_only`; they are the
  merged construct-ready anchor handoff rows, not derived core60 rows
- `seq_mean`, `anchor_mean`, and `core60_mean` are row-level pooling operations
- `anchor_mean` does not shorten the model input. Infer sends the full emitted sequence to the
  provider, then mean-pools token features over the explicit emitted-orientation
  `pooling_start_0:pooling_end_0` span.
- reverse-complement context rows must already contain reverse-complement sequences and
  reverse-complement-orientation pooling bounds; Infer must not apply a second `L-b, L-a`
  transform.
- `seq_mean`, `anchor_mean`, and `core60_mean` are distinct feature identities.
  An exact repeated input sequence can share one Evo2 forward pass through the
  `forward_pass_key`, but it still receives a distinct feature-vector key when
  its pooling semantics differ.
- view-aware bundles persist feature aliases under `_derived/infer/feature_aliases.parquet`
- view-aware bundles persist reusable feature vectors under `_derived/infer/feature_vectors.parquet`
- view-aware bundles persist log-likelihood scalar aliases under
  `_derived/infer/feature_scalar_aliases.parquet`
- view-aware bundles persist reusable log-likelihood scalar values under
  `_derived/infer/feature_scalars.parquet`
- `infer` consumes sequence views; it does not manufacture missing
  `analysis_window` or `realized_context` rows
- missing required product kinds are a Construct/USR completion problem; the
  completion planner reports them as `missing_products` before model execution
- Before large backfills, run `uv run infer validate sequence-view-completion
  --config <config.yaml> --format json` to classify vectors and scalars as
  reusable, stale, missing, or product-missing without loading the model.
  Use `--mode inventory` for host preflight/status loops that must stay bounded:
  it counts expected sequence-view products plus alias/payload sidecar
  inventory, catches stale alias-to-payload references, and avoids deriving
  every missing feature key for very large partial datasets. Keep the default
  exact mode for deeper batch planning when the runtime cost is acceptable.
  Sequence-view `root` values resolve relative to the config file directory.
  USR row-overlay payload columns are not counted as sequence-view feature
  coverage.
- Batch preflight can add `--max-missing-products 0 --max-stale-vectors 0
  --max-stale-scalars 0` to fail before submit when required sequence products
  are absent or existing vector/scalar sidecars are stale. Do not set
  `--max-missing-vectors 0` or `--max-missing-scalars 0` for a lane whose
  purpose is to generate missing features.
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

Use `dnadesign.infer.export_evo2_sequence_opal_matrix(...)` when OPAL needs a deterministic flattened matrix.
The exported rows are keyed by feature metadata, not by a promoter-only primitive.

Export guarantees:

- stable feature-group ordering
- stable vector-dimension ordering
- one feature name per flattened scalar dimension

Example exported feature names:

- `infer.evo2.evo2_7b.anchor_only.log_likelihood.total`
- `infer.evo2.evo2_7b.anchor_only.output_layer_mean.seq_mean[0]`
- `infer.evo2.evo2_7b.template_1kb.intermediate_embedding.block26_mlp_out.anchor_mean[17]`
- `infer.evo2.evo2_7b.analysis_window.intermediate_embedding.block26_mlp_out.core60_mean[17]`
- `infer.evo2.evo2_20b.template_1kb.intermediate_embedding.block23_mlp_out.anchor_mean[17]`

### Fail-fast rules

- `feature_bundle` and `outputs` are mutually exclusive on one extract job
- at least one feature group must be enabled
- `pool.dim < 1` is rejected during config parsing
- templated contexts without required `construct__*` metadata fail before model execution
- sequence-view bundles with missing required views, missing anchor bounds, or invalid pooling spans fail before model execution
- unsupported model ids for this bundle fail before adapter execution
- tokenwise persistence is intentionally unsupported in the repo-aligned v1 bundle

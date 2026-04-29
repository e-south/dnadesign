## Evo2 Provider Reference

**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-28

Use this page for the repo-aligned Evo2 sequence-feature contract inside `infer`.

### Supported Evo2 lanes

- `evo2_7b`
- `evo2_20b`

`evo2_7b` is the default tracer-bullet lane. `evo2_20b` is the higher-capacity comparison lane once the 7B path is green.

### Project default intermediate selector

`infer` uses a model-aware project default for sequence-feature extraction.

- this is a repo default, not a universal scientific claim
- `evo2_7b` uses block `26` because public Evo 2 interpretability work examined layer 26 and reported biologically meaningful features there; see [Interpreting Evo 2](https://www.goodfire.ai/research/interpreting-evo-2)
- `evo2_20b` uses block `23` because the upstream 20B surface exposes blocks `0..23`

User-facing config should set:

```yaml
feature_bundle:
  intermediate_block: 26  # on evo2_20b this resolves to the model-aware default block 23
```

The bundle resolves that to a model-aware canonical selector:

- `evo2_7b`: `block26_mlp_out` -> `blocks.26.mlp.l3`
- `evo2_20b`: `block23_mlp_out` -> `blocks.23.mlp.l3`

### Feature groups

Default feature bundles collect all three groups:

- `log_likelihood`
- `output_layer_mean`
- `intermediate_embedding`

Interpret these names as feature families, not persisted tensor shapes:

- `log_likelihood` is scalar by definition
- `output_layer_mean` refers to the final-layer embedding family
- `intermediate_embedding` refers to the selected internal-layer embedding family

`output_embedding` is accepted as a config/docs alias for continuity, but stored schema and new docs should prefer `output_layer_mean`.

### Pooling modes

For feature bundles, `infer` uses:

- `seq_mean`: mean across the full resolved sequence
- `anchor_mean`: mean across the anchor span inside a templated context
- `core60_mean`: explicit 60 bp analysis-core pooling for sequence-view bundles

Rules:

- `anchor_only` contexts emit `seq_mean` only
- templated contexts emit both `seq_mean` and `anchor_mean`
- sequence-view bundles emit the union of the row-level pooling operations requested by `sequence_view_inputs[]`
- tokenwise tensors are pooled in memory and discarded; tokenwise persistence is not part of the v1 repo-aligned contract

When writing to USR, the persisted outputs for `output_layer_mean` and
`intermediate_embedding` are the pooled summaries. The pooling mode is part of
the stored semantic id, for example `seq_mean` or `anchor_mean`.
For explicit 60 bp sequence views, `core60_mean` is semantically distinct metadata but aliases
the same feature-vector key as `seq_mean` when the emitted row is exactly 60 bp and the pooling
span is the full sequence.
This is a feature-alias rule, not a sequence-product rule: a natively 60 bp
`construct_insert` should stay an anchor insert in USR, while a true
`analysis_window` row means Construct derived a 60 bp analysis-only view.

### Context ownership

- `construct` owns anchor/template resolution and `construct__*` coordinate metadata
- `infer` reads that metadata and computes feature pooling
- `infer` does not build templates internally

Templated context bundles require these construct columns alongside the resolved sequence:

- `construct__context_id`
- `construct__template_id`
- `construct__anchor_start`
- `construct__anchor_end`

### Stored outputs

The generic USR persistence contract remains:

- `infer__<model_id>__<job_id>__<out_id>`

Feature bundles emit stable out ids such as:

- `log_likelihood__total`
- `log_likelihood__mean_per_token`
- `output_layer_mean__seq_mean`
- `output_layer_mean__anchor_mean`
- `intermediate_embedding__block26_mlp_out__seq_mean` for `evo2_7b`
- `intermediate_embedding__block26_mlp_out__anchor_mean` for templated `evo2_7b` contexts
- `intermediate_embedding__block23_mlp_out__seq_mean` for `evo2_20b`
- `intermediate_embedding__block23_mlp_out__anchor_mean` for templated `evo2_20b` contexts

Read these ids literally:

- `output_layer_mean__seq_mean` is the mean-pooled final-layer embedding across sequence positions
- `output_layer_mean__core60_mean` is the explicit 60 bp analysis-core mean-pooling surface for sequence-view bundles
- `intermediate_embedding__block26_mlp_out__seq_mean` is the mean-pooled 7B block-26 representation across sequence positions
- `intermediate_embedding__block23_mlp_out__seq_mean` is the mean-pooled 20B block-23 representation across sequence positions
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
- `overwrite: false` prevents recomputing already-written feature outputs on
  rerun
- reruns can still backfill metadata fields when stored values are null; treat
  that as idempotent metadata completion rather than duplicated feature work

### Hardware and performance posture

- `evo2_7b` is the default local and SCC smoke lane
- `evo2_20b` requires GPUs at compute capability `>= 9.0` in the current upstream contract; on SCC that usually means H200, but newer higher-capability lanes also qualify when memory is sufficient
- first-run latency is dominated by upstream Evo2 backend startup, not the bundle wrapper; for `evo2_20b`, expect a cold-start phase of `fetch -> hydration -> GPU residency -> first attach events`

Use [SCC Evo2 GPU environment runbook](../operations/scc-evo2-gpu-uv-runbook.md) for GPU environment setup and failure-mode handling.

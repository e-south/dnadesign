# Permuter Handoffs

**Owner:** dnadesign-maintainers
**Last verified:** 2026-07-14

Permuter handoffs are boundary artifacts. They name generated variants and the
next requested action, but they do not make Permuter the owner of USR, Infer,
Construct, Study, or Ops behavior.

## Handoff Rule

Permuter proposes variants. The receiving plane executes its own contract:

- USR records canonical sequence rows, overlays, events, and sidecar locations.
- Construct realizes named slots and writes sequence views.
- Infer materializes model features and sidecars.
- Studies bind biological meaning, promotion rules, and readiness gates.
- Ops runs only declared lanes.

When a handoff crosses a tool boundary, prefer a machine-readable manifest plus
public APIs over imports from sibling internals.

## Permuter To USR

Use `materialize_result(...)` when a caller needs a USR-shaped dataset:

```python
from dnadesign.permuter import (
    CodingDnaDmsRequest,
    default_codon_table_path,
    generate_variants,
    materialize_result,
)

result = generate_variants(
    CodingDnaDmsRequest(
        ref_name="rt_cds",
        sequence="AAA",
        codon_table=default_codon_table_path("ecoli"),
        positions=(1,),
    )
)
dataset = materialize_result(result, "outputs/permuter_rt_cds_dms")
```

The materialized rows use canonical USR `id` values and preserve the public API
variant identity in `permuter__var_id`.

## Permuter To Infer

Permuter does not execute Infer feature bundles. A Permuter-to-Infer
handoff is a non-executing request that names the already materialized dataset
and the Infer-owned feature-bundle config. The public `InferFeatureRequest`
contract can write and read this manifest without importing Infer internals.

```python
from dnadesign.permuter import (
    InferFeatureRequest,
    InferFeatureSourceDataset,
    InferSequenceViewSelector,
    write_infer_feature_request_manifest,
)

request = InferFeatureRequest(
    source_dataset=InferFeatureSourceDataset(
        usr_root="workspaces/studies/<study-id>/usr",
        dataset_id="<study-prefixed-dataset-id>",
    ),
    feature_bundle_ref="docs/studies/<study-id>/operations/contract/fixtures/infer/<bundle>.yaml",
    sequence_view_selectors=(
        InferSequenceViewSelector(view_name="dual_cassette_2000bp_seq_mean"),
    ),
    requested_outputs=("log_likelihood", "output_layer_mean", "intermediate_embedding"),
)
write_infer_feature_request_manifest(request, "outputs/permuter-infer-handoff.yaml")
```

```yaml
kind: permuter_infer_feature_request_v1
source_owner: permuter
execution_owner: infer
writeback_owner: infer
source_dataset:
  usr_root: workspaces/studies/<study-id>/usr
  dataset_id: <study-prefixed-dataset-id>
feature_bundle_ref: docs/studies/<study-id>/operations/contract/fixtures/infer/<bundle>.yaml
sequence_view_selectors:
  - view_name: dual_cassette_2000bp_seq_mean
requested_outputs:
  - log_likelihood
  - output_layer_mean
  - intermediate_embedding
```

`source_owner` names the owner of `source_dataset`, not the tool that created
the variant intent. Use `permuter` when Infer will read a Permuter-owned
variant dataset, and `construct` when a study has promoted variants into
realized Construct context rows.

`sequence_view_selectors[]` must set exactly one of `view_name` or `alias`.
Broad selectors such as `product_kind` plus `orientation` are rejected at the
Permuter handoff boundary so Infer cannot silently score the wrong Construct
view.

Infer owns validation, run, resume, stale detection, alias tables, vector/scalar
payloads, and `_derived/infer`. Run the Infer completion planner before long
jobs:

```bash
uv run infer validate sequence-view-completion --config <config.yaml> --mode inventory
```

For multi-view Construct outputs, selectors must use explicit `view_name` or a
stable alias.

## Permuter To Studies

Studies may call the public Permuter API to create candidate perturbations, then
write study-owned overlays. For RT-lnRNA, the study-owned lane promotes
RT-CDS DMS variants into Construct slot-input rows with:

- `construct_subject__id`
- `construct_subject__record_kind`
- `construct_subject__sequence_authority`
- `construct_subject__biological_sequence_fields`
- `construct_subject__lnrna_sequence`
- `construct_subject__rt_cds_sequence`
- `construct_subject__permuter_request_id`
- `construct_subject__permuter_variant_id`

The `construct_subject__permuter_variant_id` field is study overlay provenance. It is
not a replacement for the canonical USR `id` column.

## Rejected Patterns

- Permuter writing `_derived/infer` files.
- Infer importing `dnadesign.permuter.src.*`.
- USR `id` values copied from study candidate ids.
- Duplicate variant id columns such as both `permuter__var_id` and
  `permuter__variant_id`.
- Aliases for unsupported protocol or metric ids.

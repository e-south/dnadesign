## SPOP Scalar Objective `spop_v1`

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-26

`spop_v1` ranks candidates by a predicted SPOP endpoint scalar.

Use it when the configured `Y` column contains one finite scalar per candidate
for the Reader metric:

```text
reader_spop_endpoint_dose_mean_v1
```

OPAL does not parse Reader plate artifacts, choose endpoints, aggregate wells,
or recompute SPOP. Reader and the study bridge own those steps. OPAL receives a
scalar label or prediction and exposes a typed objective channel for selection.

## Input

- `data.y_expected_length: 1`
- `transforms_y.name: scalar_from_table_v1`
- one numeric SPOP column or shared label source value per labeled candidate
- no objective params

The objective accepts model predictions with shape `(n, 1)` and finite numeric
values. Regressors may predict values below zero; the objective does not clip
them. Negative predictions remain selectable scores, but diagnostics count them.

## Output Channels

- Score channel: `spop_v1/spop`
- Direction: maximize
- Uncertainty channels: none

The runtime emits these objective diagnostics:

- `metric_id`: `reader_spop_endpoint_dose_mean_v1`
- `numeric_scope`: `reader_experiment_normalized_tf_sponging`
- `score_channel`: `spop`
- `negative_prediction_count`
- `summary_stats`

## Configuration

```yaml
data:
  y_column_name: reader_spop_endpoint_dose_mean_v1
  y_expected_length: 1

transforms_y:
  name: scalar_from_table_v1
  params: {}

objectives:
  - name: spop_v1
    params: {}

selection:
  name: top_n
  params:
    top_k: 12
    score_ref: spop_v1/spop
    objective_mode: maximize
    tie_handling: competition_rank
```

## Record Flow

Reader computes the SPOP scalar and support vectors. A bridge materializes that
assay result onto study, Construct, USR, or label-source records with Reader
artifact provenance. OPAL reads the configured scalar `Y` surface and ranks
predictions through `spop_v1/spop`.

Keep the provenance with the records that carry the scalar:

- Reader metric id and numeric scope
- Reader artifact reference, record id, and content digest
- study or Construct identity bridge, when applicable
- observed-label round and batch metadata, when used by OPAL

Use `scalar_identity_v1/scalar` for generic scalar targets. Use `spop_v1/spop`
when the scalar is specifically the Reader SPOP endpoint metric and the campaign
should expose that semantic channel in ledgers, selection config, and review
artifacts.

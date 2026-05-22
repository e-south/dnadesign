# SFXI Reference Overlay Contract

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-11

The `sfxi_ref` namespace stores provenance-aware SFXI reference metrics for downstream annotation overlays. It is an additive USR overlay keyed by canonical `id`; it must not mutate base `records.parquet` rows or existing Infer overlays.

## Boundary

- Source observations come from Reader vec8 artifacts.
- Scoring is reached through Reader's public `reader.domains.logic.sfxi.setpoint_scatter.score_sfxi_setpoints` API.
- Reader delegates SFXI scoring to the source-owned `dnadesign.opal.api.sfxi` API.
- LatentDNA consumes materialized USR/view columns such as `sfxi_ref__metric_value`; it must not import Reader internals or OPAL internals.

## Required Semantics

Each overlay row represents one selected SFXI metric for one USR record.

- `sfxi_ref__reference_instance_id`: Reader/design identity, for example `pDual-10-ES10p`.
- `sfxi_ref__collection_id`: reference collection identity, for example `reader_sfxi_pdual10_latest`.
- `sfxi_ref__batch_id` and `sfxi_ref__campaign_id`: assay or campaign grouping metadata.
- `sfxi_ref__metric_id`: metric identity, for example `sfxi_v1/and/sfxi`.
- `sfxi_ref__metric_value`: numeric value intended for continuous annotation hue.
- `sfxi_ref__metric_provenance`, `sfxi_ref__source_ref`, and `sfxi_ref__score_ref`: source and scoring lineage.
- OPAL-compatible fields preserve `objective_name`, `api_version`, `state_order`, `setpoint_vector`, `denom_used`, and `denom_percentile`.

The builder joins Reader rows to USR base records by case-insensitive normalized DNA sequence, then writes USR `id` values into the overlay.

## Validation

Builders must fail before writing when any of these are true:

- duplicate normalized sequence values exist in either base records or the selected Reader score rows;
- duplicate `(reference_instance_id, metric_id)` pairs are produced;
- duplicate USR `id` values are produced;
- metric values are missing or non-finite;
- metric provenance is empty;
- Reader score rows do not map to the target USR dataset by sequence.

## Command

Dry run:

```bash
uv run python -m dnadesign.usr.scripts.build_reader_sfxi_reference_overlay --expected-count 23
```

The dry run reads the sibling Reader artifact and reports the planned row count with `"written": false`.
It also materializes the in-memory Arrow table against the packaged USR registry and reports
`"registry_validated": true` before any write is allowed.

Writing requires `--write`. The writer refuses to append if `sfxi_ref` already exists for the dataset, so replacement or aggregation remains an explicit maintenance workflow.

## LatentDNA Use

Reference annotations can use `sfxi_ref__metric_value` as a continuous marker hue:

```yaml
annotation:
  reference_set: reference_sfxi_archive
  hue_column: sfxi_ref__metric_value
  colorbar_label: SFXI
```

If annotation hue is configured, selected annotation rows must contain finite numeric values unless the plot explicitly sets `missing_policy: allow`.
Generated notebooks expose `sfxi_ref__metric_value` as the `SFXI metric` reference-hue option; missing columns are tolerated until the overlay is materialized, but selected rows fail fast when a configured static plot requires the metric.

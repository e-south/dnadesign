# SFXI Reference Overlay Contract

**Owner:** USR maintainers
**Last verified:** 2026-07-30

The `sfxi_ref` namespace stores provenance-aware SFXI reference metrics for downstream annotation overlays. It is an additive USR overlay keyed by canonical `id`; it must not mutate base `records.parquet` rows or existing Infer overlays.

## Boundary

- Reader owns and publishes neutral `four_state_vector/vector` measurements
  under `logic.four_state_vector.v1`.
- OPAL's public `dnadesign.opal.api.sfxi` surface owns SFXI objective math and
  conversion to overlay records.
- A study owns the selection of Reader records, setpoint, and target USR
  dataset. The stress-study recipe is documented under its `decision/opal/`
  surface.
- USR's public `Dataset` API owns generic overlay publication.
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

The study recipe joins Reader rows to USR base records by case-insensitive
normalized DNA sequence, then places USR `id` values into the proposed overlay.

## Validation

Study recipes must fail before asking USR to write when any of these are true:

- duplicate normalized sequence values exist in either base records or the selected Reader score rows;
- duplicate `(reference_instance_id, metric_id)` pairs are produced;
- duplicate USR `id` values are produced;
- metric values are missing or non-finite;
- metric provenance is empty;
- Reader score rows do not map to the target USR dataset by sequence.

## Command

Dry run:

```bash
uv run python -m dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.sfxi_reference_overlay \
  --reader-root ../reader \
  --expected-count 23
```

The default route is read-only. It verifies each selected latest Reader record
against the study's portable envelope, including
`logic.four_state_vector.v1`, record-schema
version, configuration digest, content digest, producer, and selected design
identities. It then applies OPAL's fixed SFXI v1 scoring contract, resolves the
target USR identities, and reports `"written": false`. Persisted source
provenance is the portable selection identity and digest, never a machine path.

Writing requires `--write`. The study recipe then passes the table to the
atomic `Dataset.create_overlay` operation. Its create-once check and write run
under the dataset lock, so concurrent publishers cannot append a second part.
Replacement or aggregation remains an explicit maintenance workflow.

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

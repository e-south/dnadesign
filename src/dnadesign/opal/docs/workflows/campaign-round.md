---
id: opal-campaign-round
title: OPAL campaign round
owner: dnadesign-maintainers
status: active
last_verified: 2026-10-25
---

# Campaign round

**Type:** workflow
**Owner-boundary:** opal
**Entry artifact:** one valid `opal.campaign.v3` config and its declared records
**Exit artifact:** verified round outputs and ledger rows

OPAL learns from measured labels, predicts outcomes for the remaining candidates,
and selects a batch to test. Add the next measurements to repeat the round.

## Prepare inputs and configuration

Copy the [complete random-forest configuration](../../campaigns/demo_rf_topn/configs/campaign.yaml)
into your campaign's `configs/campaign.yaml`. Adapt these fields:

- `ownership`: for a study, use `owner_scope: study_campaign`, your `study_id`
  and `dataset_id`, and `portable: false`.
- `campaign`: set its name, slug and working directory.
- `data`: point to the candidate Parquet file and name its feature and label
  columns. Candidate records need unique string IDs, sequences, `bio_type`,
  `alphabet`, and finite fixed-length Arrow feature vectors. See
  [candidate records](../reference/data-contracts.md#candidate-records).
- `transforms_y.params`: name the sequence and measurement columns in the label
  CSV or workbook. The example uses `sequence` and `y`; `data.y_column_name`
  names the stored label column, not the incoming spreadsheet column.
- `selection_views`: choose the objective, score direction and batch size.
  Copy the referenced `plots.yaml` too, or omit `plot_config` if plots are not needed.

The example fits a scalar response and requests the top five candidates,
preserving ties at the selection cutoff.
For a USR dataset with existing features, follow the
[USR workflow](usr-infer-x-active-learning.md). Plugin replacements must agree
on target shape, score channels and uncertainty channels.

`expected_improvement` requires both `score_ref` and `uncertainty_ref`. The
referenced uncertainty must be a standard deviation produced by the configured
objective. OPAL stops if the channel is absent or incompatible.

## Run one round

From the campaign directory:

```bash
# Check the config and input records without writing campaign state.
uv run opal validate -c configs/campaign.yaml
# Create the campaign state and output directories.
uv run opal init -c configs/campaign.yaml
# Record the labels observed for round zero.
uv run opal ingest-y -c configs/campaign.yaml --round 0 --csv <labels-file> --unknown-sequences error --apply
# Fit, score, and select for the same label cutoff.
uv run opal run -c configs/campaign.yaml --round 0
# Check the saved view against the prediction ledger.
uv run opal verify-outputs -c configs/campaign.yaml --view primary --round latest
# Inspect and export the selected candidates for the next experiment.
uv run opal selection-batch show -c configs/campaign.yaml --round latest
# Save candidate identities and ranks as CSV.
uv run opal selection-batch export -c configs/campaign.yaml --round latest --out selected.csv --format csv
```

`validate` is read-only. `init` creates campaign state. `ingest-y` records the
labels visible to a round. `run` fits, scores, and selects. `verify-outputs`
checks the persisted selection against the ledger. Replace `primary` if you
renamed the selection view. If a round contains multiple runs, pass `--run-id`
to inspect or export the intended run. The exported batch is a candidate proposal
for experimental review.

Use `opal guide -c configs/campaign.yaml --format markdown` for commands and
paths resolved from a particular config. Use `opal explain` before a run when
you need a read-only account of its label cutoff and preconditions.

## Add the next measurements

Measure the selected candidates, then ingest that label file with `--round 1`
and run round 1. Keep target definitions, thresholds and calibration with the
campaign so each score can be interpreted against the intended experiment.

## Continue

- [Configuration](../reference/configuration.md)
- [Objective plugins](../plugins/objectives/README.md)
- [Model plugins](../plugins/models/README.md)
- [Selection plugins](../plugins/selection/README.md)
- [CLI reference](../reference/cli.md)

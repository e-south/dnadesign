---
id: opal-campaign-round
title: OPAL campaign round
owner: dnadesign-maintainers
status: active
last_verified: 2026-10-28
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
For a SeqLedger dataset with existing features, follow the
[SeqLedger workflow](seqledger-infer-x-active-learning.md). Plugin replacements must agree
on target shape, score channels and uncertainty channels.

`expected_improvement` requires both `score_ref` and `uncertainty_ref`. The
referenced uncertainty must be a standard deviation produced by the configured
objective. OPAL stops if the channel is absent or incompatible.

## Run one round

With the [OPAL environment activated](../index.md#try-a-complete-selection-round),
run these commands from your campaign directory. The campaign does not need its
own Python project or SeqLedger dataset:

```bash
# Check the config and input records without writing campaign state.
opal validate -c configs/campaign.yaml
# Create the campaign state and output directories.
opal init -c configs/campaign.yaml
# Record the labels observed for round zero.
opal ingest-y -c configs/campaign.yaml --round 0 --csv <labels-file> --unknown-sequences error --apply
# Fit, score, and select for the same label cutoff.
opal run -c configs/campaign.yaml --round 0
# Check the saved view against the prediction ledger.
opal verify-outputs -c configs/campaign.yaml --view primary --round latest
# Inspect and export the selected candidates for the next experiment.
opal selection-batch show -c configs/campaign.yaml --round latest
# Save DNA sequences, identities, ranks and scores for this selection view.
opal selection-set export -c configs/campaign.yaml --view primary --round latest --out selected.csv --format csv
```

`validate` is read-only. `init` creates campaign state. `ingest-y` records the
labels visible to a round. `run` fits, scores, and selects. `verify-outputs`
checks the persisted selection against the ledger. Replace `primary` if you
renamed the selection view. If a round contains multiple runs, replace `--round` with `--run-id`
to inspect or export the intended run. The exported selection is a candidate proposal
for experimental review.

Use `opal guide -c configs/campaign.yaml --format markdown` for commands and
paths resolved from a particular config. Use `opal explain` before a run when
you need a read-only account of its label cutoff and preconditions.

## Add the next measurements

Measure the selected candidates, then ingest that label file with `--round 1`
and run round 1. Keep target definitions, thresholds and calibration with the
campaign so each score can be interpreted against the intended experiment.

## Inspect or retry a run

A round identifies the label cutoff; each execution receives its own run ID.
If an interrupted run left round artifacts, inspect `opal status` and
`opal runs list` before rerunning that round with `--resume`. This creates a
new run and preserves completed runs. When several runs share a round, replace
`--round` with the intended `--run-id` in verification, inspection and export
commands.

The default batch removes duplicate candidate IDs. Set
`selection_batch.deduplicate_by: sequence` when distinct IDs may contain the
same DNA and the experiment requires unique sequences.

## Continue

- [Configuration](../reference/configuration.md)
- [Objective plugins](../plugins/objectives/README.md)
- [Model plugins](../plugins/models/README.md)
- [Selection plugins](../plugins/selection/README.md)
- [CLI reference](../reference/cli.md)

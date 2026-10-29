<!--
**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-28
-->

# Run a campaign round

Use OPAL to choose the next candidates to measure from a pool of sequences with
numerical features. Measurements from tested candidates supply the training
labels. A predictive model estimates outcomes for the remaining candidates;
an objective scores those predictions, and a selection rule chooses a batch.

This guide uses the random-forest example with one measured response per
candidate. Its objective uses the predicted response directly as the score,
and its `top_n` rule prefers larger scores. The example requests five candidates;
retaining ties at the cutoff can produce a larger selection.

## Prepare inputs and configuration

Use the [OPAL installation](../index.md#try-a-complete-selection-round) and
create a campaign directory outside the tool checkout:

```text
my-campaign/
├── configs/
│   └── campaign.yaml
├── records.parquet
└── labels-round-0.csv
```

The candidate table contains the whole pool, including candidates without
measurements. Each row needs an ID, sequence, sequence type and a numeric feature
vector of the same length. The [candidate-table example](../reference/data-contracts.md#prepare-a-candidate-table)
shows how to create the required Arrow columns from a CSV and a feature matrix.
Use the same feature definition for every candidate.

Copy the [random-forest configuration](../../campaigns/demo_rf_topn/configs/campaign.yaml)
to `configs/campaign.yaml`, then set:

- **Campaign identity:** choose a name and slug. Keep `campaign.workdir: "."`
  for the layout above. In `ownership`, use `owner_scope: study_campaign`, stable
  `study_id` and `dataset_id` names for your project and candidate pool, and
  `portable: false`.
- **Candidate data:** set the local path to `records.parquet` and
  `data.x_column_name` to your feature column. With a config inside `configs/`,
  both this path and the working directory resolve from the campaign directory.
- **Measurements:** the example reads `sequence` and `y` from your label file.
  Change `transforms_y.params` if your input column names differ.
  `data.y_column_name` names the stored label column.
- **Selection:** keep the `primary` selection view for this walkthrough. Its
  objective, score direction and requested batch size live in `selection_views`.
- **Plots:** copy the example's `plots.yaml` into `configs/`, or remove
  `plot_config`. Plot paths resolve from the config directory.

Include only measured candidates in `labels-round-0.csv`. For example:

```csv
sequence,y
ACGTACGT,0.42
TGCATGCA,0.87
```

Replace these illustrative rows with your sequences and measured values. Each
sequence must match a candidate. If multiple candidate IDs share the same DNA,
include an `id` column and set `transforms_y.params.id_column: id` so measurements
are assigned unambiguously.

For candidates already stored in SeqLedger, use the
[SeqLedger integration](seqledger-infer-x-active-learning.md) instead of the local
file setup. The selection loop below is the same.

## Run one round

Run these commands from `my-campaign` with the OPAL environment activated.
The abridged outputs below illustrate a synthetic run with 40 candidates and
eight initial measurements; counts depend on your inputs.

### 1. Check and initialize

Validation checks your configuration and candidate records without writing
campaign state. Initialization creates the state and output directories.

```bash
# Check configuration and feature types.
opal validate -c configs/campaign.yaml
# Create campaign state and outputs.
opal init -c configs/campaign.yaml
```

Validation reports the number and width of feature vectors:

```text
X contract
  Rows           : 40
  X dim          : 2
...
OK: validation passed.
```

### 2. Import the first measurements

Round 0 identifies the first batch of observations available for training.
The `--apply` flag records them; unknown sequences are rejected.

```bash
# Record the measured outcomes for this round.
opal ingest-y -c configs/campaign.yaml \
  --round 0 --csv labels-round-0.csv \
  --unknown-sequences error --apply
```

The import previews ID matching before recording the observations. Check that
the committed count matches the measurements you supplied:

```text
[Committed] ingest-y
  round          : 0
  labels appended: 8
  y column updated: Y
  ledger labels appended: yes
```

### 3. Fit, select and check the result

The `primary` view names the objective and selection settings in this example.
Verification compares its saved selections with the recorded predictions.

```bash
# Fit on measurements; rank unmeasured candidates.
opal run -c configs/campaign.yaml --round 0
# Check selections against predictions.
opal verify-outputs -c configs/campaign.yaml \
  --view primary --round 0
# Inspect the next experimental batch.
opal selection-batch show -c configs/campaign.yaml --round 0
```

The run summary distinguishes the training set from the remaining candidate
pool. Verification should report zero mismatches:

```text
trained_on: 8 | scored: 32
selection batch: 5
...
- compared: 5  mismatches: 0  max_abs_diff: 0.0
```

### 4. Export candidates for the next experiment

Export the selected DNA sequences, IDs, ranks and scores to a round-specific
file. These are proposed candidates; the scores describe model predictions,
not measurements of the selected sequences.

```bash
# Export selected IDs, DNA and scores.
opal selection-set export -c configs/campaign.yaml \
  --view primary --round 0 --out selected-round-0.csv --format csv
```

```text
wrote 5 selection rows to selected-round-0.csv (csv)
```

The CSV includes `id`, `sequence`, ranks, `score`, `selection_score`, `run_id` and
`as_of_round`. Here, both score columns contain the predicted response because
the objective and selector rank it directly. The first two rows of this example,
showing a subset of columns, are:

```csv
id,sequence,selection_rank,score
p36,ACGAAAACGTACGTACGTACGT,1,0.779296875
p37,CCGAAAACGTACGTACGTACGT,2,0.763671875
```

A selection set belongs to one view. With multiple views, use
`opal selection-batch export` to export their combined, deduplicated proposal;
see the [export commands](../reference/cli.md).

## Add the next measurements

Measure the selected candidates and put the new observations in
`labels-round-1.csv`. Do not resubmit the round-0 file. The copied configuration
trains cumulatively using measurements through the requested round; when a
candidate has been measured in several rounds, it uses the latest value for
training. Already-labeled candidates are excluded from selection, while a prior
suggestion can appear again until it is measured.

```bash
# Record the measured outcomes for this round.
opal ingest-y -c configs/campaign.yaml \
  --round 1 --csv labels-round-1.csv \
  --unknown-sequences error --apply
# Refit using observations available through round 1.
opal run -c configs/campaign.yaml --round 1
# Check the new selections before using them.
opal verify-outputs -c configs/campaign.yaml \
  --view primary --round 1
# Export selected IDs, DNA and scores.
opal selection-set export -c configs/campaign.yaml \
  --view primary --round 1 --out selected-round-1.csv --format csv
```

Keep the measurement definition and objective consistent across rounds.
[Training policies](../reference/configuration.md) explain alternatives for
combining observations and choosing eligible candidates.

## Inspect or retry a run

A round specifies which observations are available; a run is one execution of
fitting and selection. Inspect `opal status` and `opal runs list` before retrying
an interrupted round with `--resume`. A retry creates a new run ID and
preserves completed runs. When several runs share a round, replace `--round`
with `--run-id` in verification, inspection and export commands.

The default batch removes duplicate IDs. Set
`selection_batch.deduplicate_by: sequence` when the experiment requires unique
DNA sequences across different IDs.

Use `opal guide -c configs/campaign.yaml --format markdown` for commands resolved
from your configuration, or `opal explain` for a read-only check of a planned run.

## Choose another prediction or selection strategy

[Models](../plugins/models/README.md) predict the measured outcome;
[objectives](../plugins/objectives/README.md) score those predictions against your
goal. The `top_n` selector ranks scores directly. The
[pool-relative weighted acquisition rule](../plugins/selection/expected-improvement.md)
named `expected_improvement` also uses uncertainty and a reference derived from
the predicted pool. Its objective must supply both the requested score and
standard-deviation channels. See the [selection reference](../plugins/selection/README.md)
for those requirements before changing strategies.

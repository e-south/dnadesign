<!--
**Type:** workflow
**Plane:** downstream-tool
**Owner-boundary:** opal
**Entry artifact:** SeqLedger dataset plus one explicit infer-derived X column
**Exit artifact:** initialized OPAL campaign state plus round outputs and selection ledgers
**Registry-id:** opal.downstream.seqledger-infer-x-active-learning
**Summary:** Train on measured outcomes and select candidates using features stored in SeqLedger.
**Execution-kind:** round-loop
**Status-kind:** opal-campaign-state

**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-28
-->

# Use SeqLedger features in an OPAL campaign

Use this workflow when your candidate sequences and features are already in
SeqLedger. OPAL reads that table, learns from measurements you supply, and
selects the next candidates to test. The learning loop is the same as the
[local-file workflow](campaign-round.md); SeqLedger provides a shared dataset
and a separate file for observed labels.

## Before you start

- Activate the [OPAL environment](../index.md#try-a-complete-selection-round).
- Complete Infer write-back to the chosen SeqLedger dataset.
- Choose one finite, fixed-length feature column for every candidate. If Infer
  produced several coordinated feature columns, first assemble them into the
  vector column you intend to use as `X`.
- Prepare measurements separately; OPAL imports them as observed labels.

The [feature-matrix workflow](../../../seqledger/docs/operations/promoter/characterization-feature-matrix.md)
covers upstream preparation. Select the candidate rows before starting the
campaign, retaining stable IDs and the feature-to-sequence correspondence.
See the [candidate-record contract](../reference/data-contracts.md#candidate-records)
for the required column types.

## Configure the campaign

Create a campaign directory outside the SeqLedger dataset. Put the following
YAML blocks together in `configs/campaign.yaml`. Replace the example paths and
identifiers with your own. Here, `features` stands for your chosen feature-column
name; OPAL does not create or infer that column.

```yaml
# Read candidates and features from the existing SeqLedger dataset.
schema_version: opal.campaign.v3
campaign:
  name: Sequence selection
  slug: sequence_selection
  workdir: /abs/path/to/opal_campaign
ownership:
  owner_scope: study_campaign
  study_id: example_study
  dataset_id: example_candidates
  portable: false
data:
  location:
    kind: usr
    path: /abs/path/to/seqledger_root
    dataset: example_candidates
  x_column_name: features
  y_column_name: measured_activity
  y_expected_length: 1
```

The configuration uses `data.location.kind: usr` to read a SeqLedger dataset.
Store observations in a dataset-local sidecar and keep predictions and
selections in the campaign's output ledgers:

```yaml
# Keep observed measurements separate from prediction outputs.
labels:
  source:
    kind: usr_sidecar
    dataset: example_candidates
    path: _opal/observed_labels.parquet
  y_space: scalar_v1
writeback:
  prediction_records: ledger_only
```

This keeps measured values separate from predictions. Each selection view uses
the same observation snapshot. Sidecar writes use a local path lock; concurrent
ingest from multiple hosts requires external coordination.

The example fits a random forest and ranks predicted activity directly. It
requests twelve candidates, retaining any ties at the cutoff:

```yaml
# Predict scalar activity, then select the highest predicted values.
transforms_x:
  name: identity
  params: {}
transforms_y:
  name: scalar_from_table_v1
  params:
    sequence_column: sequence
    y_column: measured_activity
model:
  name: random_forest
  params:
    n_estimators: 100
    random_state: 7
selection_views:
  - id: primary
    objective:
      name: scalar_identity_v1
      params: {}
    selection:
      name: top_n
      params:
        top_k: 12
        score_ref: scalar
        objective_mode: maximize
        tie_handling: competition_rank
selection_batch:
  deduplicate_by: id
```

Keep one campaign when the candidate pool, features, measurements and learning
history are shared. Add named selection views for different objectives on the
same predictions. Use a separate campaign when the learning inputs differ.

## Run and inspect a round

Set the campaign path for the commands below:

```bash
# Keep campaign state outside the shared dataset.
export OPAL_WORKDIR=/abs/path/to/opal_campaign
# Check the configured feature column and dataset.
opal validate -c "$OPAL_WORKDIR/configs/campaign.yaml"
# Initialize campaign state and outputs.
opal init -c "$OPAL_WORKDIR/configs/campaign.yaml"
```

Prepare `labels-round-0.csv` with `sequence` and numeric `measured_activity`
columns. Include only measured candidates whose sequences match the candidate
table. If distinct IDs share the same DNA, include `id` and configure
`transforms_y.params.id_column: id`. To preview an import without writing,
replace `--apply` with `--json`.

```bash
# Append observations to the dataset-local label file.
opal ingest-y -c "$OPAL_WORKDIR/configs/campaign.yaml" \
  --round 0 --csv "$OPAL_WORKDIR/labels-round-0.csv" \
  --unknown-sequences error --apply
# Fit, predict activity and select candidates.
opal run -c "$OPAL_WORKDIR/configs/campaign.yaml" --round 0
# Verify that selections match the recorded predictions.
opal verify-outputs -c "$OPAL_WORKDIR/configs/campaign.yaml" \
  --view primary --round 0
# Export selected IDs and DNA for experimental review.
opal selection-set export -c "$OPAL_WORKDIR/configs/campaign.yaml" \
  --view primary --round 0 \
  --out "$OPAL_WORKDIR/selected-round-0.csv" --format csv
```

The [campaign walkthrough](campaign-round.md#run-one-round) shows example
terminal output for these commands. Validation must pass before running a round. Verification should report zero
mismatches; the export contains the selected candidate IDs, DNA and scores.
Use `opal status` and `opal runs list` with the same config to inspect progress.
The candidate table remains the input; predictions and selections are saved
under the campaign's `outputs/` directory.

## Continue with new measurements

Use the next round number for newly observed results. Re-importing an existing
candidate and round fails by default; `--if-exists replace` is for an intentional
correction. Follow the [next-round steps](campaign-round.md#add-the-next-measurements)
for ingestion, fitting, verification and export, or the
[retry guidance](campaign-round.md#inspect-or-retry-a-run) after an interruption.

Changing the predictor or selection rule does not change the dataset handoff.
The [strategy guide](campaign-round.md#choose-another-prediction-or-selection-strategy)
explains direct score ranking and uncertainty-aware selection.

## Related documentation

- [OPAL configuration](../reference/configuration.md): labels, training and selection settings.
- [Command reference](../reference/cli.md): inspection and export options.
- [Exploratory clustering](../../../cluster/docs/workflows/exploratory-clustering.md): inspect the feature matrix before selection.
- [DNADesign workflows](../../../../../docs/README.md): other cross-tool tasks.

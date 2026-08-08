## USR Dataset With Infer-Derived X -> OPAL Active Learning

**Type:** workflow
**Plane:** downstream-tool
**Owner-boundary:** opal
**Entry artifact:** USR dataset plus one explicit infer-derived X column
**Exit artifact:** initialized OPAL campaign state plus round outputs and selection ledgers
**Registry-id:** opal.downstream.usr-infer-x-active-learning
**Summary:** Start the label, train, and select loop once one OPAL-ready candidate table with an explicit infer-derived X column exists.
**Execution-kind:** round-loop
**Status-kind:** opal-campaign-state

**Owner:** dnadesign-maintainers
**Last verified:** 2026-08-08

Use this workflow when a USR candidate table already contains one or more
infer-derived feature columns and OPAL should own the label/train/select loop.

This workflow applies after Infer write-back completes.

For Evo2 sequence-feature bundles, `infer` may write several coordinated
`infer__...` columns per job. In that case, materialize one OPAL-ready
candidate table with the chosen vector-valued `X` column before entering the
OPAL round loop.

Use one campaign when the candidate universe, X, Y, labels, transforms, model,
and round history are shared. Declare different setpoints or target masks as
named `selection_views`; OPAL fits and predicts once, then scores each view.
Create another campaign only when the learning lifecycle differs.

The candidate universe is a contract, not "whatever rows happened to be in the
upstream representation view." A study can use a dense generated subset from a
larger LatentDNA review view as long as the generated `records.parquet` keeps
stable IDs, preserves the view row order as an ordered subset, and carries the
chosen fixed-length `X` column for every row.

Set `labels.source.kind: usr_sidecar` to keep observed assay truth at a typed
path such as `_opal/observed_labels.parquet` under the USR dataset. `ingest-y`
writes the observation once; every selection view consumes the same label
snapshot. Set `writeback.prediction_records: ledger_only` so predictions,
scores, selections, and batch membership remain campaign artifacts rather than
candidate-table columns. Campaign-history labels remain available for portable
local demos.

Sidecar writes use a local path lock. Do not run concurrent multi-host ingest
against that file without an external transaction or lease.
If a campaign must be fully transient, copy the candidate table into a
campaign-local `records.parquet` and point the campaign at
`data.location.kind: local`.

### Boundary decisions

- upstream source assembly, optional construct expansion, and infer write-back remain outside OPAL
- OPAL consumes one explicit `X` column; it does not decide which infer job or model lane produced that column
- `data.location.kind: usr` is the contract for reading the USR dataset directly
- no hidden orchestration exists between `infer` and `opal`; the handoff is a deliberate choice of dataset plus `x_column_name`
- campaign-history labels and round state stay under OPAL namespaces; USR
  campaigns keep primary labels in the sidecar and derived state in ledgers
- explicit shared-record prediction writeback requires an operator-visible
  records-path lock; the shared-label contract uses `ledger_only` and a
  sidecar path lock by default
- do not duplicate a USR dataset or campaign for target masks alone
- prune campaign artifacts by campaign and round; do not treat shared USR
  records as disposable OPAL output

### Preconditions

- one USR dataset already exists at a known root
- that dataset already has the chosen infer-derived `X` column such as `infer__evo2_7b__anchor_only_7b_features__intermediate_embedding__block26_mlp_out__seq_mean`
- or the Evo2 sequence bundle has already been materialized into one
  deterministic OPAL-ready `X` column outside OPAL
- labels will be ingested incrementally through OPAL rounds rather than attached silently during infer

For the upstream shared-dataset and infer matrix assembly, use:

- [Promoter characterization feature matrix](../../../usr/docs/operations/promoter/characterization-feature-matrix.md)

### Ordered procedure

### 1) Choose the dataset and the explicit `X` column

```bash
export USR_ROOT=/abs/path/to/usr_root # Reuse the same explicit USR root used for infer write-back.
export DATASET_ID="promoter_feature_matrix_demo" # Choose the infer-annotated dataset that OPAL should consume.
export X_COLUMN="infer__evo2_7b__anchor_only_7b_features__intermediate_embedding__block26_mlp_out__seq_mean" # Choose one concrete infer-derived feature column for this campaign.
export OPAL_WORKDIR=/abs/path/to/opal_campaign # Keep OPAL campaign state and ledgers outside the USR dataset root.
```

### 2) Point the OPAL campaign at the USR dataset

Declare study ownership and the USR data contract in `campaign.yaml`:

```yaml
ownership:
  owner_scope: study_campaign # Bind execution to a routed study.
  study_id: promoter_feature_matrix_study # Identify the study source of truth.
  dataset_id: promoter_feature_matrix_demo # Identify the USR candidate dataset.
  portable: false # Prevent discovery as a portable OPAL demo.

data: # Point OPAL at the infer-annotated USR dataset.
  location: { kind: usr, path: /abs/path/to/usr_root, dataset: promoter_feature_matrix_demo } # Resolve the USR root and dataset explicitly.
  x_column_name: "infer__evo2_7b__anchor_only_7b_features__intermediate_embedding__block26_mlp_out__seq_mean" # Choose one infer-derived feature column as X.
  y_column_name: "measured_activity" # Name the observed label column for this campaign.
  y_expected_length: 1 # Keep the baseline workflow on scalar labels.
```

Declare the label source and prediction writeback policy explicitly:

```yaml
labels: # Use one observed-label sidecar for this candidate universe.
  source: { kind: usr_sidecar, dataset: promoter_feature_matrix_demo, path: _opal/observed_labels.parquet } # Keep labels dataset-local.
  y_space: scalar_v1 # Name the assay label space used by the sidecar rows.
writeback: # Keep prediction outputs out of the shared candidate table.
  prediction_records: ledger_only # Store predictions/scores/selections in campaign ledgers.
```

The following RF and `top_n` blocks provide a deterministic baseline:

```yaml
transforms_x: { name: identity, params: {} } # Pass the chosen infer-derived X column through unchanged.
transforms_y: { name: scalar_from_table_v1, params: {} } # Parse scalar labels from the observed table.

model: # Use a cheap deterministic surrogate for the first tracer bullet.
  name: random_forest # Select the RF baseline model plugin.
  params: { n_estimators: 100, random_state: 7 } # Keep the baseline model deterministic.

selection_views: # Declare target-specific scoring and selection views.
  - id: primary # Give the only view a stable public identifier.
    objective: {name: scalar_identity_v1, params: {}} # Expose the scalar label as a score.
    selection: # Configure this view's selector.
      name: top_n # Use deterministic greedy ranking.
      params: # Bind top-N to the objective channel.
        top_k: 12 # Request twelve candidates.
        score_ref: scalar # Use the local scalar score channel.
        objective_mode: maximize # Prefer larger scalar values.
        tie_handling: competition_rank # Preserve score ties.

selection_batch: # Build the logical union of all view selections.
  deduplicate_by: id # Keep one logical row per candidate ID.
```

### 3) Validate and initialize the campaign

```bash
uv run opal validate -c "$OPAL_WORKDIR/configs/campaign.yaml" # Validate the USR-backed campaign config and plugin wiring.
uv run opal init -c "$OPAL_WORKDIR/configs/campaign.yaml" # Initialize campaign state and output ledgers.
```

### 4) Ingest the first label batch

```bash
# Ingest the first observed label batch into the USR-backed campaign.
uv run opal ingest-y \
  -c "$OPAL_WORKDIR/configs/campaign.yaml" \
  --round 0 \
  --csv "$OPAL_WORKDIR/inputs/r0_labels.xlsx" \
  --unknown-sequences drop \
  --if-exists replace \
  --apply
```

### 5) Run the first round against the infer-derived `X` column

```bash
uv run opal run -c "$OPAL_WORKDIR/configs/campaign.yaml" --round 0 # Train, score, and select against the chosen infer-derived X column.
```

### 6) Verify artifacts and round state

```bash
uv run opal verify-outputs -c "$OPAL_WORKDIR/configs/campaign.yaml" --view primary --round latest # Verify ledgers and round outputs.
uv run opal status -c "$OPAL_WORKDIR/configs/campaign.yaml" # Inspect current round and selection state.
uv run opal runs list -c "$OPAL_WORKDIR/configs/campaign.yaml" # Review recorded runs for this campaign.
uv run opal ctx audit -c "$OPAL_WORKDIR/configs/campaign.yaml" --round latest # Audit the round contract payload.
```

### 7) Iterate as labels accumulate

```bash
# Ingest the next observed label batch before resuming the round loop.
uv run opal ingest-y \
  -c "$OPAL_WORKDIR/configs/campaign.yaml" \
  --round 1 \
  --csv "$OPAL_WORKDIR/inputs/r1_labels.xlsx" \
  --unknown-sequences drop \
  --if-exists replace \
  --apply
uv run opal run -c "$OPAL_WORKDIR/configs/campaign.yaml" --round 1 --resume # Refit and reselection with labels visible through round 1.
```

### Model and selection variations

- use a random forest when a deterministic ensemble is sufficient;
- use a Gaussian process when the configured objective and selector need
  predictive uncertainty;
- use `top_n` for direct score ranking; or
- use `expected_improvement` only when the objective emits the referenced
  score and standard-deviation channels.

The Infer-to-OPAL handoff does not change across those choices. See the
[campaign round](campaign-round.md) for the shared lifecycle and the plugin
pages for exact channel contracts.

## Verification checklist

- `campaign.yaml` uses `data.location.kind: usr`
- `campaign.yaml` declares one explicit `x_column_name`
- `opal validate` passes before any round runs
- `opal run` succeeds with labels visible through the intended `--round` cutoff
- `verify-outputs` reports zero mismatches for the latest round

## Related docs

- Root docs router: [../../../../../docs/README.md](../../../../../docs/README.md)
- Upstream shared-dataset and infer matrix assembly: [../../../usr/docs/operations/promoter/characterization-feature-matrix.md](../../../usr/docs/operations/promoter/characterization-feature-matrix.md)
- OPAL docs index: [../index.md](../index.md)
- OPAL configuration contract: [../reference/configuration.md](../reference/configuration.md)
- OPAL CLI reference: [../reference/cli.md](../reference/cli.md)
- Cluster exploratory branch: [../../../cluster/docs/workflows/exploratory-clustering.md](../../../cluster/docs/workflows/exploratory-clustering.md)

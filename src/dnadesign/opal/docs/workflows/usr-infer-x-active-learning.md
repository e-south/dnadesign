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
**Last verified:** 2026-05-17

Use this workflow when a USR candidate table already contains one or more
infer-derived feature columns and OPAL should own the label/train/select loop.

This workflow starts after infer write-back is already complete.

For Evo2 sequence-feature bundles, `infer` may write several coordinated
`infer__...` columns per job. In that case, materialize one OPAL-ready
candidate table with the chosen vector-valued `X` column before entering the
OPAL round loop.

One OPAL-ready USR candidate table can serve multiple OPAL campaigns when the
candidate universe and chosen `X` are shared. Keep objective setpoints, model
state, scoring outputs, selected batches, notebooks, plots, and ledgers
campaign-scoped. For multi-objective studies where every campaign should train
on every newly observed assay label, keep observed labels in a shared
study-level label source and have each campaign derive its own objective from
that same label pool.

The candidate universe is a contract, not "whatever rows happened to be in the
upstream representation view." A study can use a dense generated subset from a
larger LatentDNA review view as long as the generated `records.parquet` keeps
stable IDs, preserves the view row order as an ordered subset, and carries the
chosen fixed-length `X` column for every row.

Current OPAL run/explain planning and `opal ingest-y` support that shared
label-source contract with `labels.source.kind: usr_sidecar`, for example
`_opal/observed_labels.parquet` under the shared USR candidate dataset. The
runtime reads training labels from that sidecar when configured, and
`ingest-y` writes observed labels there once for all campaigns sharing the
dataset. Legacy/local campaigns still use `opal__<slug>__label_hist` and the
configured `y_column_name`.

Shared-label campaigns should declare `writeback.prediction_records:
ledger_only` so `run` keeps predictions, scores, and selections in
campaign-local ledgers instead of writing prediction history into the shared
`records.parquet`. For `usr_sidecar` label sources, `ingest-y` writes observed
labels to the shared sidecar instead of duplicating assay truth into campaign
label-history columns, and unknown IDs fail unless they are explicitly dropped.
Sidecar appends are guarded by a local path lock around load/merge/write. This
is enough for local multi-campaign operation; multi-host shared ingest requires
a stronger lease/transaction layer.
If a campaign must be fully transient, copy the candidate table into a
campaign-local `records.parquet` and point the campaign at
`data.location.kind: local`.

### Boundary decisions

- upstream source assembly, optional construct expansion, and infer write-back remain outside OPAL
- OPAL consumes one explicit `X` column; it does not decide which infer job or model lane produced that column
- `data.location.kind: usr` is the contract for reading the USR dataset directly
- no hidden orchestration exists between `infer` and `opal`; the handoff is a deliberate choice of dataset plus `x_column_name`
- campaign-history labels and round state stay under OPAL namespaces;
  shared-label campaigns keep primary labels in the sidecar and write
  campaign-derived state to ledgers
- for shared-label multi-campaign studies, observed assay labels should be
  represented once as a study-level label source; campaign namespaces should
  hold derived predictions, selections, and run state, not duplicate primary
  label truth
- explicit shared-record prediction writeback requires an operator-visible
  records-path lock; the shared-label contract uses `ledger_only` and a
  sidecar path lock by default
- do not duplicate a USR dataset per campaign unless the source candidate table
  contract actually differs
- avoid broad OPAL-source cleanup on shared USR records; campaign-scoped pruning
  is the only safe cleanup posture when several campaigns share one table

### Preconditions

- one USR dataset already exists at a known root
- that dataset already has the chosen infer-derived `X` column such as `infer__evo2_7b__anchor_only_7b_features__intermediate_embedding__block26_mlp_out__seq_mean`
- or the Evo2 sequence bundle has already been materialized into one
  deterministic OPAL-ready `X` column outside OPAL
- labels will be ingested incrementally through OPAL rounds rather than attached silently during infer

For the upstream shared-dataset and infer matrix assembly, use:

- [Promoter characterization feature matrix](../../../usr/docs/operations/promoter-characterization-feature-matrix.md)

### Ordered procedure

### 1) Choose the dataset and the explicit `X` column

```bash
export USR_ROOT=/abs/path/to/usr_root # Reuse the same explicit USR root used for infer write-back.
export DATASET_ID="promoter_feature_matrix_demo" # Choose the infer-annotated dataset that OPAL should consume.
export X_COLUMN="infer__evo2_7b__anchor_only_7b_features__intermediate_embedding__block26_mlp_out__seq_mean" # Choose one concrete infer-derived feature column for this campaign.
export OPAL_WORKDIR=/abs/path/to/opal_campaign # Keep OPAL campaign state and ledgers outside the USR dataset root.
```

### 2) Point the OPAL campaign at the USR dataset

The cross-tool-specific contract is the `data` block in `campaign.yaml`:

```yaml
data: # Point OPAL at the infer-annotated USR dataset.
  location: { kind: usr, path: /abs/path/to/usr_root, dataset: promoter_feature_matrix_demo } # Resolve the USR root and dataset explicitly.
  x_column_name: "infer__evo2_7b__anchor_only_7b_features__intermediate_embedding__block26_mlp_out__seq_mean" # Choose one infer-derived feature column as X.
  y_column_name: "measured_activity" # Name the observed label column for this campaign.
  y_expected_length: 1 # Keep the baseline workflow on scalar labels.
```

For a shared-label multi-campaign study, also declare the label source and
prediction writeback policy explicitly:

```yaml
labels: # Use one observed-label sidecar for campaigns that share this dataset.
  source: { kind: usr_sidecar, dataset: promoter_feature_matrix_demo, path: _opal/observed_labels.parquet } # Keep labels dataset-local.
  y_space: scalar_v1 # Name the assay label space used by the sidecar rows.
writeback: # Keep prediction outputs out of the shared candidate table.
  prediction_records: ledger_only # Store predictions/scores/selections in campaign ledgers.
```

Use any OPAL model/objective/selection workflow you want after that. For a cheap deterministic tracer bullet, reuse the RF + `top_n` baseline:

```yaml
transforms_x: { name: identity, params: {} } # Pass the chosen infer-derived X column through unchanged.
transforms_y: { name: scalar_from_table_v1, params: {} } # Parse scalar labels from the observed table.

model: # Use a cheap deterministic surrogate for the first tracer bullet.
  name: random_forest # Select the RF baseline model plugin.
  params: { n_estimators: 100, random_state: 7 } # Keep the baseline model deterministic.

objectives: # Emit one scalar objective channel for baseline ranking.
  - name: scalar_identity_v1 # Reuse the scalar objective baseline.
    params: {} # Keep the scalar objective at default settings.

selection: # Rank candidates deterministically by the scalar objective channel.
  name: top_n # Use the deterministic top_n selector.
  params: # Keep selection wiring explicit in the campaign config.
    top_k: 12 # Select a small candidate set each round.
    score_ref: "scalar_identity_v1/scalar" # Rank by the scalar objective channel.
    objective_mode: maximize # Treat larger scalar values as better.
    tie_handling: competition_rank # Keep tie handling explicit.
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
  --observed-round 0 \
  --in "$OPAL_WORKDIR/inputs/r0_labels.xlsx" \
  --unknown-sequences drop \
  --if-exists replace \
  --apply
```

### 5) Run the first round against the infer-derived `X` column

```bash
uv run opal run -c "$OPAL_WORKDIR/configs/campaign.yaml" --labels-as-of 0 # Train, score, and select against the chosen infer-derived X column.
```

### 6) Verify artifacts and round state

```bash
uv run opal verify-outputs -c "$OPAL_WORKDIR/configs/campaign.yaml" --round latest # Verify ledgers and round outputs.
uv run opal status -c "$OPAL_WORKDIR/configs/campaign.yaml" # Inspect current round and selection state.
uv run opal runs list -c "$OPAL_WORKDIR/configs/campaign.yaml" # Review recorded runs for this campaign.
uv run opal ctx audit -c "$OPAL_WORKDIR/configs/campaign.yaml" --round latest # Audit the round contract payload.
```

### 7) Iterate as labels accumulate

```bash
# Ingest the next observed label batch before resuming the round loop.
uv run opal ingest-y \
  -c "$OPAL_WORKDIR/configs/campaign.yaml" \
  --observed-round 1 \
  --in "$OPAL_WORKDIR/inputs/r1_labels.xlsx" \
  --unknown-sequences drop \
  --if-exists replace \
  --apply
uv run opal run -c "$OPAL_WORKDIR/configs/campaign.yaml" --labels-as-of 1 --resume # Refit and reselection with labels visible through round 1.
```

### Model and selection variations

- use [RF + SFXI + top_n](rf-sfxi-topn.md) for the cheapest deterministic baseline
- use [GP + SFXI + top_n](gp-sfxi-topn.md) when you want predictive uncertainty recorded but deterministic ranking
- use [GP + SFXI + expected_improvement](gp-sfxi-ei.md) when selection should consume both score and uncertainty

The infer-to-OPAL handoff contract does not change across those choices. Only `x_column_name` and the campaign config do.

## Verification checklist

- `campaign.yaml` uses `data.location.kind: usr`
- `campaign.yaml` declares one explicit `x_column_name`
- `opal validate` passes before any round runs
- `opal run` succeeds with labels visible through the intended `--labels-as-of` cutoff
- `verify-outputs` reports zero mismatches for the latest round

## Related docs

- Root docs router: [../../../../../docs/README.md](../../../../../docs/README.md)
- Upstream shared-dataset and infer matrix assembly: [../../../usr/docs/operations/promoter-characterization-feature-matrix.md](../../../usr/docs/operations/promoter-characterization-feature-matrix.md)
- OPAL docs index: [../index.md](../index.md)
- OPAL configuration contract: [../reference/configuration.md](../reference/configuration.md)
- OPAL CLI reference: [../reference/cli.md](../reference/cli.md)
- Cluster exploratory branch: [../../../cluster/docs/workflows/exploratory-clustering.md](../../../cluster/docs/workflows/exploratory-clustering.md)

## OPAL Architecture and Data Flow

**Owner:** dnadesign-maintainers
**Last verified:** 2026-07-15

OPAL executes each round through the stages below.

### Round lifecycle

1. Load `configs/campaign.yaml` and validate schema + plugin names.
2. Resolve one label snapshot from the configured `labels` source.
3. Apply candidate-scope and candidate-eligibility rules before scoring. These
   rules declare their required candidate columns, filter rows, and do not
   mutate `records.parquet`.
4. Build feature matrices with `transforms_x`.
5. Fit `model` and predict `y_pred` (and optional predictive std-dev).
6. Apply `training.y_ops` inversion to both mean and std-dev when configured.
7. Evaluate every `selection_views[].objective` against the shared phenotype
   prediction. View IDs namespace score and uncertainty channels.
8. Run each view's selector independently and persist long-form selection sets.
9. Build one deterministic `selection_batch` union without implicit fill or
   discard behavior.

### Runtime surfaces

- Source records: `records.parquet`
- Label source: `labels.source.kind`, either `campaign_history` or
  `usr_sidecar`; a USR sidecar may be mutable or pinned to a study-issued
  `opal.observed_label_promotion.v1` manifest
- Round artifacts: `outputs/rounds/round_<k>/...`
- Ledger sinks:
  - `outputs/ledger/labels.parquet`
  - `outputs/ledger/predictions/`
  - `outputs/ledger/runs.parquet`

### Config to stage mapping

- `campaign`, `data`: workspace and dataset resolution.
- `labels`: training-label source resolution and batch/round label semantics.
- `transforms_y`: ingest-only label construction.
- `candidate_eligibility`: generic pre-selection exclusion rules and audit
  reports; study-specific cloning or ordering semantics must enter through
  rule parameters, not OPAL candidate records.
- `transforms_x`: feature matrix for training/scoring.
- `training.y_ops`: fit-time Y transforms and inference-time inversion.
- `model`: fit/predict implementation.
- `selection_views`: named objective instances and ranking policies over
  explicit, view-local channel refs.
- `selection_batch`: deduplication and optional exact-cardinality contract for
  the logical union.
- `scoring`: prediction batch size.
- `safety`: preflight guards before writes.

### Channel contract

- Each view objective emits score channels and optional uncertainty channels.
- A view selector reads unqualified plugin channels such as
  `score_ref: feasibility_margin`; persisted refs are namespaced as
  `<selection_view_id>/<channel>`.
- `objective_mode` and `tie_handling` are explicit required controls.

A campaign owns learning, a selection view owns a target, and a selection
batch owns the logical union. Different setpoints over the same X, Y, labels,
and model are selection views, not separate campaigns.

### Failure model

OPAL is fail-fast by design:
- unknown plugins fail at config load/validation
- unknown candidate-eligibility rules or invalid rule parameters fail during
  config validation before round execution
- unresolved score/uncertainty refs fail before selection
- duplicate view IDs and objective channel collisions fail at config load
- non-finite/invalid model/objective/selection outputs fail before writeback
- declared selection-batch cardinality mismatch fails without filling slots
- ledger schema violations fail at write time
- a configured shared label sidecar fails rather than falling back to
  campaign-local label history when it is missing, malformed, or points at
  unknown candidate IDs
- a manifest-pinned sidecar fails when campaign, study, Y space, candidate/X
  snapshot, path, digest, schema, columns, or row count differs from its
  promotion manifest; generic ingest cannot mutate that source

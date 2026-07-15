## OPAL Campaign Configuration v3

**Owner:** dnadesign-maintainers
**Last verified:** 2026-07-15

`campaign.yaml` uses the strict schema `opal.campaign.v3`. OPAL rejects v2
keys. There is no compatibility parser.

The central invariant is:

> A campaign owns learning; a selection view owns a target; a selection batch owns the logical union.

### Ontology

- **Campaign:** candidate universe, X, Y, labels, transforms, model, and round
  history shared by one learning loop.
- **Selection view:** one objective instance, objective parameters, selector,
  and top-k policy evaluated from the campaign's shared predictions.
- **Round:** one immutable label snapshot, one model fit, one prediction pass,
  and all declared selection views.
- **Selection set:** rows selected by one view.
- **Selection batch:** deterministic union of all selection sets, deduplicated
  by the configured candidate field.

Use separate campaigns only when X, Y, labels, transforms, model, candidate
universe, or round lifecycle differ. Different target masks or setpoints over
the same predicted phenotype belong in selection views of one campaign.

### Required blocks

- `schema_version: opal.campaign.v3`
- `ownership`: `opal_demo` or `study_campaign` with its required identifiers
- `campaign`: `name`, `slug`, `workdir`
- `data`: candidate location and X/Y columns
- `transforms_x`, `transforms_y`, `model`
- `selection_views`: non-empty list of named objective/selection pairs

Study-owned execution uses:

```yaml
ownership:
  owner_scope: study_campaign
  study_id: stress_ethanol_cipro_growth
  dataset_id: usr_prom_eth_cip_opal_candidates
  portable: false
```

Portable tool examples use `owner_scope: opal_demo`, omit study and dataset
IDs, and set `portable: true`. Runtime state does not belong in `ownership`;
read `state.json` and the run ledger instead.

`selection_batch` is optional. If omitted, OPAL deduplicates by `id` and does
not enforce a cardinality. Declare `expected_unique_count` only when exact
logical-batch cardinality is part of the method; mismatch is fatal and OPAL
never fills or discards rows implicitly.

### Selection-view contract

Every view requires a unique slug-like `id` and:

```yaml
selection_views:
  - id: ethanol
    objective:
      name: response_magnitude_feasibility_v1
      params: {...}
    selection:
      name: top_n
      params:
        top_k: 6
        score_ref: feasibility_margin
        objective_mode: maximize
        tie_handling: competition_rank
        require_exact_top_k: true
```

Channel references inside a view are unqualified plugin channel names. OPAL
namespaces persisted channels as `<selection_view_id>/<channel>`. This permits
multiple instances of one objective plugin without collisions.

`require_exact_top_k` is optional and defaults to `false`. When true, OPAL
requires the normalized selection to contain exactly `top_k` rows. Boundary
ties therefore fail before ledger or batch publication; OPAL does not truncate
or fill the result.

`expected_improvement` also requires `uncertainty_ref`. The referenced model
and objective path must emit predictive standard deviation; missing or invalid
uncertainty is fatal.

### Multi-view example

```yaml
schema_version: opal.campaign.v3

ownership:
  owner_scope: study_campaign
  study_id: example_two_factor_study
  dataset_id: candidates
  portable: false

campaign:
  name: "Two-factor response campaign"
  slug: two_factor_response
  workdir: "."

data:
  location: {kind: usr, path: src/dnadesign/usr/datasets, dataset: candidates}
  x_column_name: sequence_features
  y_column_name: response_window_y
  y_expected_length: 8

labels:
  source:
    kind: usr_sidecar
    dataset: candidates
    path: _opal/response_window_labels_v1/observed_labels.parquet
    manifest_path: _opal/response_window_labels_v1/promotion.manifest.json
  y_space: reader_response_window_vector_v1
  id_column: id
  round_column: observed_round
  batch_column: batch_id
  dedup_policy: error_on_duplicate

writeback:
  prediction_records: ledger_only

transforms_x: {name: identity, params: {}}
transforms_y:
  name: vector_from_table_v1
  params:
    id_column: id
    value_columns: [r00, r10, r01, r11, b00, b10, b01, b11]

model:
  name: random_forest
  params: {n_estimators: 100, random_state: 7, n_jobs: -1}

selection_views:
  - id: factor_a
    objective:
      name: response_magnitude_feasibility_v1
      params:
        state_ids: ["00", "10", "01", "11"]
        target_mask: [0, 1, 0, 1]
        calibration: &calibration
          response_separation_min: 0.0
          on_magnitude_min: 0.0
          off_magnitude_max: 0.0
          response_separation_scale: 1.0
          on_magnitude_scale: 1.0
          off_magnitude_scale: 1.0
    selection:
      name: top_n
      params: {top_k: 6, score_ref: feasibility_margin, objective_mode: maximize, tie_handling: competition_rank}

  - id: factor_b
    objective:
      name: response_magnitude_feasibility_v1
      params:
        state_ids: ["00", "10", "01", "11"]
        target_mask: [0, 0, 1, 1]
        calibration: *calibration
    selection:
      name: top_n
      params: {top_k: 6, score_ref: feasibility_margin, objective_mode: maximize, tie_handling: competition_rank}

selection_batch:
  deduplicate_by: sequence
  expected_unique_count: 12
```

### Shared execution

For each round OPAL:

1. resolves one label snapshot;
2. prepares X and Y once;
3. fits one multi-output model once;
4. predicts the candidate universe once;
5. evaluates every selection view from those shared predictions;
6. writes one selection set per view and one deduplicated selection batch.

Validation rejects duplicate view IDs, qualified or unresolved channel refs,
plugin-output collisions, unsupported selectors, and view-specific drift in
already-labeled exclusion policy.

### Candidate and label contracts

`data.candidate_scope` optionally restricts scoring to an ID table without
copying the candidate table. `candidate_eligibility` applies generic,
manifested exclusion rules before scoring. Neither surface changes labels or
rewrites source candidates.

The streaming runtime loads only core candidate fields plus columns declared by
the configured eligibility plugins and `selection_batch.deduplicate_by`.
Eligibility plugins own that required-column declaration. A configured column
missing from `records.parquet` is an execution error; OPAL does not silently
load the full candidate table or skip the rule. The configured X column cannot
serve as candidate metadata; X stays in the bounded score-batch stream.

`labels.source.kind` is `campaign_history` or `usr_sidecar`. A configured USR
sidecar must match the candidate dataset and must exist for execution. OPAL does
not fall back to campaign history. v3 predictions are always ledger-only;
`writeback.prediction_records` accepts only `ledger_only`.

`labels.source.manifest_path` is optional for a USR sidecar and is relative to
the same dataset root as `labels.source.path`. When present, it pins the label
source to a study-published snapshot and requires `study_campaign` ownership.
OPAL verifies the manifest each time it reads the sidecar and rejects generic
`ingest-y` writes. When absent, the sidecar remains an OPAL-managed mutable
label source.

The pinned manifest uses this objective-agnostic observed-label contract:

```json
{
  "schema_version": "opal.observed_label_promotion.v1",
  "campaign_slug": "two_factor_response",
  "study_id": "example_two_factor_study",
  "y_space": "reader_response_window_vector_v1",
  "study_provenance": {
    "schema_id": "example_two_factor_study.observed_labels.v1",
    "path": "_opal/response_window_labels_v1/study_provenance.json",
    "sha256": "<lowercase 64-character SHA-256>"
  },
  "candidate_artifact": {
    "path": "records.parquet",
    "sha256": "<lowercase 64-character SHA-256>",
    "row_count": 42000,
    "columns": ["id", "sequence", "<configured X column>"],
    "schema_sha256": "<lowercase 64-character SHA-256>"
  },
  "label_artifact": {
    "path": "_opal/response_window_labels_v1/observed_labels.parquet",
    "sha256": "<lowercase 64-character SHA-256>",
    "row_count": 24
  }
}
```

The campaign slug, study ID, Y-space ID, study-provenance artifact, candidate
snapshot, label path, digests, schemas, columns, and Parquet row counts must
match exactly. The candidate snapshot must contain the configured candidate-ID
and X columns. Paths cannot escape the dataset root. The study-owned provenance artifact records the assay bundle,
identity binding, reduction, and aggregation contracts needed to interpret Y;
OPAL verifies its digest without interpreting study fields. The study publisher
stages the label, provenance, and promotion records as one publication. OPAL reads and verifies that
promotion but does not publish or revise it.

### Defaults and fail-fast behavior

- `ingest.duplicate_policy`: `error`
- `scoring.score_batch_size`: `10000`
- `selection_batch.deduplicate_by`: `id`
- `training.y_ops`: empty
- safety guards: mixed type/alphabet rejection, duplicate-ID rejection,
  canonical X validation, and an 8 GiB X-matrix budget

Unknown plugins, duplicate YAML keys, malformed vectors, missing sidecars,
non-finite model/objective output, batch underfill, and ledger schema drift are
errors. Plot rendering is separate from round execution and cannot alter model
or selection state.

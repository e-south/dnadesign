## OPAL Data and Artifact Contracts v3

**Owner:** dnadesign-maintainers
**Last verified:** 2026-07-15

### Candidate records

`records.parquet` requires unique string `id`, `sequence`, `bio_type`, and
`alphabet` columns. The configured X column must be a non-null, finite Arrow
`fixed_size_list<float32|float64>[x_dim]`. Ragged lists and serialized arrays
are import formats, not runtime formats.

Observed Y may come from campaign label history or a typed dataset-local
sidecar. For `usr_sidecar`, required columns are:

- `id`: candidate ID
- `observed_round`: integer label round
- `batch_id`: study batch identity
- `y_space`: exact configured label-space ID
- `y_obs`: finite fixed-length vector

OPAL rejects unknown IDs, mixed Y spaces, malformed lengths, and a missing
configured sidecar. It does not infer or fall back to another label source.

A USR sidecar may declare `labels.source.manifest_path`. That manifest must use
`opal.observed_label_promotion.v1` and bind the artifact to the configured
campaign slug, study ID, Y-space ID, a digest-pinned study-provenance manifest,
the exact `records.parquet` candidate/X snapshot, and the relative sidecar path,
SHA-256 digest, schema, columns, and Parquet row count. OPAL verifies the
candidate snapshot, provenance, and labels before every read. The study-provenance manifest owns assay,
identity, reduction, and aggregation semantics; OPAL records its schema and
digest without importing study logic. A
manifest-pinned sidecar is immutable through generic OPAL ingest; its owning
study publishes the label table, provenance manifest, and promotion manifest
together. Any later candidate ID, sequence, X, schema, or row change invalidates
the promotion until the study publishes a new versioned snapshot.

### Shared prediction ledger

`outputs/ledger/predictions/` stores one row per scored candidate and run:

- `event`, `run_id`, `as_of_round`, `id`, `sequence`
- `pred__y_dim`, `pred__y_hat_model`
- `pred__score_channels`, `pred__uncertainty_channels`
- `pred__selection_views`

`pred__selection_views` is a list of structs, one per named view:

- `selection_view_id`, `objective_name`, `selection_name`
- `score`, `score_ref`, `selection_score`
- `rank_competition`, `is_selected`, `top_k`
- optional `uncertainty`, `uncertainty_ref`
- view-local `diagnostics`

The shared model prediction is stored once. Public readers project one view to
the analysis fields `view__score`, `view__selection_score`,
`view__rank_competition`, `view__is_selected`, and related metadata. Raw ledger
consumers must not invent a default view.

### Run ledger

`outputs/ledger/runs.parquet` stores one row per shared model fit:

- model, X transform, Y ingest, and training-Y operation metadata
- `objective__defs_json`: view-indexed objective declarations
- `selection_views__defs_json`: view-indexed selector declarations and summary
  statistics
- training/scoring counts
- artifact paths and digests
- ledger and OPAL schema versions

### Round artifacts

Each `outputs/rounds/round_<k>/` contains:

- `model/`: one shared model artifact and optional shared diagnostics
- `predictions` in the append-only ledger, referenced by run ID
- `metadata/objective_meta.json`: all objective and selection-view definitions
- `selection/selections.parquet`: long-form selected rows keyed by
  `selection_view_id`; `score` is the objective channel value and
  `selection_score` is the selector's ranking value
- `selection/selection_batch.parquet`: deduplicated logical union with
  `selection_view_ids` and `selection_memberships`; each membership retains
  both scores
- `labels/labels_used.parquet`: immutable training-label snapshot
- logs and context snapshots

`selections.parquet` is the verification artifact for one view.
`selection_batch.parquet` is the logical union for downstream study review.
Neither artifact authorizes synthesis.

### Public inspection contracts

- `opal selection-set show/export --view <id>` projects and verifies one view.
- `opal selection-batch show/export` reads the logical union.
- `opal verify-outputs --view <id>` compares one selection artifact to the
  shared prediction ledger.
- `opal record-show --view <id>` reports view-specific rank and score.

When a campaign has multiple views, view-specific commands require `--view`.
They never choose the first configured view silently.

### Plot and notebook artifacts

View-specific plot manifests require `selection_view_id`. In multi-view
campaigns, outputs are namespaced under
`outputs/plots/selection_views/<view_id>/`. A generated notebook exposes one
`Selection view` control and filters masks, scores, selections, and plot
deliverables to that view. Shared model diagnostics are rendered once.

### Validation

Ledger schemas are strict. Unknown columns, duplicate prediction IDs within a
run, unresolved selection rows, non-finite values, mixed run IDs, and artifact
digest mismatches are errors. `OPAL_LEDGER_ALLOW_EXTRA=1` exists only for
controlled schema development and must not be used in production workflows.

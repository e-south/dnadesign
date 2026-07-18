## OPAL Data and Artifact Contracts v3

**Owner:** dnadesign-maintainers
**Last verified:** 2026-07-18

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
The immutable event key is `(id, observed_round)`: duplicate events fail, while
the same candidate may appear again in a strictly later round. Cumulative
training applies the campaign's declared cross-round policy, and run-scoped
observed-event snapshots retain every verified event and exact `batch_id`.

A USR sidecar may declare `labels.source.manifest_path`. That manifest must use
`opal.observed_label_promotion.v1` and bind the artifact to the configured
campaign slug, study ID, Y-space ID, a digest-pinned study-provenance manifest,
the exact `records.parquet` candidate/X snapshot, and the relative sidecar path,
SHA-256 digest, schema, columns, and Parquet row count. OPAL verifies the
candidate snapshot, provenance, and labels before every read. The study-provenance manifest owns assay,
identity, reduction, and candidate-observation formation semantics; OPAL
records its schema and
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

`outputs/ledger/runs.parquet` stores one row per shared model fit. Each fit has a
collision-resistant `run_id`; committed run IDs are create-only across both the
run and prediction ledgers:

- model, X transform, Y ingest, and training-Y operation metadata
- `objective__defs_json`: view-indexed objective declarations
- `selection_views__defs_json`: view-indexed selector declarations and summary
  statistics
- training/scoring counts
- artifact paths and digests
- ledger and OPAL schema versions

### Round artifacts

Each `outputs/rounds/round_<k>/` contains mutable latest-run mirrors for command
compatibility plus immutable evidence for every retained run:

- `model/`: one shared model artifact and optional shared diagnostics
- `predictions` in the append-only ledger, referenced by run ID
- `metadata/objective_meta.json`: all objective and selection-view definitions
- `selection/selections.parquet`: long-form selected rows keyed by
  `selection_view_id`; `score` is the objective channel value and
  `selection_score` is the selector's ranking value
- `selection/selection_batch.parquet`: final deduplicated batch with
  `selection_view_ids`, `preferred_view_ids`, allocation ownership, and
  `selection_memberships`; each membership retains both scores
- `selection/allocation_trace.parquet`: when coordinated allocation is
  configured, the ordered allocated and skipped-overlap decisions with view,
  slot, ranks, scores, batch key, and conflict owner
- `run_artifacts/<run-slug>/`: a digest-bound snapshot of every artifact named
  by that run's ledger row, including model, selection, metadata, and label
  evidence
- logs and context snapshots

The run ledger addresses each snapshot by its stable logical key, including
`labels/labels_used.parquet` and `labels/observed_events.parquet`. A same-round
resume replaces only the latest-run mirrors, retains every prior run directory,
and creates a new immutable snapshot; it never rewrites evidence pinned by an
earlier run.

`selections.parquet` is the verification artifact for one view.
`selection_batch.parquet` is the final physical-batch proposal for downstream
study review. Under the default policy it is the logical union; under explicit
allocation it contains the exact unique-slot result. Neither artifact
authorizes synthesis.

`opal selection-batch show/export` exposes this artifact as
`opal.selection_batch.v3`. The loader requires `run_id`, `as_of_round`, and
`campaign_slug` provenance on every row and verifies them against the resolved
run. Selection rows carry the projected `selection_batch_key` and
`deduplicate_by` fields used for that run. The loader verifies the batch and
long-form selection artifacts against their run-ledger SHA-256 digests, then
reconciles each nested batch membership to the corresponding candidate/view
selection row: batch key, ranks, scores, score reference, selection origin, and
allocation slot must agree. Coordinated allocations additionally require the
digest-bound allocation trace; `preferred_view_ids` must match its complete set
of top-k preferences for the deduplication key. It also verifies configured
view membership and allocation ownership.
Logical-union rows are returned in competition-rank order;
coordinated rows follow allocation slot and declared view priority. Rows that
do not satisfy the v3 contract fail validation; the loader does not infer or
upgrade missing provenance. An explicit batch-path audit override bypasses only
the batch-file digest and must still reconcile to the digest-bound selection
artifact.

### Public inspection contracts

- `opal selection-set show/export --view <id>` projects and verifies one view.
- `opal selection-batch show/export` reads the final deduplicated batch.
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

### Reader evidence manifest adapter

Reader evidence remains producer-owned. A study may keep its own
`schema_version`, but it must opt into OPAL's notebook surface with
`opal_adapter: opal.reader_evidence_manifest.v1`. The public adapter is a
projection contract, not a study-schema alias.

The adapter requires:

- a trimmed producer `schema_version` and round label;
- a `rows` list whose entries have a candidate or record ID, Reader design ID,
  Reader experiment ID, and an `artifacts` list;
- artifact entries with semantic kind, producer kind, record ID, scope, path,
  existence flag, and media type; and
- an exact five-field summary: row count, distinct ID count, Reader experiment
  count, artifact count, and rows with missing artifact evidence.

OPAL recomputes every summary count from the projected rows. Missing or
duplicated artifact identities, malformed fields, non-boolean existence flags,
and count drift reject the adapter before notebook discovery. Producer-specific
assay semantics, candidate identity, artifact digests, and scientific claims
remain in their owning contracts; the adapter only defines the common evidence
shape OPAL is allowed to display.

### Validation

Ledger schemas are strict. Unknown columns, duplicate or previously committed
run IDs, duplicate prediction IDs within a run, unresolved selection rows,
non-finite values, mixed run IDs, unsupported observed-label source kinds, and
artifact digest mismatches are errors. `OPAL_LEDGER_ALLOW_EXTRA=1` exists only for
controlled schema development and must not be used in production workflows.

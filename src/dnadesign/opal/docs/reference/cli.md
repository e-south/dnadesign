---
id: opal-reference-cli
title: OPAL Command Line Interface
owner: dnadesign-maintainers
status: active
last_verified: 2026-07-15
audience:
  - operator
  - maintainer
  - agent
entrypoints:
  cli: uv run opal
---

**Owner:** dnadesign-maintainers
**Last verified:** 2026-07-15

## OPAL Command Line Interface


The OPAL CLI initializes campaigns, ingests labels, executes rounds, inspects
records and models, validates data contracts, and generates plots.

Commands are registry-driven and plugin‑agnostic: they operate on the configured plugin names and enforce only declared contracts.

### Command overview

Each command has one operational purpose. Usage blocks show required arguments;
optional flags appear in brackets.

Round semantics:

- `ingest-y --round` stamps label events with the observed round.
- `run/explain --round` chooses the training cutoff (labels with
  `observed_round <= round`).

Guided hints:

- Human output for `init`, `validate`, `ingest-y`, `run`, `explain`, and `verify-outputs` prints next-step hints by default.
- Use `--no-hints` to disable hint lines.

### `guide`

Generate a guided runbook from the current campaign config.

**Usage**

```bash
opal guide --config <yaml> [--round <r>] [--format text|markdown|json] [--out <path>]
opal guide next --config <yaml> [--round <r>] [--json]
```

**Notes**

* `guide` is read-only. It summarizes plugin wiring, lifecycle steps, round semantics, and deep-dive docs/source pointers.
* `guide next` inspects the candidate table, campaign state, and label source,
  then prints the recommended next command sequence. If `records.parquet` is
  missing, it reports `stage=candidate_table` before suggesting ingest or run.
* Prefer `guide next` in agent/automation loops for state-aware progression.

---

### `demo-matrix`

Run canonical demo workflows end-to-end in isolated temp copies.

**Usage**

```bash
opal demo-matrix [--tmp-root <dir>] [--rounds 0|0,1] [--json] [--keep] [--fail-fast]
```

**Notes**

* Runs `demo_rf_sfxi_topn`, `demo_gp_topn`, and `demo_gp_ei`.
* Executes reset/init/validate/ingest/run/verify checks per flow and round.
* Exits non-zero when any flow fails or verify mismatches occur.
* Intended for demo pressure testing and CI-style validation.

### `init`

Initialize/validate a campaign workspace and write `state.json`.

**Usage**

```bash
opal init --config <yaml> [--json]
```

**Flags**

* `--config, -c`: Path to `configs/campaign.yaml` (required unless `$OPAL_CONFIG` is set).
* `--json`: Output as machine-readable JSON (default output is plain text).

**Notes**

* Ensures the campaign `workdir` has `outputs/`.
* Writes/updates `state.json` with campaign identity, data location, and settings.

---

### `ingest-y`

Transform a tidy CSV/Parquet/XLSX to model-ready **Y**, preview, confirm, and
append to the configured label source.

**Usage**

```bash
opal ingest-y --config <yaml> --round <r> --csv <path> \
  [--transform <name>] [--params <transform_params.json>] \
  [--unknown-sequences create|drop|error] [--infer-missing-required] \
  [--if-exists fail|skip|replace] [--apply] [--json]
```

**Flags**

* `--config, -c`: Path to `configs/campaign.yaml` (required unless `$OPAL_CONFIG` is set).
* `--round, -r`: Observed round stamp for these labels.
* `--csv`: CSV/Parquet/XLSX input (`.csv`, `.parquet`, `.pq`, or `.xlsx`).
* `--transform`: Override YAML `transforms_y.name`.
* `--params`: JSON file (.json) with transform params (overrides YAML `transforms_y.params`).
* `--unknown-sequences`: How to handle sequences not found in records (default: `create`). Use `drop` to skip
  unknown sequences when required columns are missing or for a strict in-place update.
* `--infer-missing-required`: Auto-fill missing required columns for new sequences (`bio_type`, `alphabet`)
  using the most common values found in `records.parquet`.
* `--if-exists`: Behavior if `(id, round)` already exists in the configured label source (`fail`/`skip`/`replace`).
* `--apply`: Apply ingest without interactive confirmation.
* `--json`: Output as machine-readable JSON (default output is plain text). With `--apply`, the command emits
  one final JSON object containing commit counts, preview data, and `ingest_runtime` telemetry.

**Behavior & checks**

* Uses `transforms_y` from YAML unless overridden by `--transform/--params`.
* **Strict preflights**: schema checks, completeness.
* **Preview is printed** (counts + sample) before any write.
* Duplicate handling is controlled by `ingest.duplicate_policy` (error | keep_first | keep_last).
* **New IDs** are allowed only for campaign-history flows when the input
  includes `sequence`, `bio_type`, `alphabet`, and the
  configured X column. Shared `usr_sidecar` label sources use a fixed candidate
  universe and reject unknown IDs unless `--unknown-sequences drop` is used.
* If new sequences are missing required columns, OPAL will prompt to infer defaults for `bio_type`/`alphabet`
  (or use `--infer-missing-required` for non-interactive runs). For other missing columns, use
  `--unknown-sequences drop` or provide the columns.
* If `records.parquet` contains duplicate sequences, `ingest-y` requires an explicit `id` column for all rows
  to avoid ambiguous sequence → id mapping.
* When unknown sequences are missing **X** data, `ingest-y --unknown-sequences create` fails fast.
  Provide X values for new rows or pass `--unknown-sequences drop` to skip unknown rows explicitly.
* If adding **new sequences** and X is list-valued, prefer **Parquet** input so the X column remains list-typed
  (CSV will coerce lists to strings).
* For `labels.source.kind: campaign_history`, appends to
  `opal__<slug>__label_hist` and writes the current Y column.
* For `labels.source.kind: usr_sidecar` without `manifest_path`, appends
  observed labels to the shared sidecar such as
  `_opal/observed_labels.parquet`; it does not duplicate assay truth into
  campaign label-history columns. Fixed-universe sidecar ingest loads only the
  records identity frame (`id`, `sequence`) and does not materialize the
  configured X column.
* For a USR sidecar with `manifest_path`, `ingest-y --apply` fails before any
  label or ledger write. The source is a study-published immutable snapshot;
  its owning workflow publishes a new Parquet artifact and promotion manifest.
* The text preview includes a `[Runtime] ingest-y` block. JSON output includes
  `ingest_runtime.schema_version: opal.ingest_runtime.v1`, `mode`, loaded
  columns, candidate index rows, estimated frame memory, unknown-sequence policy,
  and write scope. Shared `usr_sidecar` appends report `mode=identity_index`,
  `write_scope=label_sidecar`, and `full_records_loaded=false`.
* Emits `label` events into `outputs/ledger/labels.parquet`.

---

### `run`

Train on labels with **`observed_round <= R`**, score the candidate universe,
evaluate every selection view, and write campaign artifacts and ledgers.

**Usage**

```bash
opal run --config <yaml> --round <r> \
  [--k <n>] [--resume] [--score-batch-size <n>] [--max-x-matrix-gib <gib>] \
  [--verbose|--quiet] [--json]
```

**Flags**

* `--config, -c`: Path to `configs/campaign.yaml` (required unless `$OPAL_CONFIG` is set).
* `--round, -r`: Training cutoff (use labels with `observed_round <= r`).
* `--k, -k`: Override `selection_views[].selection.params.top_k` for every
  declared selection view.
* `--score-batch-size`: Override `scoring.score_batch_size` for this run.
* `--max-x-matrix-gib`: Override `safety.max_x_matrix_gib` for this run. Prefer lowering `--score-batch-size` before raising this on memory-constrained hosts.
* `--resume`: Allow overwriting existing per-round artifacts (required if `outputs/rounds/round_<r>/` already contains artifacts). When set, the round directory is wiped before writing new artifacts.
* `--verbose/--quiet`: Control log verbosity (default: verbose).
* `--json`: Output as machine-readable JSON (default output is plain text).

**Pipeline**

* Pulls effective labels per `training.policy` (cumulative vs current round, dedup policy).
* Validates the Parquet X contract in bounded batches before round execution.
* Loads record metadata without X, including the columns declared by candidate
  eligibility plugins and `selection_batch.deduplicate_by`, then streams
  model-ready candidate X in bounded score batches.
* Aborts if the train plus score batch X footprint exceeds `safety.max_x_matrix_gib`.
* Predicts in batches (`scoring.score_batch_size` or `--score-batch-size`).
  The batch stream may follow Parquet storage order; OPAL realigns predictions
  to the requested candidate ID order before objectives, selection, and ledger
  writes, and fails fast on missing, extra, or duplicate streamed IDs.
* Fits one model and predicts the candidate universe once.
* Evaluates every `selection_views` objective against the shared prediction.
* Resolves each view's `selection.params.score_ref` and optional
  `uncertainty_ref`, then applies that view's selector and tie policy.
  * If a view sets `selection.params.exclude_already_labeled: true` (default), designs already labeled are **excluded**;
    scope is controlled by `training.policy.allow_resuggesting_candidates_until_labeled`.
* Builds `selection_batch` as the declared deduplicated union of all selection
  sets. A declared `expected_unique_count` is exact. OPAL fails on overlap
  under the default union contract; an explicit
  `round_robin_next_best_unallocated` allocation may instead advance each view
  through its deterministic ranking. Every skipped overlap and replacement is
  recorded; OPAL never fills or drops candidates silently.

**Artifacts written** (`outputs/rounds/round_<r>/`)

* `model/`
  * `model.joblib`
  * `model_meta.json`
  * `feature_importance.csv` (optional)
* `selection/`
  * `selections.parquet` (long form, keyed by `selection_view_id`)
  * `selection_batch.parquet` (final deduplicated batch)
  * `allocation_trace.parquet` (configured unique-slot allocation only)
* `labels/`
  * `labels_used.parquet` (training snapshot for this run)
* `metadata/`
  * `round_ctx.json`
  * `objective_meta.json`
* `logs/`
  * `round.log.jsonl` — compact JSONL with stage events and prediction batch progress

**Events appended** to **ledger sinks** under `outputs/`

* `run_pred` → `outputs/ledger/predictions/` (one row per candidate with one
  **`pred__y_hat_model`** plus nested **`pred__selection_views`** score,
  rank, selection, uncertainty, and diagnostic records).
* `run_meta` → `outputs/ledger/runs.parquet` (one row per run with the shared
  model/config snapshot, view definitions, and artifact checksums).

`pred__y_hat_model` is **objective-space** (after any Y‑ops inversion), so downstream logic is plugin‑agnostic.

**Reruns & non-interactive mode**

Rerunning a round already present in `state.json` prompts before overwriting.
In non-TTY contexts such as CI, the command exits and requires an explicit
rerun with `--resume`.

`opal run` keeps predictions, scores, and selections in campaign ledgers and
does not mutate `records.parquet`.

---

### `predict`

Run **ephemeral** predictions from a frozen model. No writes to `records.parquet`.

**Usage**

```bash
opal predict --config <yaml> \
  [--model-path <path> | --round <r>] \
  [--model-name <registry_name> --model-params <params.json>] \
  [--in <csv|parquet>] [--out <csv|parquet>] \
  [--id-col <name>] [--sequence-col <name>] \
  [--generate-id-from-sequence] [--assume-no-yops]
```

**Flags**

* `--config, -c`: Path to `configs/campaign.yaml` (required unless `$OPAL_CONFIG` is set).
* `--model-path`: Path to `model.joblib` (overrides `--round`, e.g. `outputs/rounds/round_<r>/model/model.joblib`).
* `--round, -r`: Round index to resolve model from `state.json` (default: latest). Accepts `latest`.
* `--model-name` / `--model-params`: Required if `model_meta.json` is missing. `--model-params` must be a `.json`.
* `--in`: Optional input CSV/Parquet (`.csv`, `.parquet`, `.pq`; defaults to `records.parquet`).
* `--out`: Optional output CSV/Parquet (`.csv`, `.parquet`, `.pq`; defaults to stdout CSV).
* `--id-col`, `--sequence-col`: Column names in the input table.
* `--generate-id-from-sequence`: Deterministically generate ids if id column is missing.
* `--assume-no-yops`: Skip Y‑ops inversion even if training used Y‑ops.

**Notes**

* `--model-path` and `--round` are mutually exclusive; passing both is an error.
* When resolving from `state.json`, OPAL requires a recorded `model.artifact_path` for the selected round.
* OPAL fails fast if `model.artifact_path` is missing, relative, missing on disk, or points to a directory.
* Defaults to `records.parquet` when `--in` is omitted.
* Writes CSV to stdout by default; use `--out` for CSV/Parquet files (Parquet keeps vectors as list<float>).

---

### `record-show`

Per-record history report (ground truth + per-round predictions/rank/selected).

**Usage**

```bash
opal record-show --config <yaml> \
  [<ID-or-SEQ> | --id <ID> | --sequence <SEQ> | --selected-rank <n>] \
  [--view <selection-view-id>] \
  [--round <k|latest>] \
  [--run-id <id>] [--with-sequence|--no-sequence] [--json]
```

**Flags**

* `--config, -c`: Path to `configs/campaign.yaml` (required unless `$OPAL_CONFIG` is set).
* `<ID-or-SEQ>`: Positional id or sequence (use `--id/--sequence` to disambiguate).
* `--id`, `--sequence`: Explicit lookup key (mutually exclusive).
* `--selected-rank`: Resolve an ID from the named selection set by competition rank (1-based).
* `--view`: Selection view used for prediction fields and `--selected-rank`.
  Required when the campaign declares more than one view.
* `--round`: Round selector used with `--selected-rank` (default: `latest`).
* `--run-id`: Explicit run_id for ledger predictions (or `latest` to pick the latest ledger run by `(as_of_round, run_id)`).
* `--with-sequence/--no-sequence`: Include the sequence in output (default: on).
* `--json`: Output as JSON.

**Notes**

* `--selected-rank` cannot be combined with `<ID-or-SEQ>`, `--id`, or `--sequence`.
* If reruns exist for a round, pass `--run-id` to avoid mixing predictions.
* If the requested record does not exist, the command exits non-zero.

### `model-show`

Inspect a saved model; optionally dump full feature importances.

**Usage**

```bash
opal model-show \
  [--model-path <path> | --config <yaml> --round <k|latest>] \
  [--model-name <registry_name> --model-params <params.json>] \
  [--out-dir <dir>] [--json]
```

**Flags**

* `--model-path`: Path to `model.joblib` (overrides `--config/--round`, e.g. `outputs/rounds/round_<r>/model/model.joblib`).
* `--config, -c`: Path to `configs/campaign.yaml` (required if resolving from `state.json`).
* `--round, -r`: Round selector (integer or `latest`) to resolve model.
* `--model-name` / `--model-params`: Required if `model_meta.json` is missing. `--model-params` must be a `.json`.
* `--out-dir`: Write `feature_importance_full.csv` and print top-20 in JSON.
* `--json`: Output as machine-readable JSON (default output is plain text).

**Notes**

* When resolving from `state.json`, OPAL requires a recorded `model.artifact_path` for the selected round.
* OPAL fails fast if `model.artifact_path` is missing, relative, missing on disk, or points to a directory.

### `objective-meta`

List objective metadata and diagnostic keys for a round.

**Usage**

```bash
opal objective-meta --config <yaml-or-dir> --view <selection-view-id> \
  [--round <k|latest> | --run-id <id>] [--profile|--no-profile]
  [--json]
```

**Flags**

* `--config, -c`: Path to `configs/campaign.yaml` (required unless `$OPAL_CONFIG` is set). Directories are supported only for `opal progress`, `opal plot`, `opal notebook`, `opal review`, and `opal objective-meta`.
* `--round, -r`: Round selector (integer or `latest`; default: latest).
* `--run-id`: Explicit run_id to disambiguate when a round has multiple runs.
* `--view`: Required selection view ID.
* `--profile/--no-profile`: Profile candidate hue/size fields from the selected run.
* `--json`: Output as machine-readable JSON (default output is plain text).

**Notes**

* If multiple run_ids exist for the selected round, `--run-id` is required.

---

### `verify-outputs`

Compare selection artifacts against ledger predictions for a single run (run-aware, audit-grade).

**Usage**

```bash
opal verify-outputs --config <yaml> --view <selection-view-id> \
  [--round <k|latest> | --run-id <id>] \
  [--selection-path <path>] [--eps <float>] [--json]
```

**Notes**

* Resolves the selection artifact path from `outputs/ledger/runs.parquet` when possible.
* Projects the named view from `pred__selection_views` and compares its score,
  rank, and selected flag with `selections.parquet`.
* Fails fast with leakage-guard contract errors when ledger prediction IDs are
  duplicated for the selected run/round or when selected IDs are outside the
  run-scoped prediction/eval evidence.
* `--selection-path` accepts `.csv` or `.parquet`.
* `--round, -r`: Round selector (integer or `latest`).
* If the selected round has multiple runs, pass `--run-id` to disambiguate.
* Reads from `outputs/ledger/runs.parquet` and `outputs/ledger/predictions/`.
* `--json` writes JSON to stdout for optional redirection to a saved report.

---

### `selection-set`

Inspect or export the canonical selected-row contract for downstream consumers
such as study-owned synthesis handoffs or probe analysis.

**Usage**

```bash
opal selection-set show --config <yaml> --view <selection-view-id> \
  [--round <k|latest>] [--run-id <id>] [--json]
opal selection-set export --config <yaml> --view <selection-view-id> \
  [--round <k|latest>] [--run-id <id>] \
  --out <csv|json> [--format csv|json] [--json]
```

**Notes**

* Reads the shared campaign ledger, projects `--view`, and verifies the matching
  rows in `selections.parquet` before returning rows.
* Returns `schema_version: opal.selection_set.v2` for `show` and
  `opal.selection_set_export.v1` for `export`.
* Rows preserve OPAL candidate `id`, `sequence`, rank, score metadata,
  `run_id`, and `as_of_round`.
* If a round has multiple run IDs, pass `--run-id`; OPAL fails fast instead of
  guessing which rerun a downstream handoff should use.
* Missing ledgers or mismatched selection artifacts are structured JSON errors
  when `--json` is requested.

---

### `selection-batch`

Inspect or export the final deduplicated selection batch in a
run. This is an OPAL decision artifact, not a physical synthesis authorization.

```bash
opal selection-batch show --config <yaml> [--round <k|latest>] [--run-id <id>] [--json]
opal selection-batch export --config <yaml> [--round <k|latest>] [--run-id <id>] \
  --out <csv|json> [--format csv|json] [--json]
```

Rows retain `selection_view_ids` and per-view rank/score memberships. OPAL
fails when the batch artifact is absent or its deduplication key is not unique.

---

### `ctx`

Inspect `round_ctx.json` carriers.

**Usage**

```bash
opal ctx show  --config <yaml> [--round <k|latest>] [--keys <prefix> ...] [--json]
opal ctx audit --config <yaml> [--round <k|latest>] [--json]
opal ctx diff  --config <yaml> --round-a <k|latest> --round-b <k|latest> [--keys <prefix> ...] [--json]
```

**Flags**

* `--config, -c`: Path to `configs/campaign.yaml` (required unless `$OPAL_CONFIG` is set).
* `--round, -r`: Round selector for `show`/`audit`.
* `--round-a`, `--round-b`: Round selectors for `diff`.
* `--keys`: Filter by key prefix (repeatable; applies to `show`/`diff`).

**How to read ctx output**

* `ctx show`: raw key/value snapshot for one round.
* `ctx audit`: per-plugin contract audit (`consumed` / `produced` keys).
* `ctx diff`: key-level change summary between two rounds (`added`, `removed`, `changed`).
* Stage-scoped keys (for example model `predict` summaries) appear as final committed values after stage-end checks.

**Common checks**

```bash
# Model contract keys captured in the latest round
opal ctx show -c <yaml> --round latest --keys core/contracts/model

# Full per-plugin consumed/produced audit
opal ctx audit -c <yaml> --round latest

# What changed in objective/runtime keys between two rounds
opal ctx diff -c <yaml> --round-a 0 --round-b 1 --keys objective/
```

---

### `explain`

Dry-run planner for a round: counts, plan, warnings. **No writes.**

**Usage**

```bash
opal explain --config <yaml> --round <k>
  [--json]
```

**Flags**

* `--config, -c`: Path to `configs/campaign.yaml` (required unless `$OPAL_CONFIG` is set).
* `--round, -r`: Training cutoff.
* `--json`: Output as machine-readable JSON (default output is plain text).

Prints: number of training labels, candidate universe size, transforms/models/selection used,
vector dimension, and any preflight warnings.

---

### `status`

Dashboard from config, `records.parquet`, label source, and `state.json`.
Before initialization it still reports the records path, whether it exists, and
the configured label-source path.

For shared `usr_sidecar` campaigns, JSON output includes
`label_source.leakage` with schema `opal.leakage_guard.v1`. Status reads only
narrow label-status columns (`id`, configured Y when present, and
`opal__<slug>__label_hist` when present), never the configured X column.
Manifest-pinned sources also report `manifest_pinned`, `mutable`, manifest and
label digests, and the verified promoted row count.

**Usage**

```bash
opal status --config <yaml> [--round <k> | --all] [--with-ledger] [--json]
```

**Flags**

* `--config, -c`: Path to `configs/campaign.yaml` (required unless `$OPAL_CONFIG` is set).
* `--round`: Specific round details.
* `--all`: Dump every round (JSON output, even without `--json`).
* `--with-ledger`: Include ledger run_meta summaries in output.
* `--json`: Output as JSON.

### `runs`

List or inspect `run_meta` entries from `outputs/ledger/runs.parquet`.

**Usage**

```bash
opal runs list --config <yaml> [--round <k|latest>] [--json]
opal runs show --config <yaml> [--round <k|latest> | --run-id <rid>] [--json]
```

**Flags**

* `--config, -c`: Path to `configs/campaign.yaml` (required unless `$OPAL_CONFIG` is set).
* `--round, -r`: Round selector (integer or `latest`).
* `--run-id`: Explicit run_id to display (show only).
* `--json`: Output as machine-readable JSON (default output is plain text).

**Notes**

* `runs show --round <k>` requires `--run-id` if round `<k>` has multiple runs.
* JSON output is schema-bearing. `runs list --json` emits
  `schema_version: opal.runs_list.v1` with campaign metadata, requested and
  resolved round scope, and a `runs` array. `runs show --json` emits
  `schema_version: opal.run_meta.v1` with campaign metadata and a single `run`
  object.

---

### `log`

Summarize `round.log.jsonl` for a round.

**Usage**

```bash
opal log --config <yaml> [--round <k|latest>] [--json]
```

**Flags**

* `--config, -c`: Path to `configs/campaign.yaml` (required unless `$OPAL_CONFIG` is set).
* `--round, -r`: Round selector (integer or `latest`).
* `--json`: Output as machine-readable JSON (default output is plain text).

**Notes**

* If a round has been re-run (multiple `start` events in the same log), the summary focuses on the **latest run**.

---

### `progress`

Summarize campaign progress from `state.json` and round logs.

**Usage**

```bash
opal progress --config <yaml-or-dir> [--round <k|latest|all>] [--run-id <id>] [--json]
```

**Flags**

* `--config, -c`: Path to `configs/campaign.yaml` or a campaign directory (required unless `$OPAL_CONFIG` is set).
* `--round, -r`: Round selector (`latest`, `all`, or an integer).
* `--run-id`: Filter round-log summaries to one run when a round has reruns.
* `--json`: Output as machine-readable JSON (default output is plain text).

**Notes**

* This is the campaign-generic progress surface for operators and harnesses.
* The JSON payload uses schema `opal.campaign_progress.v1` and reports per-round
  status, last stage, elapsed seconds, prediction batch count, log path,
  summarized stage counts, run scope, event phase counts, active lock state,
  warnings, and stale review artifacts.
* Round-log events written by current OPAL versions include schema
  `opal.progress_event.v1`, an event ID, a phase (`command`, `preflight`,
  `run`, `abort`, or `finalize`), and severity. Events that do not satisfy this
  contract are rejected.
* Study probes and campaign dashboards should consume this primitive instead of
  parsing OPAL round-log paths directly.

### `validate`

End-to-end table checks (essentials present; X non-null, finite, and fixed-length).

**Usage**

```bash
opal validate --config <yaml> [--json]
```

**Flags**

* `--config, -c`: Path to `configs/campaign.yaml` (required unless `$OPAL_CONFIG` is set).
* `--json`: Output schema `opal.validate.v1` as machine-readable JSON (default output is plain text).

**Notes**

* Config parsing is strict: malformed YAML or duplicate keys fail as bad-args config errors.
* Verifies **USR essentials** exist in `records.parquet`.
* Verifies the configured **X** column exists.
* If Y is present, validates vector length & numeric/finite cells.
* JSON errors use the shared `opal.cli_error.v1` payload, so scripts can rely
  on parseable stdout for both pass and fail paths.

---

### `label-hist`

Validate, repair, or explicitly attach-from-y into the label history column (no silent fixes).

**Usage**

```bash
opal label-hist <validate|repair|attach-from-y> --config <yaml> [--apply] [--round <int>] [--json]
```

**Flags**

* `--config, -c`: Path to `configs/campaign.yaml` (required unless `$OPAL_CONFIG` is set).
* `<validate|repair|attach-from-y>`: Action (alias: `check` = `validate`).
* `--apply`: Apply changes for `repair` or `attach-from-y` (default: dry-run).
* `--round, -r`: Required for `attach-from-y`; round stamp to attach.
* `--src`: Optional label_hist source tag for `attach-from-y` (default: `manual_attach`).
* `--json`: Output as machine-readable JSON (default output is plain text).

**Notes**

* `attach-from-y` is a **manual** fix for datasets with a populated Y column but empty label history.
  It only attaches entries for rows where `label_hist` is empty and Y is finite.

### Records label history

OPAL manages a primary per‑record label history column in `records.parquet`:

* `opal__<slug>__label_hist` — append‑only history of observed labels and run‑aware predictions.

`opal init` will add the label history column if it is missing.

---

### `plot`

Generate plots declared in the campaign’s `plots:` block. Plots are plugin-driven and campaign-scoped.

**Usage**

```bash
opal plot --config <yaml-or-dir> [--plot-config <plots.yaml>] \
  [--round <selector>] [--run-id <id>] [--view <selection-view-id>] \
  [--name <plot-name>] [--tag <tag> ...]
opal plot --list
opal plot --list-config --config <yaml-or-dir>
opal plot --describe <plot-kind>
opal plot --list --json
opal plot --list-config --config <yaml-or-dir> --json
opal plot --describe <plot-kind> --json
```

**Flags**

* `--config, -c`: Campaign YAML or campaign directory.
* `--plot-config`: Path to a plots YAML (overrides `plot_config` in configs/campaign.yaml).
* `--list`: List registered plot kinds and exit (does not require config).
* `--list-config`: List plots configured in YAML and exit (requires `--config`).
* `--describe`: Show parameters + required fields for a plot kind.
* `--round, -r`: `latest | all | 3 | 1,3,7 | 2-5` (plugin may define defaults).
* `--run-id`: Explicit run_id to disambiguate ledger predictions (required if multiple runs exist for a round).
* `--view`: Selection view ID; required for multi-view campaign plots.
* `--name, -n`: Run a single plot by name; omit to run **all**.
* `--tag`: Run plots with the given tag (repeatable).
* `--json`: Emit machine-readable JSON for `--list`, `--list-config`, and
  `--describe`.

**Notes**

* Overwrites files by default; continues on error; exit code **1** if any plot failed.
* `plot --list-config --json` emits structured plot objects with `name`,
  `kind`, `enabled`, `tags`, and optional `preset`; text output keeps the
  compact `name: kind (enabled)` listing.
* Output directory defaults to `outputs/plots`, or honors `output.dir` if provided.
* Every plot run writes a per-plot manifest next to the rendered media and an
  aggregate `plot_manifest.json` index in the output directory. Manifests record
  plot kind, params, run/round scope, inputs, media outputs, tidy CSV outputs,
  status, generated time, schema version, warnings, and errors.
* `output.save_data: true` asks plot plugins to write tidy CSV data; the plot
  manifest records the CSV paths that were produced.
* Plot-specific knobs **must** live under `params:`; top-level plotting keys are errors.
* Prefer `plot_config: plots.yaml` in configs/campaign.yaml to keep runtime config lean.
* `plot_defaults` and `plot_presets` reduce redundancy; `preset: <name>` merges into each plot entry.
* Set `enabled: false` on any plot entry to keep it in the YAML without running it.
* If a round has multiple run_ids, plots require `--run-id` to avoid mixing reruns.
* If `--run-id` is provided, OPAL resolves its round from `outputs/ledger/runs.parquet`; `--round all` is invalid and conflicting `--round` values error.

**Campaign YAML (example)**

```yaml
plot_config: plots.yaml
```

**plots.yaml (example)**

```yaml
plot_defaults:
  output:
    format: "png"                       # png/svg/pdf
    dpi: 600

plots:
  - name: score_vs_rank_latest
    kind: scatter_score_vs_rank         # plot plugin id
    params:
      score_field: "view__selection_score" # projected field for --view
      hue: null                         # or "round"
      highlight_selected: false
    output:
      dir: "{campaign}/plots/{kind}/{name}"  # {campaign|workdir|kind|name|round_suffix}
      filename: "{name}{round_suffix}.png"
```

**Data sources**
Plot plugins typically read from the campaign’s **ledger sinks** under `outputs/` and/or **`records.parquet`**.
Additional sources may be declared per plot entry:

```yaml
data:
  - name: extra_csv
    path: ./extras/scores.csv
```

Built-ins injected for plots:

* `records`
* `outputs`
* `ledger_predictions_dir`
* `ledger_runs_parquet`
* `ledger_labels_parquet`

---

### `notebook`

Generate or run OPAL marimo artifact viewers.

**Usage**

```bash
uv run opal notebook
uv run opal notebook generate --config <yaml-or-dir> [--round <latest|k>] [--run-id <id>] [--out <path>] [--name <file>] [--force] [--validate/--no-validate] [--json]
uv run opal notebook generate --campaign <yaml-or-dir> --campaign <yaml-or-dir> [--config <anchor-yaml-or-dir>] [--out <path>] [--round <latest|k>] [--json]
uv run opal notebook run --config <yaml-or-dir> [--path <notebook.py>] [--host <host>] [--port <port>] [--headless]
uv run opal notebook edit --config <yaml-or-dir> [--path <notebook.py>] [--host <host>] [--port <port>] [--headless]
```

**Notes**

* `generate` writes the campaign-specific artifact viewer for record contract
  status, round/run state, ledger readiness, label and prediction summaries,
  selection summaries, and manifest-backed `outputs/plots` deliverables.
* Repeating `--campaign` writes an explicit campaign-set notebook with a
  campaign dropdown, at-a-glance campaign table, selected-campaign status, plot
  deliverable dropdowns, selected plot artifacts at the top, and
  warnings/stale-artifact panels behind supporting detail sections. This is a
  review surface over OPAL campaign contracts, not a study/probe dashboard.
* `generate` requires the campaign `records.parquet` to exist so the notebook
  can inspect schema and identity columns. The generated preview is
  schema-pruned and does not load the configured X payload on startup.
* `generate` works before the first OPAL run. Missing ledger, label,
  prediction, and plot artifacts appear as explicit notebook states.
* `generate --json` emits schema `opal.notebook_generate.v1` with the written
  notebook path, config paths, round selector, optional pinned run ID, and
  follow-up commands. JSON output has no human preamble.
* `generate --run-id <id>` is single-campaign only. It resolves the run ID
  through `outputs/ledger/runs.parquet`, pins the generated default round to
  that run's `as_of_round`, and rejects mismatched `--round` values.
* Generated notebooks import public helpers from `dnadesign.opal`, build a
  `NotebookViewModel`, and render plot cards from review and plot manifests
  rather than treating directory contents as authoritative.
* The notebook surface is campaign review only: records contract, X provenance,
  ledgers, progress, selections, labels, predictions, plots, and limitations.
  Representation browsers, UMAP atlases, and study/probe-specific visuals live
  outside canonical OPAL notebooks.
* When `--validate` is on and ledger runs already exist, `--round` must resolve
  in those runs. `--no-validate` skips that round check.
* `run` launches `marimo run` app mode for review. Use `--host`,
  `--port`, and `--headless` for local dogfood or automation.
* `edit` launches `marimo edit` for notebook authoring. Do not use edit mode
  as review evidence when app behavior is the question.
* `run` resolves the notebook under `<workdir>/notebooks`. If multiple exist, it prompts in TTY or requires `--path` in non-interactive mode.
* Running `uv run opal notebook` (no subcommand) lists available notebooks and nudges the next step.
* Checked-in notebook fixtures are maintainer/test surfaces. Operator-facing
  campaign notebooks should come from `opal notebook generate`.

---

### `review`

Persist a campaign-scoped review bundle from completed OPAL run artifacts.

**Usage**

```bash
opal review --config <yaml-or-dir> --view <selection-view-id> \
  [--round <latest|k>] [--run-id <id>] [--out-dir <dir>] [--plots/--no-plots] [--json]
```

**Notes**

* Writes the view-scoped bundle under
  `outputs/review/selection_views/<view>/` by default.
* Reads campaign ledgers and per-round artifacts; it does not rerun OPAL and
  does not mutate records or labels.
* Fails fast if run-scoped prediction evidence contains duplicate candidate IDs.
* If a round has multiple run IDs, pass `--run-id` so the review does not mix
  reruns.
* The manifest is authoritative. Review reports stale files that exist under
  the review output directory but are absent from the active manifest, and it
  validates the configured X column before publishing review evidence.

---

### `artifacts`

Audit and prune generated OPAL artifacts using active manifests as the
authority.

**Usage**

```bash
opal artifacts audit --config <yaml-or-dir> [--json]
opal artifacts prune --config <yaml-or-dir> [--apply] [--json]
```

**Notes**

* `audit` is read-only. It inventories `outputs/`, `notebooks/`, active review
  and plot manifests, stale manifest-absent plot/review siblings, byte counts,
  and whether the campaign root is local-only because it lives under `.var`.
* `prune` is a dry-run by default. It deletes only stale artifacts from the
  active prune plan when `--apply` is passed.
* The command does not read `records.parquet` or the configured X column; it is
  an artifact-gardening surface, not a campaign execution or scientific-review
  step.
* JSON errors use the shared `opal.cli_error.v1` envelope with context,
  category, message, and exit code.

---

### `campaign-reset` (advanced)

Reset a campaign to a clean slate: prune OPAL-derived columns from `records.parquet`, remove `outputs/`, remove `notebooks/`, and clear `state.json`.

**Usage**

```bash
opal campaign-reset --config <yaml> [--apply] [--backup|--no-backup] [--json]
```

**Flags**

* `--config, -c`: Path to `configs/campaign.yaml` (required unless `$OPAL_CONFIG` is set).
* `--apply`: Apply reset without interactive confirmation.
* `--backup/--no-backup`: Backup `records.parquet` before pruning (default: no-backup).
* `--json`: Output as machine-readable JSON (default output is plain text).

**Notes**

* This command is hidden from top-level `opal --help` because it is destructive.
* Uses slug confirmation in interactive mode (or `--apply` in non-interactive mode).

---

### `ledger-compact`

Compact ledger datasets after repeated append cycles.

**Usage**

```bash
opal ledger-compact --config <yaml> --runs [--apply] [--json]
```

**Flags**

* `--config, -c`: Path to `configs/campaign.yaml` (required unless `$OPAL_CONFIG` is set).
* `--runs`: Compact `outputs/ledger/runs.parquet` (required; command exits if omitted).
* `--apply`: Apply compaction without interactive confirmation.
* `--json`: Output as machine-readable JSON (default output is plain text).

**Notes**

* Rewrites ledger datasets in place and should be run when no other OPAL process is writing.

---

### `prune-source`

Remove OPAL-derived columns (`opal__*`) and the configured Y column from `records.parquet`.

**Usage**

```bash
opal prune-source --config <yaml> [--scope any|campaign] [--keep <col> ...] \
  [--apply] [--backup|--no-backup] [--json]
```

**Flags**

* `--config, -c`: Path to `configs/campaign.yaml` (required unless `$OPAL_CONFIG` is set).
* `--scope`: Which opal namespaces to prune: `any` (default) or `campaign` (this campaign’s slug only).
* `--keep, -k`: Column name(s) to keep even if matched for deletion (repeatable).
* `--apply`: Apply prune without interactive confirmation.
* `--backup/--no-backup`: Backup original file before pruning (default: on).
* `--json`: Output as machine-readable JSON (default output is plain text).

**Notes**

* Designed as a **start fresh** option before re-running round 0.

---

### Config resolution

Campaign-scoped commands require explicit config context:

1. **Explicit flag** (`--config`)
2. **Environment variable** (`OPAL_CONFIG`)

Passing a **directory** to `--config` is supported only for `opal progress`, `opal plot`, `opal notebook`, `opal review`, and `opal objective-meta`.
Other campaign-scoped commands require a YAML config path.
`plot_config` paths are resolved relative to the `configs/campaign.yaml` that declares them.
For scripts and CI, pass `--config` explicitly.

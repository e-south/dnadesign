## Shared Label Source Plan

**Owner:** dnadesign-maintainers
**Status:** implementation in progress
**Last verified:** 2026-05-17

This plan tracks the OPAL label-source change needed for multi-campaign active
learning where campaigns share one candidate universe and one observed-label
pool, but keep objective, model, scoring, and selection state campaign-local.

### Intent

Implement a shared OPAL label-source contract so each campaign can train on all
observed `Y` labels available through a given round, while `top_k`, objective,
and selection behavior remain campaign/selector-specific.

### Worth-Doing Preflight

Best case: one shared candidate/X data product, one durable observed-label
ledger, and many lightweight campaigns. This removes duplicate ingest, prevents
campaign label drift, and keeps ethanol/ciprofloxacin/AND or future OPAL
campaigns extensible without multiplying USR datasets or coupling sibling
tools through private internals.

### Scope

In scope:

- OPAL label-source abstraction.
- Shared observed-label storage for USR-backed campaigns.
- `ingest-y`, `run`, `validate`, and `status` behavior.
- Study configs/docs for stress ethanol/ciprofloxacin/AND.
- Tests proving shared labels plus campaign-specific selection.

Out of scope:

- Changing LatentDNA `X` materialization.
- Rewriting model, objective, or selection plugins.
- Adding a new orchestration layer.
- Moving campaign-specific predictions/selections into USR as primary truth.

### Target Contract

- `records.parquet`: stable candidate identity, sequence, provenance, and
  shared `X`.
- Shared label source: append-only observed assay labels, preferably a
  USR-adjacent sidecar under the candidate dataset.
- OPAL campaign workdir: campaign state, model outputs, scores, selections,
  notebooks, and ledgers.
- Study docs: route map and batch semantics, not primary runtime storage.

Target configuration shape:

```yaml
labels:
  source:
    kind: usr_sidecar
    dataset: usr_prom_eth_cip_opal_candidates
    path: _opal/observed_labels.parquet
  y_space: sfxi_vec8
  id_column: id
  round_column: observed_round
  batch_column: batch_id
  dedup_policy: latest_by_round
writeback:
  prediction_records: ledger_only
```

`selection.params.top_k`, `score_ref`, `objective_mode`, and selector plugin
params stay in each campaign config.

### Ordered Checklist

1. Add failing tests first:
   - two campaign configs share one label source and train on the same labels
     through `--labels-as-of`.
   - same labels produce campaign-specific objective/selection output because
     setpoints differ.
   - `top_k` remains selector-specific per campaign.
   - candidate exclusion uses shared observed labels, not slug-local label
     history.
   - missing/malformed label source fails fast.
2. Add `ObservedLabelStore` / `TrainingLabelSource`:
   - append labels as rows, not JSON-in-record cells.
   - required fields: `id`, `observed_round`, `batch_id`, `y_obs`, `y_space`,
     schema/state-order metadata, `src`, and timestamp.
   - provide `training_labels(as_of_round, policy)` and
     `observed_ids(as_of_round)`.
3. Refactor OPAL training path:
   - keep `RecordsStore` responsible for candidates and `X`.
   - move label lookup out of `RecordsStore.training_labels_with_round`.
   - make `plan_round` receive training labels and observed-id sets through a
     label-source seam.
4. Update ingest:
   - `opal ingest-y` writes once to the shared label source when
     `labels.source.kind: usr_sidecar`.
   - keep campaign-local label history as an explicit campaign-scoped mode, not an
     implicit fallback from shared labels.
   - default unknown IDs should fail or drop by explicit flag; no silent row
     creation for fixed shared candidate universes.
5. Update run/writeback:
   - shared-label campaigns train from `ObservedLabelStore`.
   - disable full prediction writeback to shared `records.parquet` by default.
   - keep predictions, scores, selections, and run metadata in campaign ledgers.
6. Update configs/docs:
   - ethanol/ciprofloxacin/AND configs point to the same label source.
   - remove durable meaning from per-campaign `label_hist_column` in study batch
     docs.
   - document that batches aggregate labels by
     `observed_round <= labels_as_of`; selectors decide `top_k`.
7. Add validation:
   - `opal validate` checks label-source schema, ID compatibility with the
     candidate table, `y_expected_length`, and `y_space`.
   - status output reports label-source kind, available rounds, label counts,
     and selected candidate counts.

### Sprint Contract

Current implementation state:

- The `labels` config block, `ObservedLabelStore`, shared run/explain training,
  shared `ingest-y`, candidate exclusion, stress configs, ledger-only
  prediction writeback, local sidecar path locking, and targeted CLI
  status/validate visibility are implemented.
- Shared-label configs require explicit `writeback.prediction_records`; the
  stress campaigns use `ledger_only` so `records.parquet` remains the
  candidate/X table during `run`.
- `ingest-y` for shared sidecars rejects unknown IDs unless they are explicitly
  dropped; it does not create candidate rows through the label path.
- Shared sidecar appends are serialized with a local path lock around the
  load/merge/write critical section. This is a single-host operator lock, not a
  distributed filesystem lease.

First implementation slice:

- Add the `labels` config block, `ObservedLabelStore`, and unit tests.
- Wire `run` to read shared labels, but leave `ingest-y` campaign-history
  behavior untouched until the second slice.
- Done when two temp campaigns with different setpoints can train from the same
  shared label table and produce independent selection ledgers.

Second slice:

- Wire `ingest-y` to the shared label source.
- Add shared observed-id candidate exclusion.
- Update stress configs/docs.

Third slice:

- Turn off shared-record prediction writeback for shared-label campaigns.
- Add migration docs and cleanup safety checks.

Current third-slice status: `ledger_only` prediction writeback and sidecar path
locking are implemented for shared-label stress configs; broader
migration/cleanup safety checks remain as future hardening.

### Validation

- targeted OPAL label-source unit tests.
- integration test with two campaigns, same labels, different setpoints/top-k.
- stress batch-0 tests.
- `uv run python -m dnadesign.devtools.docs.checks`
- `uv run ruff check ...`
- targeted `opal validate`, `opal ingest-y`, `opal run`, and `opal status` on
  temp copied data.

### Risk Handling

The main risk is accidental backcompat ambiguity. Handle it by making label
source mode explicit: `campaign_history` for campaign-scoped labels and
`usr_sidecar` for shared-label campaigns. Do not silently fall back from a
configured shared label source to slug-local label history.

Concurrency risk is intentionally scoped. `ObservedLabelStore` protects
shared-label writes with a local path lock, which is suitable for sequential
operator workflows and concurrent local campaign runs. If shared ingest runs on
multiple hosts against a network filesystem, promote this to a lease/transaction
primitive before treating parallel writes as supported.

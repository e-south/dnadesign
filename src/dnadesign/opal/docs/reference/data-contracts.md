# OPAL Data Contracts

## Safety and validation

OPAL is assertive by default: it fails fast on inconsistent inputs.

- `opal validate` checks essentials + X presence; if Y exists it must be finite and expected length.
- `label_hist` is required input for `run`/`explain` and the canonical dashboard source;
  `outputs/ledger/labels.parquet` remains the audit sink.
- Labels in Y but missing from `label_hist` are rejected
  (use `opal ingest-y` or `opal label-hist attach-from-y` for legacy Y columns).
- Ledger writes are strict: unknown columns are errors (override only with `OPAL_LEDGER_ALLOW_EXTRA=1`).
- Duplicate handling on ingest is explicit via `ingest.duplicate_policy` (`error|keep_first|keep_last`).

## Records schema

Required columns in `records.parquet`:

| column | type | notes |
| --- | --- | --- |
| `id` | string | unique per record |
| `bio_type` | string | `"dna"` or `"protein"` |
| `sequence` | string | case-insensitive |
| `alphabet` | string | e.g. `dna_4` |

Recommended/common columns (not enforced by `validate`):

| column | type | notes |
| --- | --- | --- |
| `length` | int32 | `len(sequence)` |
| `source` | string | provenance |
| `created_at` | timestamp (UTC) | ingest time |

X and Y representation:

- X: Arrow `list<float>` or JSON array string; fixed length across used rows
- Y: Arrow `list<float>`; label history stored in `opal__<campaign>__label_hist`

Naming convention: secondary columns use `<tool>__<field>`.

### Records label history (OPAL-managed)

| column | type | purpose |
| --- | --- | --- |
| `opal__<slug>__label_hist` | list<struct> | Append-only per-record history of observed labels and run-aware predictions (dashboard canonical). |

Label history entry shapes:

- Observed label: `{kind:"label", observed_round:int, ts:str, src:str, y_obs:{value:<json>, dtype:str, schema?:{...}}}`
- Prediction/scoring: `{kind:"pred", as_of_round:int, run_id:str, ts:str, y_pred:{value:<json>, dtype:str, schema?:{...}}, y_space:str, objective:{name,params}, metrics:{score,logic_fidelity,effect_scaled,...}, selection:{rank,top_k}}`

`opal init` ensures label history exists in `records.parquet`.
Use `opal prune-source` to remove OPAL-derived columns (including Y) for clean restarts.

## Canonical vs ledger vs overlay (notebook)

- Canonical (dashboard): `records.parquet` label history + campaign artifacts/state
- Ledger (audit): append-only run metadata and predictions under `outputs/ledger/`
- Overlay (notebook): in-memory rescoring from stored predictions for exploration only
- Y-ops gating: notebook SFXI scoring runs only when predictions are objective-space

## Ledger output schema (append-only)

Ledger sinks are append-only audit records.

### labels (`outputs/ledger/labels.parquet`)

- `event`: `"label"`
- `observed_round`, `id`, `sequence` (if available)
- `y_obs`: `list<float>`
- `src`, `note`

### run_pred (`outputs/ledger/predictions/`)

- `event`: `"run_pred"`, plus `run_id`, `as_of_round`, `id`, `sequence`
- `pred__y_dim`, `pred__y_hat_model` (objective-space), `pred__y_obj_scalar`
- `sel__rank_competition`, `sel__is_selected`
- Optional row diagnostics under `obj__*`

### run_meta (`outputs/ledger/runs.parquet`)

- `event`: `"run_meta"`, plus `run_id`, `as_of_round`
- Config snapshot: `model__*`, `x_transform__*`, `y_ingest__*`, `objective__*`, `selection__*`, `training__y_ops`
- Counts + summaries: `stats__*`, `objective__summary_stats`, `objective__denom_*`
- Provenance: `artifacts` (paths + hashes), `schema__version`, `opal__version`

Design notes:

- Keep row-level diagnostics in `run_pred`, run-level summaries in `run_meta`.
- Prefer adding new columns over changing semantics; keep prefix conventions explicit.
- Treat `schema__version` as compatibility guardrail for evolution.

## OPAL — Optimization with Active Learning

**OPAL** is an [EVOLVEpro-style](https://www.science.org/doi/10.1126/science.adr6006) active-learning engine for biological sequence design.

- **Train** a top-layer regressor on your chosen representation **X** and label **Y**.
- **Score** every candidate sample with **X** present.
- **Reduce** each prediction Ŷ to a scalar **score** via your configured **objective** → `pred__y_obj_scalar`.
- **Rank** candidates by that score and **select top-k**.
- **Append** runtime events to **ledger sinks** under **`outputs/`** (per-round predictions + run metadata).
- **Persist** artifacts per round (model, selection CSV, round context, objective meta, logs) for auditability.

The pipeline is plugin-driven: swap **data transforms** (X/Y), **models**, **objectives**, and **selection** strategies in `campaign.yaml` without touching core code.

> Using OPAL day-to-day? See the **[CLI Manual](./docs/cli.md)**

---

### Contents

* [Quick start](#quick-start)
* [How OPAL is wired](#how-opal-is-wired)
  * [Code layout](#code-layout)
* [Campaign layout](#campaign-layout)
* [Configuration (`campaign.yaml`)](#configuration)
  * [Key blocks](#key-blocks)
  * [Defaults](#defaults)
  * [Minimal example](#minimal-example)
  * [Notes on precedence & wiring](#notes-on-precedence--wiring)
* [Architecture & data flow](#architecture--data-flow)
  * [RoundCtx (runtime carrier)](#roundctx-runtime-carrier)
* [Safety & validation](#safety--validation)
* [Data contracts](#data-contracts)
  * [Records schema](#records-schema)
  * [Ledger output schema](#ledger-output-schema-append-only)
* [More documentation](#more-documentation)
* [Demo campaign](#demo-campaign)


---

### Quick start

OPAL lives inside the `dnadesign` repo. The CLI entrypoint is defined in `pyproject.toml`:

```toml
[project.scripts]
opal = "dnadesign.opal.src.cli:main"
```

```bash
# 1) Initialize a campaign workspace (creates inputs/, outputs/, state.json, and .opal/config marker)
opal init -c path/to/campaign.yaml

# 2) Validate essentials + X + (if present) Y shape/values
opal validate -c path/to/campaign.yaml

# 3) Ingest experimental labels (CSV/Parquet → Y), preview, then write new columns to X dataset
opal ingest-y -c path/to/campaign.yaml --round 0 --csv path/to/my_labels.csv

# 4) Train on labels with observed_round ≤ R, score the pool, select top-k, persist artifacts & events
opal run -c path/to/campaign.yaml --round 0

# 5) Inspect progress
opal status -c path/to/campaign.yaml
opal status -c path/to/campaign.yaml --with-ledger
opal runs list -c path/to/campaign.yaml
opal log -c path/to/campaign.yaml --round latest
opal record-show -c path/to/campaign.yaml --id <some_id>
opal explain -c path/to/campaign.yaml --round 1
opal plot --list                              # list available plot kinds
opal plot -c path/to/campaign.yaml
opal plot -c path/to/campaign.yaml --quick    # run default plots without plots.yaml
opal plot -c path/to/campaign.yaml --run-id <run_id>  # run-aware; resolves round, conflicts error
opal predict -c path/to/campaign.yaml               # uses latest round
opal objective-meta -c campaign.yaml --round latest
opal verify-outputs -c campaign.yaml --round latest
opal notebook generate -c path/to/campaign.yaml     # create a marimo analysis notebook
opal notebook generate -c path/to/campaign.yaml --no-validate  # scaffold before any runs

# (Optional) Start fresh: remove OPAL-derived columns from records.parquet
opal prune-source -c path/to/campaign.yaml --scope campaign

```

**Round terminology**

- **observed_round**: the round stamp recorded when labels are ingested (`opal ingest-y`).
- **labels-as-of**: the training cutoff used by `opal run`/`opal explain` (uses labels with `observed_round ≤ R`).

### What gets saved

* **Per-round**
  - `outputs/round_<k>/`
  - `model.joblib`
  - `model_meta.json` *(includes training__y_ops when configured)*
  - `selection_top_k.csv`
  - `selection_top_k.parquet`
  - `selection_top_k__run_<run_id>.csv` *(immutable per-run copy)*
  - `selection_top_k__run_<run_id>.parquet` *(immutable per-run copy)*
  - `labels_used.parquet`
  - `round_ctx.json` *(runtime audit & fitted Y-ops)*
  - `objective_meta.json` *(objective mode/params/keys)*
  - `round.log.jsonl`

* **Campaign-wide ledger (append-only)**

  * `outputs/ledger.runs.parquet`
    - Plugin configs, counts, objective summaries, artifact hashes, versions.
  * `outputs/ledger.predictions/`
    - Ŷ vector, scalar score, selection rank/flag, and row-level diagnostics (e.g., logic fidelity/effects).
    - `pred__y_hat_model` is in objective-space.
  * `outputs/ledger.labels.parquet`
    - 1 row per label event (observed round, id, y).

Schemas are **append-only**; uniqueness is enforced for:
  `run_id` (runs), `(run_id,id)` (predictions). Labels are event rows; exact duplicates are de‑duplicated,
  but distinct sources for the same `(observed_round,id)` are preserved.

**state.json** tracks campaign state per round, including `run_id` and `round_log_jsonl` paths for auditability.

---

## How OPAL is wired

At the top level, OPAL is a CLI (Typer) over an application layer with plugin registries.

### Code layout

```
src/dnadesign/opal/src/
├─ cli/                     # CLI app + command registry
│  ├─ app.py                # Typer entrypoint
│  ├─ registry.py           # auto-discovers and mounts commands
│  └─ commands/             # run, ingest_y, predict, explain, record_show, status, runs, log, validate, init, plot, prune_source
├─ config/                  # YAML loader + plugin param schemas (Pydantic)
├─ registries/              # transforms_x, transforms_y, models, objectives, selection, plots
├─ transforms_x/            # X transforms (import = register)
├─ transforms_y/            # Y ingests (import = register)
├─ models/                  # model wrappers (e.g., RandomForest)
├─ objectives/              # objective fns (Ŷ → scalar score + diagnostics)
├─ selection/               # selection strategies (scores → ranks/selected)
├─ plots/                   # plot plugins
├─ core/                    # RoundCtx, console helpers, core utils/errors
├─ runtime/                 # run_round, ingest, predict, explain, preflight, round_plan
├─ storage/                 # data_access, ledger, artifacts, writebacks, workspace, state, locks
├─ reporting/               # status, summary, record_show
├─ analysis/                # analysis utilities (dashboard modules under analysis/dashboard)
└─ …
```

Dashboard notebooks (e.g., `prom60_eda.py`) now pull their shared logic from
`src/dnadesign/opal/src/analysis/dashboard/` to keep data contracts explicit and reusable for future dashboards.

---

### Campaign layout

`opal init` scaffolds a campaign folder and ensures OPAL cache columns exist in `records.parquet`. Edit
`campaign.yaml` to define behavior.

```
<repo>/src/dnadesign/opal/campaigns/<my_campaign>/
├─ campaign.yaml
├─ plots.yaml                    # optional plot config (recommended)
├─ .opal/
│  └─ config                     # auto-discovery marker (path to campaign.yaml)
├─ state.json
├─ inputs/                       # drop experimental label files here
└─ outputs/
   ├─ ledger.predictions/        # append-only run_pred parts
   ├─ ledger.runs.parquet        # run_meta (deduped)
   ├─ ledger.labels.parquet      # label events (ingest-only)
   └─ round_<k>/
      ├─ model.joblib
      ├─ model_meta.json
      ├─ selection_top_k.csv
      ├─ labels_used.parquet
      ├─ round_ctx.json
      ├─ objective_meta.json
      └─ round.log.jsonl
```

> **Note:** sequences may live in USR datasets under `src/dnadesign/usr/datasets/<dataset>/records.parquet` and are not copied into campaigns.

---

### Configuration

OPAL reads a configuration YAML, `campaign.yaml`.

#### Key blocks

* `campaign`: `name`, `slug`, `workdir`
* `data`: `location`, `x_column_name`, `y_column_name`, `y_expected_length`
* `transforms_x`: `{ name, params }` (raw X → model-ready X)
* `transforms_y`: `{ name, params }` (CSV → model-ready Y)
* `model`: `{ name, params }`
* `objective`: `{ name, params }`
* `selection`: `{ name, params }` *(strategy, tie handling, objective mode)*
* `training`: `policy`
* `training.y_ops`: list of `{ name, params }` (bespoke transforms applied at train and/or predict time)
* `ingest`: duplicate handling for label CSVs
* `scoring`: batch sizing
* `safety`: preflight/data guards
* `metadata`: optional notes
* `plot_config`: optional path to a separate plots YAML (recommended)

#### Defaults

If a block is omitted, OPAL supplies conservative defaults:

* `ingest.duplicate_policy`: `error`
* `scoring.score_batch_size`: `10000`
* `training.policy`: `{}` and `training.y_ops`: `[]`
* `safety`: fail_on_mixed_biotype_or_alphabet=true, require_biotype_and_alphabet_on_init=true,
  conflict_policy_on_duplicate_ids=error, write_back_requires_columns_present=true, accept_x_mismatch=false
* `metadata.notes`: `""`
* `plots`: `[]` (no plotting unless declared)

Plugin `params` default to `{}`, but **plugin names are required**.

#### Minimal example

```yaml
campaign:
  name: "My Campaign"
  slug: "my_campaign"
  workdir: "src/dnadesign/opal/campaigns/my_campaign_dir"

data:
  location: { kind: usr, path: src/dnadesign/usr/datasets, dataset: my_dataset }
  x_column_name: "my_x_column"
  y_column_name: "my_y_column"
  y_expected_length: 4   # enforce Y length on validate/run

ingest:
  duplicate_policy: "error"

transforms_x: { name: my_x_preprocessing_prior_to_model, params: {} }
transforms_y: { name: my_y_preprocessing_prior_to_model, params: {} }

model:
  name: random_forest
  params: { ... }

training:
  policy:
    cumulative_training: true
    label_cross_round_deduplication_policy: "latest_only"
    allow_resuggesting_candidates_until_labeled: true

objective:
  name: my_objective
  params: { ... }

selection:
  name: top_n
  params: { ... }

scoring:
  score_batch_size: 10000

safety:
  fail_on_mixed_biotype_or_alphabet: true
```

---

#### Notes on precedence & wiring

* Paths in `campaign.yaml` resolve **relative to the YAML file** (including `campaign.workdir`);
  prefer `workdir: "."` for portability.
* CLI flags override YAML **for that invocation**:
  `run --k` overrides `selection.params.top_k`, `run --score-batch-size` overrides `scoring.score_batch_size`,
  and `ingest-y --transform/--params` (JSON file, `.json`) overrides `transforms_y`.
* `--round` is the canonical flag; `--labels-as-of` and `--observed-round` are aliases.
* `transforms_y` is used for **ingest only**; model training/prediction uses `transforms_x` plus optional `training.y_ops`.
* `state.json` records the resolved config per round; ledger sinks are the long‑term source of truth.
* `plot_config` paths resolve **relative to the campaign.yaml** that declares them.

---

### Architecture & data flow

* Models are not aware of downstream objectives (see **RoundCtx (runtime carrier)**)
* Objectives derive their own round constants via `train_view` and publish them.
* Selection can read whatever objectives produced.
* The persisted `round_ctx.json` makes runs **auditable** alongside ledger sinks, `model.joblib`, `model_meta.json`, `selection_top_k.csv`, and `objective_meta.json`.

```bash
# Labels
my_experimental_data.csv -> transforms_y -> labels [id, y(list<float>)] -> appends event: label

# Features
records.parquet [X] -> transforms_x -> X (fixed width)

# Train & score
X + labels -> model.fit -> predict Ŷ -> objective -> pred__y_obj_scalar -> selection (top-k)

# Canonical ledger sinks
outputs/ledger.*: { label | run_pred | run_meta }
```

---

#### RoundCtx (runtime carrier)

**RoundCtx** is a runtime companion to `campaign.yaml`: the YAML picks which plugins to run; the carrier records what the run computed that others depend on. The runner persists a compact **`round.log.jsonl`** with stage events (preflight, y‑ops fit/transform, fit start/done, predict batches, y‑ops inverse, objective, selection) and a full **`round_ctx.json`** snapshot for audit.

#### How plugins use `RoundCtx`

The runner injects a **plugin-scoped context** (`ctx`) that auto-expands `"<self>"` to your plugin’s registered name and enforces your contract. Declare what your plugin **requires** (must exist before you run) and what it will **produce** (must exist after you run):

```python
from dnadesign.opal.src.core.round_context import roundctx_contract

@roundctx_contract(
  category="objective",  # 'model' | 'objective' | 'selection' | 'transform_x' | 'transform_y'
  requires=["core/labels_as_of_round"],
  produces=["objective/<self>/foo"],
)
def my_objective_plugin(..., ctx=None, train_view=None): ...
```

**Important**: Plugins may always read context via `ctx.get(...)`, but must declare keys in `produces` to be able to write via `ctx.set(...)`. Writing to undeclared keys is rejected (contract error), which keeps runs auditable and deterministic.

`round_ctx.json` records exactly what each plugin read and wrote:

- `core/contracts/<category>/<plugin>/consumed` — keys read via ctx.get
- `core/contracts/<category>/<plugin>/produced` — keys written via ctx.set

#### Validation lifecycle

1. Runner builds `RoundCtx` with **core keys** and **plugin names**.
2. It creates **plugin-scoped contexts** (model, objective, selection, transform_x, y-ops) and checks **`requires`**.
3. Plugins run, using `ctx.get(...)` / `ctx.set(...)`.
   Reads/writes are **audited** into `core/contracts/...`.
4. Runner checks **`produces`**; on success, writes `round_ctx.json`.

Use `opal ctx show|audit|diff` to inspect these carriers directly.
Use `opal log --round <k|latest>` to summarize `round.log.jsonl`.

---

### Safety & validation

OPAL is **assertive by default**: it will fail fast on inconsistent inputs rather than guessing.

* `opal validate` checks essentials + X presence; if Y exists it must be finite and the expected length.
* `label_hist` is a **required input** for `run`/`explain`, but **ledger.labels is canonical** for auditability.
* Labels present in the Y column but **missing from `label_hist` are rejected** (use `opal ingest-y` or `opal label-hist attach-from-y` for legacy Y columns).
* Ledger writes are strict: unknown columns are **errors** (override only with `OPAL_LEDGER_ALLOW_EXTRA=1`).
* Duplicate handling on ingest is explicit via `ingest.duplicate_policy` (error | keep_first | keep_last).

### Data contracts

#### Records schema

**Required columns** in `records.parquet`:

| column     | type   | notes                  |
| ---------- | ------ | ---------------------- |
| `id`       | string | unique per record      |
| `bio_type` | string | `"dna"` or `"protein"` |
| `sequence` | string | case-insensitive       |
| `alphabet` | string | e.g. `dna_4`           |

**Recommended / commonly present** (not enforced by `validate`):

| column       | type            | notes           |
| ------------ | --------------- | --------------- |
| `length`     | int32           | `len(sequence)` |
| `source`     | string          | provenance      |
| `created_at` | timestamp (UTC) | ingest time     |

**X (representation):** Arrow `list<float>` or JSON array string; **fixed length** across used rows.
**Y (labels):** Arrow `list<float>`; label history is cached as JSON in `opal__<campaign>__label_hist`.

Naming: secondary columns follow `<tool>__<field>`.

**Records cache columns (OPAL‑managed)**

| column                              | type                    | purpose                                  |
| ----------------------------------- | ----------------------- | ---------------------------------------- |
| `opal__<slug>__label_hist`          | list<struct{r,ts,src,y}> | label history cache (ledger is canonical) |
| `opal__<slug>__latest_as_of_round`  | int                     | last scored round for each record         |
| `opal__<slug>__latest_pred_scalar`  | double                  | latest objective scalar cache             |
| `opal__<slug>__latest_pred_run_id`  | string                  | run_id for latest_pred_scalar (cache)     |
| `opal__<slug>__latest_pred_as_of_round` | int                 | as_of_round for latest_pred_scalar        |
| `opal__<slug>__latest_pred_written_at`  | string                | cache write timestamp (UTC ISO)           |

These caches are **derived** and can be recomputed; canonical records live in ledger sinks.
`opal init` will add these cache columns to `records.parquet` if they are missing.
Use `opal prune-source` to remove OPAL‑derived columns (including the Y column) when you need to start fresh.

#### Canonical vs cache vs transient (notebook)

* **Canonical (ledger)**: append-only, run-aware records under `outputs/ledger.*`.
* **Cache (records)**: `latest_pred_*` columns in `records.parquet` for convenience; can be stale or not run-aware.
* **Transient (notebook)**: in-memory overlays (e.g., RF surrogate) for exploration only; never persisted.
* **Y-ops gating**: notebook SFXI scoring only runs when predictions are in objective space (Y-ops inverse applied).

#### Ledger output schema (append-only)

Ledger sinks are the **canonical, append-only** record of what happened in a campaign. They are designed
for long-term inspection and downstream analysis; avoid treating `records.parquet` caches as the source of truth.

**labels (`outputs/ledger.labels.parquet`)**

* `event`: `"label"`
* `observed_round`, `id`, `sequence` (if available)
* `y_obs`: list<float> (canonical Y vector)
* `src`, `note`

**run_pred (`outputs/ledger.predictions/`)**

* `event`: `"run_pred"`, plus `run_id`, `as_of_round`, `id`, `sequence`
* `pred__y_dim`, `pred__y_hat_model` (list<float>, objective-space), `pred__y_obj_scalar`
* `sel__rank_competition`, `sel__is_selected`
* Optional row diagnostics under `obj__*` (e.g., `obj__logic_fidelity`, `obj__effect_raw`, `obj__effect_scaled`)

**run_meta (`outputs/ledger.runs.parquet`)**

* `event`: `"run_meta"`, plus `run_id`, `as_of_round`
* Config snapshot: `model__*`, `x_transform__*`, `y_ingest__*`, `objective__*`, `selection__*`, `training__y_ops`
* Counts + summaries: `stats__*`, `objective__summary_stats`, `objective__denom_*`
* Provenance: `artifacts` (paths + hashes), `schema__version`, `opal__version`

**Design notes (pragmatic)**

* Keep **row-level diagnostics** in `run_pred`, and **run-level summaries** in `run_meta` for clarity.
* Prefer **adding new columns** over changing semantics; prefixes (`pred__`, `obj__`, `sel__`, `model__`, `objective__`, `selection__`, `stats__`) keep intent explicit.
* Treat `schema__version` as the compatibility guardrail when evolving outputs.

---

## More documentation

Centralized OPAL docs live in `docs/`:

* [CLI manual](./docs/cli.md)
* [Plots](./docs/plots.md)
* [Models registry](./docs/models.md)
* [Selection strategies](./docs/selection.md)
* [X transforms](./docs/transforms-x.md)
* [Y transforms](./docs/transforms-y.md)
* [Setpoint fidelity x intensity](./docs/setpoint_fidelity_x_intensity.md)

## Demo campaign

See the **[Demo Guide](./docs/DEMO.md)** for a runnable example using `sfxi_v1` (vec8) with a Random Forest model.

---

@e-south

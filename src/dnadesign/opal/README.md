## OPAL — Optimization with Active Learning

**OPAL** is an [EVOLVEpro-style](https://www.science.org/doi/10.1126/science.adr6006) active-learning engine for DNA/protein sequence design.

- **Train** a top-layer regressor on your chosen representation **X** and label **Y**.
- **Score** every candidate sample with **X** present.
- **Reduce** each prediction Ŷ to a **single scalar** via your configured **objective** → `pred__y_obj_scalar`.
- **Rank** candidates by that score and **select top-k**.
- **Append** canonical events to **ledger sinks** under **`outputs/`** (per-round predictions + run metadata).
- **Persist** artifacts per round (model, selection CSV, round context, objective meta, logs) for auditability.

The pipeline is plugin-driven: swap **transforms** (X/Y), **models**, **objectives**, and **selection** strategies in `campaign.yaml` without touching core code.

> Using OPAL day-to-day? See the **[CLI Manual](./src/cli/README.md)**

---

### Contents

* [Quick install](#quick-install)
* [Quick start](#quick-start)
* [How OPAL is wired](#how-opal-is-wired)

  * [Code layout](#code-layout)
  * [CLI overview](#cli-overview)
* [Campaign layout](#campaign-layout)
* [Configuration (`campaign.yaml`)](#configuration)

  * [Key blocks](#key-blocks)
  * [Defaults](#defaults)
  * [Minimal example](#minimal-example)
  * [Notes on precedence & wiring](#notes-on-precedence--wiring)
* [Architecture & data flow](#architecture--data-flow)

  * [RunCarrier (RoundCtx)](#runcarrier-roundctx)
* [Safety & validation](#safety--validation)
* [Data contracts](#data-contracts)

  * [Records schema (required vs recommended)](#records-schema-required-vs-recommended)
* [Demo campaign](#demo-campaign)

---

### Quick install

OPAL lives inside the `dnadesign` repo. The CLI entrypoint is defined in `pyproject.toml`:

```toml
[project.scripts]
opal = "dnadesign.opal.src.cli.app:main"
```

Install in editable mode (from the repo root):

```bash
uv pip install -e .
# or: pip install -e .
```

---

### Quick start

```bash
# 1) Initialize a campaign workspace (creates inputs/, outputs/, state.json, and .opal/config marker)
opal init -c path/to/campaign.yaml

# 2) Validate essentials + X + (if present) Y shape/finite values
opal validate -c path/to/campaign.yaml

# 3) Ingest experimental labels (CSV/Parquet → Y), preview, then write
#    --round is the observed round stamp (alias: --observed-round)
opal ingest-y -c path/to/campaign.yaml --round 0 --csv path/to/my_labels.csv

# 4) Train on labels with observed_round ≤ R, score the pool, select top-k, persist artifacts & events
#    --round is the labels-as-of cutoff (alias: --labels-as-of)
opal run -c path/to/campaign.yaml --round 0

# 5) Inspect progress
opal status -c path/to/campaign.yaml
opal record-show -c path/to/campaign.yaml --id <some_id>
opal explain -c path/to/campaign.yaml --round 1
opal plot -c path/to/campaign.yaml
opal predict -c path/to/campaign.yaml --round latest  # or --model-path
opal objective-meta -c campaign.yaml --round latest

```

### What gets saved

* **Per-round**
  - `outputs/round_<k>/`
  - `model.joblib`
  - `model_meta.json`
  - `selection_top_k.csv`
  - `labels_used.parquet`
  - `round_ctx.json` *(runtime audit & fitted Y-ops)*
  - `objective_meta.json` *(objective mode/params/keys)*
  - `round.log.jsonl`

* **Campaign-wide ledger (append-only)**

  * `outputs/ledger.runs.parquet`
    - Plugin configs, counts, objective summaries, artifact hashes, versions.
  * `outputs/ledger.predictions/`
    - Ŷ vector, scalar score, selection rank/flag, and row-level diagnostics (e.g., logic fidelity/effects).
  * `outputs/ledger.labels.parquet`
    - 1 row per label event (observed round, id, y).

Schemas are **append-only**; keys are unique:
  `run_id` (runs), `(run_id,id)` (predictions), `(observed_round,id)` (labels).

---

## How OPAL is wired

At the top level, OPAL is a CLI (Typer) over an application layer with plugin registries.

### Code layout

```
src/dnadesign/opal/src/
├─ cli/                     # CLI app + command registry
│  ├─ app.py                # Typer entrypoint
│  ├─ registry.py           # auto-discovers and mounts commands
│  └─ commands/             # run, ingest_y, predict, explain, record_show, status, validate, init, plot
├─ config/                  # YAML loader + plugin param schemas (Pydantic)
├─ registries/              # transforms_x, transforms_y, models, objectives, selections, plot
├─ transforms_x/            # X transforms (import = register)
├─ transforms_y/            # Y ingests (import = register)
├─ models/                  # model wrappers (e.g., RandomForest)
├─ objectives/              # objective fns (Ŷ → scalar score + diagnostics)
├─ selection/               # selection strategies (scores → ranks/selected)
├─ artifacts.py             # round artifacts IO (ctx/meta/log/selection CSV, ledger sinks)
├─ writebacks.py            # canonical event builders + cache writers
├─ data_access.py           # RecordsStore (IO, label history, fixed-width X)
├─ round_context.py         # RoundCtx (runtime carrier)
└─ …
```

### CLI overview

Common commands (details in the **[CLI Manual](./src/cli/README.md)**):

* `opal init` — scaffold & register the campaign workspace; write `state.json`
* `opal ingest-y` — transform and append labels to `records.parquet`
* `opal run` — train/score/select for a round; write artifacts + ledger sinks under `outputs/`
* `opal predict` — score a table from a frozen model (no write-backs)
* `opal record-show` — per-record history view
* `opal status` — dashboard summary from `state.json`
* `opal explain` — dry-run planner (no writes)
* `opal validate` — table checks (essentials + X present; Y sane if present)
* `opal plot` — run campaign-declared plots

---

### Campaign layout

`opal init` scaffolds a campaign folder. Edit `campaign.yaml` to define behavior.

```
<repo>/src/dnadesign/opal/campaigns/<my_campaign>/
├─ campaign.yaml
├─ state.json
├─ inputs/                       # drop experimental label files here
└─ outputs/
   ├─ ledger.predictions/        # append-only run_pred parts
   ├─ ledger.runs.parquet         # run_meta (deduped)
   ├─ ledger.labels.parquet       # label events (ingest-only)
   └─ round_<k>/
      ├─ model.joblib
      ├─ model_meta.json
      ├─ selection_top_k.csv
      ├─ labels_used.parquet
      ├─ round_ctx.json
      ├─ objective_meta.json
      └─ round.log.jsonl
```

> **Note:** sequences usually live in USR datasets under
> `src/dnadesign/usr/datasets/<dataset>/records.parquet` and are not copied into campaigns.

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
* `ingest`: duplicate handling for label CSVs
* `scoring`: batch sizing
* `safety`: preflight/data guards
* `metadata`: optional notes

### Minimal example

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

### Architecture and data flow

* Models are not aware of downstream objectives (see **Runtime Carriers**)
* Objectives derive their own round constants via `train_view` and publish them.
* Selection can read whatever objectives produced.
* The persisted `round_ctx.json` makes runs **auditable** alongside ledger sinks, `model.joblib`, `model_meta.json`, `selection_top_k.csv`, and `objective_meta.json`.

```bash
# Labels
my_experimental_data.csv -> transforms_y -> labels [id, y(list<float>)]
  -> appends event: label

# Features
records.parquet [X] -> transforms_x -> X (fixed width)

# Train & score
X + labels -> model.fit -> predict Ŷ -> objective -> pred__y_obj_scalar -> selection (top-k)

# Canonical ledger sinks
outputs/ledger.*: { label | run_pred | run_meta }
```

---

### Runtime Carriers

**RunCarrier** is a runtime companion to `campaign.yaml`: the YAML picks which plugins to run; the carrier records what the run computed that others depend on. The runner persists a compact **`round.log.jsonl`** with stage events (preflight, y‑ops fit/transform, fit start/done, predict batches, y‑ops inverse, objective, selection) and a full **`round_ctx.json`** snapshot for audit.

#### How plugins use `RoundCtx`

The runner injects a **plugin-scoped context** (`ctx`) that auto-expands `"<self>"` to your plugin’s registered name and enforces your contract. Declare what your plugin **requires** (must exist before you run) and what it will **produce** (must exist after you run):

```python
from dnadesign.opal.src.round_context import roundctx_contract

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
2. It creates **plugin-scoped contexts** and checks **`requires`**.
3. Plugins run, using `ctx.get(...)` / `ctx.set(...)`.
   Reads/writes are **audited** into `core/contracts/...`.
4. Runner checks **`produces`**; on success, writes `round_ctx.json`.

---

### Safety & validation

OPAL is **assertive by default**: it will fail fast on inconsistent inputs rather than guessing.

* `opal validate` checks essentials + X presence; if Y exists it must be finite and the expected length.
* `label_hist` is the **single source of truth** for labels. `run`/`explain` require it to be valid.
* Labels present in the Y column but **missing from `label_hist` are rejected** (use `opal ingest-y` or `opal label-hist repair`).
* Ledger writes are strict: unknown columns are **errors** (override only with `OPAL_LEDGER_ALLOW_EXTRA=1`).
* Duplicate handling on ingest is explicit via `ingest.duplicate_policy` (error|keep_first|keep_last).

### Data contracts

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
**Y (labels):** Arrow `list<float>`; label history is append-only JSON in `opal__<campaign>__label_hist`.

Naming: secondary columns follow `<tool>__<field>`.

---

## Demo campaign

See the **[Demo Guide](./DEMO.md)** for a runnable example using `sfxi_v1` (vec8) with a Random Forest model.

---

@e-south

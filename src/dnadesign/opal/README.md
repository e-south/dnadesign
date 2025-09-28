# OPAL — Optimization with Active Learning

**OPAL** is an [EVOLVEpro-style](https://www.science.org/doi/10.1126/science.adr6006)  active-learning engine for DNA/protein sequence design.

- **Train** a top-layer regressor on your chosen representation **X** and label **Y**.
- **Score** every candidate sequence that isn’t labeled yet.
- **Reduce** each prediction Ŷ to a **single scalar score** via your configured **objective**.
- **Rank** candidates by that score and **select top-k**.
- **Append** a single event per `(round, id)` into **`outputs/events.parquet`** (score, rank, selection, uncertainty, flags, fingerprint).
- **Save artifacts** (model, selection csv, round context, objective meta, logs) for auditability and reproducibility.

The pipeline is plugin-driven: you can swap **transforms** (X/Y), **models**, **objectives**, and **selection** strategies in `campaign.yaml` without touching core code.

---

## Contents

- [Quick install](#quick-install)
- [Quick start](#quick-start)
- [How OPAL is wired](#how-opal-is-wired)
  - [Code layout](#code-layout)
  - [CLI overview](#cli-overview)
- [Campaign layout](#campaign-layout)
- [Configuration (`campaign.yaml`)](#configuration-campaignyaml)
  - [Key blocks](#key-blocks)
  - [Minimal example](#minimal-example)
- [Architecture & data flow](#architecture--data-flow)
  - [RoundContext (per-round state)](#roundcontext-per-round-state)
- [Data contracts](#data-contracts)
  - [Records schema (USR essentials)](#records-schema-usr-essentials)
- [Demo campaign](#demo-campaign)

---

## Quick install

OPAL lives inside the `dnadesign` repo. The CLI entrypoint is defined in `pyproject.toml`:

```toml
[project.scripts]
opal = "dnadesign.opal.src.cli.app:main"
````

Install in editable mode (from the repo root):

```bash
uv pip install -e .
# or: pip install -e .
```

---

## Quick start

```bash
# 1) Initialize a campaign workspace (creates inputs/, outputs/, state.json)
opal init -c path/to/campaign.yaml

# 2) Validate your data table has essentials + X column
opal validate -c path/to/campaign.yaml

# 3) Ingest experimental labels (CSV/Parquet → Y), preview, then write
opal ingest-y -c path/to/campaign.yaml --round 0 --csv path/to/my_labels.csv

# 4) Train on labels ≤ round, score the candidate pool, select top-k, persist artifacts
opal run -c path/to/campaign.yaml --round 0

# 5) Inspect progress
opal status -c path/to/campaign.yaml
opal record-show -c path/to/campaign.yaml --id <some_id>
opal explain -c path/to/campaign.yaml --round 1
```

After running a round, artifacts will appear in `outputs/round_0/`: 

* `model.joblib`
* `selection_top_k.csv`
* `round_ctx.json`
* `objective_meta.json`
* `round.log.jsonl`
* and an append-only table at `outputs/events.parquet`.

Per-round write-backs to `records.parquet`:

* `opal__<slug>__latest_round` (int)
* `opal__<slug>__latest_score` (float map per id)

---

## How OPAL is wired

At the top level, OPAL is a thin CLI (Typer) over an application layer with plugin registries.

### Code layout

```text
src/dnadesign/opal/src/
├─ cli/                     # CLI app + command registry
│  ├─ app.py                # Typer entrypoint
│  ├─ registry.py           # auto-discovers and mounts commands
│  └─ commands/             # run, ingest_y, predict, explain, record_show, status, validate, init
├─ config/                  # YAML loader + plugin param schemas (Pydantic)
├─ registries/              # transforms_x, transforms_y, models, objectives, selections
├─ transforms_x/            # concrete X transforms (import = register)
├─ transforms_y/            # concrete Y ingests (import = register)
├─ models/                  # model wrappers (e.g., RandomForest)
├─ objectives/              # objective functions (Ŷ → scalar score + diagnostics)
├─ selection/               # selection strategies (scores → ranks/selected)
├─ artifacts.py             # round artifacts (metrics, logs, ctx, CSVs)
├─ data_access.py           # RecordsStore: IO, label history, fixed-width X
├─ round_context.py         # RoundContext + fingerprinting helpers
├─ writebacks.py            # minimal per-row columns writer
└─ …
```

### CLI overview

Common commands (more details in `src/cli/README.md`):

* `opal init` — scaffold & register the campaign workspace; write `state.json`
* `opal ingest-y` — import experimental data from CSVs to a `records.parquet` file
* `opal run` — train/score/select for a given round; write artifacts and per-round columns
* `opal predict` — score a table from a frozen model (no write-backs)
* `opal record-show` — compact per-record history (ground truth + predictions, ranks, selected)
* `opal status` — dashboard summary from `state.json`
* `opal explain` — dry-run planner for a round (no writes)
* `opal validate` — end-to-end table checks (essentials present; X present)

---

## Campaign layout

`opal init` scaffolds a campaign folder. Edit `campaign.yaml` to define behavior.

```text
dnadesign/opal/
├─ src/                                     # OPAL code
├─ campaigns/
│   └─ <campaign_name>/
│       ├─ campaign.yaml                    # configuration (plugins + policies)
│       ├─ state.json                       # append-only campaign state across rounds
│       ├─ inputs/                          # drop experimental label files here
│       └─ outputs/
│           ├─ events.parquet               # append-only canonical events table
│           └─ round_<k>/
│               ├─ model.joblib
│               ├─ selection_top_k.csv
│               ├─ round_ctx.json
│               └─ round.log.jsonl
└─ README.md
```

> **Note:** sequences typically live in USR datasets under `src/dnadesign/usr/datasets/<dataset>/records.parquet` and are not copied into campaigns.

---

## Configuration (`campaign.yaml`)

`campaign.yaml` declares **what to plug in** (transforms, model, objective, selection) and **how to run** (policies, scoring performance).

### Key blocks

* `campaign`: `name`, `slug`, `workdir`
* `data`:

  * `location` (USR/local)
  * `x_column_name`
  * `y_column_name`
  * `y_expected_length` (optional, but recommended)
* `transforms_x`: `{ name, params }` (raw X → model-ready X)
* `transforms_y`: `{ name, params }` (tidy CSV → model-ready Y)
* `models`: `{ name, params }`
* `objectives`: `{ name, params }`
* `selection`: `{ name, params }` (strategy, tie handling)
* `training`: policy + target scaler config
* `scoring`: batch sizing (performance)

### Minimal example

```yaml
campaign:
  name: "My Campaign"
  slug: "my_campaign"
  workdir: "src/dnadesign/opal/campaigns/my_campaign_dir"

data:
  # USR datasets: provide root + dataset name
  location: { kind: usr, path: src/dnadesign/usr/datasets, dataset: my_dataset }
  # Local parquet (alternative):
  # location: { kind: local, path: /abs/path/to/records.parquet }
  x_column_name: "my_X_value"
  y_column_name: "my_label"

# X transform (raw -> model-ready X)
transforms_x: { name: identity, params: {} }

# Y ingest (tidy -> model-ready y)
transforms_y: { name: my_y_ingest_plugin, params: {} }

models:
  name: random_forest
  params:
    n_estimators: 100
    criterion: friedman_mse
    bootstrap: true
    oob_score: true
    random_state: 7
    n_jobs: -1

training:
  policy:
    cumulative_training: true
    label_cross_round_deduplication_policy: "latest_only"
    allow_resuggesting_candidates_until_labeled: true
  target_normalizer:
    enable: true
    minimum_labels_required: 5
    center_statistic: median
    scale_statistic: iqr

objectives:
  name: my_objective
  params: { ... }

selection:
  name: top_n
  params:
    top_k_default: 12
    tie_handling: competition_rank
    objective: maximize  # or: minimize

scoring:
  score_batch_size: 10000
```

**Target normalizer**

A per-target Y normalization applied only for model fitting, then inverted on prediction. Configure under training.target_normalizer (median/IQR by default).

**Objective scaling (per-round)**

A percentile-based effect scaling used by objectives (e.g., SFXI) to make scores comparable within a round. Lives under objectives.params.scaling.

---

## Architecture & data flow

Plugin-driven pipeline:

```
my_experimental_data.csv ──► transforms_y ──► labels[id,sequence,y]
records.parquet[X] ──► transforms_x ──► X (fixed width)
X + labels ──► model.fit
model + RoundContext ──► predict Ŷ ──► objective ──► score ──► selection
```

### RoundContext (per-round state)

`RoundContext` captures everything needed to **score** and **reproduce** a round without bloating `records.parquet`.

**Persisted fields (typical):**

* Identity: `slug`, `round_index`, `run_id`, `code_version`
* Config snapshot: plugin names + params, `y_expected_length`
* Setpoint & label pool: `setpoint`, `label_ids`, counts
* Objective scaling snapshot (if applicable): `effect_pool_for_scaling`, `percentile_cfg`
* Determinism: RNG seed(s)
* Fingerprint: short + full SHA-256 over the above

**Why it matters**

* Minimal per-row storage with maximal auditability
* Recompute scores/diagnostics offline using a frozen context
* Straightforward “explain this selection” workflows

---

## Data contracts

### Records schema (USR essentials)

Required columns (or created at ingest):

| column       | type            | notes                  |
| ------------ | --------------- | ---------------------- |
| `id`         | string          | unique per record      |
| `bio_type`   | string          | `"dna"` or `"protein"` |
| `sequence`   | string          | case-insensitive       |
| `alphabet`   | string          | e.g. `dna_4`           |
| `length`     | int32           | `len(sequence)`        |
| `source`     | string          | provenance             |
| `created_at` | timestamp (UTC) | ingest time            |

* **X (representation):** Arrow `list<float>` or JSON array string; **fixed length** across used rows.
* **Y (labels):** Arrow `list<float>`; label history is append-only JSON in `opal__<campaign>__label_hist` (also stores per-round prediction snapshots).

Naming: secondary columns follow `<tool>__<field>`.

---

## Demo campaign

See the dedicated [Demo Guide](./campaigns/demo/README.md) for a runnable OPAL example.

---

@e-south